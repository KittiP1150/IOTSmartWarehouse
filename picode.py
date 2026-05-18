import cv2
import numpy as np
import os
import time
import logging
import sys
import traceback
import json
import threading
import queue
from datetime import datetime
from pathlib import Path
from logging.handlers import RotatingFileHandler
 
import paho.mqtt.client as mqtt
 
# ============================================================
#  CONFIGURATION
# ============================================================
BASE_DIR          = Path("/home/pi/warehousedoor_ml")
FACES_DIR         = BASE_DIR / "faces"
DETECTION_MODEL   = BASE_DIR / "models/face_detection_yunet_2023mar.onnx"
RECOGNITION_MODEL = BASE_DIR / "models/face_recognition_sface_2021dec.onnx"
SNAP_PATH         = Path("/tmp/snap_temp.jpg")
LOG_DIR           = Path("/home/pi/warehousedoor_ml/logs")
 
MATCH_THRESHOLD        = 0.36
CONFIDENCE_THRESHOLD   = 0.8
SCAN_INTERVAL_SEC      = 5
CAMERA_TIMEOUT_SEC     = 2.0
CAMERA_WIDTH           = 640
CAMERA_HEIGHT          = 480
MAX_CONSECUTIVE_ERRORS = 10
 
# MQTT
MQTT_BROKER     = "broker.netpie.io"
MQTT_PORT       = 1883
MQTT_CLIENT_ID  = ""   # Secret
MQTT_TOKEN      = ""   # Secret
MQTT_SECRET     = ""   # Secret
PUBLISH_TOPIC   = "@msg/Data"
SUBSCRIBE_TOPIC = "@msg/Data"
 
# ============================================================
#  LOGGING SETUP
# ============================================================
LOG_DIR.mkdir(parents=True, exist_ok=True)
 
log_formatter = logging.Formatter(
    fmt="%(asctime)s [%(levelname)s] %(threadName)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
 
file_handler = RotatingFileHandler(
    LOG_DIR / "access_control.log",
    maxBytes=5 * 1024 * 1024, backupCount=5, encoding="utf-8"
)
file_handler.setFormatter(log_formatter)
 
error_handler = RotatingFileHandler(
    LOG_DIR / "errors.log",
    maxBytes=2 * 1024 * 1024, backupCount=3, encoding="utf-8"
)
error_handler.setLevel(logging.ERROR)
error_handler.setFormatter(log_formatter)
 
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(log_formatter)
 
logger = logging.getLogger("FaceAccess")
logger.setLevel(logging.DEBUG)
logger.addHandler(file_handler)
logger.addHandler(error_handler)
logger.addHandler(console_handler)
 
# ============================================================
#  SHARED STATE
# ============================================================
# scan_cycle() puts results here; MqttWorker drains and publishes them.
# maxsize=10: if MQTT is slow, the scan loop is never blocked.
publish_queue: queue.Queue = queue.Queue(maxsize=10)
 
# Signals all threads to stop cleanly.
shutdown_event = threading.Event()
 
 
# ============================================================
#  MQTT WORKER
# ============================================================
class MqttWorker(threading.Thread):
    """
    Dedicated thread for all MQTT I/O.
    - Never blocks the scan loop.
    - Auto-reconnects on unexpected disconnect.
    - Re-subscribes inside on_connect so it survives reconnects.
    """
 
    def __init__(self):
        super().__init__(name="MqttWorker", daemon=True)
        self.client = mqtt.Client(
            protocol=mqtt.MQTTv311,
            client_id=MQTT_CLIENT_ID,
            clean_session=False
        )
        self.client.username_pw_set(MQTT_TOKEN, MQTT_SECRET)
        self.client.on_connect    = self._on_connect
        self.client.on_disconnect = self._on_disconnect
        self.client.on_message    = self._on_message
        self.connected = threading.Event()
 
    def _on_connect(self, client, userdata, flags, rc):
        if rc == 0:
            logger.info(f"MQTT connected to {MQTT_BROKER}:{MQTT_PORT}")
            self.connected.set()
            # Subscribe must live here so it runs again after every reconnect.
            client.subscribe(SUBSCRIBE_TOPIC)
            logger.info(f"MQTT subscribed: {SUBSCRIBE_TOPIC}")
        else:
            logger.error(f"MQTT connect failed, rc={rc}")
            self.connected.clear()
 
    def _on_disconnect(self, client, userdata, rc):
        self.connected.clear()
        if rc != 0:
            logger.warning(f"MQTT unexpected disconnect (rc={rc}), will auto-reconnect")
 
    def _on_message(self, client, userdata, msg):
        try:
            payload = msg.payload.decode("utf-8")
            logger.info(f"MQTT received [{msg.topic}]: {payload[:200]}")
        except Exception as e:
            logger.warning(f"MQTT message decode error: {e}")
 
    def publish(self, data: dict) -> bool:
        """Serialize data to JSON and publish. Returns True on success."""
        if not self.connected.is_set():
            logger.warning("MQTT not connected, skipping publish")
            return False
        try:
            payload = json.dumps(data, ensure_ascii=False)
            info = self.client.publish(PUBLISH_TOPIC, payload, retain=True)
            info.wait_for_publish(timeout=3.0)
            logger.info(f"MQTT published: {payload[:200]}")
            return True
        except Exception as e:
            logger.error(f"MQTT publish error: {e}")
            return False
 
    def run(self):
        """Connect, start paho network loop, then drain the publish queue."""
        logger.info("MqttWorker starting...")
        try:
            self.client.connect(MQTT_BROKER, MQTT_PORT, keepalive=60)
            self.client.loop_start()
        except Exception as e:
            logger.error(f"MQTT initial connect failed: {e}")
 
        while not shutdown_event.is_set():
            try:
                # Block with a timeout so shutdown_event is checked regularly.
                data = publish_queue.get(timeout=1.0)
                self.publish(data)
                publish_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"MqttWorker error: {e}")
 
        logger.info("MqttWorker shutting down...")
        self.client.loop_stop()
        self.client.disconnect()
 
 
# ============================================================
#  STARTUP DIAGNOSTICS
# ============================================================
def run_startup_checks() -> bool:
    """Verify all dependencies before the main loop starts."""
    logger.info("=" * 50)
    logger.info("SYSTEM STARTUP - Face Access Control")
    logger.info(f"OpenCV version : {cv2.__version__}")
    logger.info(f"Python version : {sys.version.split()[0]}")
    logger.info(f"Working dir    : {Path.cwd()}")
    logger.info("=" * 50)
 
    all_ok = True
 
    for label, path in [("Detection model", DETECTION_MODEL),
                         ("Recognition model", RECOGNITION_MODEL)]:
        if path.exists():
            logger.info(f"{label}: {path.name} ({path.stat().st_size // 1024} KB)")
        else:
            logger.error(f"{label} NOT FOUND: {path}")
            all_ok = False
 
    if FACES_DIR.exists():
        logger.info(f"Faces dir: {len(list(FACES_DIR.iterdir()))} files found")
    else:
        logger.warning(f"Faces dir missing, creating: {FACES_DIR}")
        FACES_DIR.mkdir(parents=True, exist_ok=True)
 
    logger.info("Checking camera...")
    if _test_camera():
        logger.info("Camera: OK")
    else:
        logger.error("Camera: FAILED")
        all_ok = False
 
    return all_ok
 
 
def _test_camera() -> bool:
    """Fire a short test shot to confirm the camera is reachable."""
    test_path = Path("/tmp/cam_test.jpg")
    ret = os.system(
        f"rpicam-still -o {test_path} -t 500 "
        f"--width 320 --height 240 --nopreview 2>/dev/null"
    )
    ok = (ret == 0) and test_path.exists() and test_path.stat().st_size > 0
    if test_path.exists():
        test_path.unlink()
    return ok
 
 
# ============================================================
#  MODEL INITIALIZATION
# ============================================================
def init_models():
    """Load detection and recognition models. Raises on failure."""
    logger.info("Loading AI models...")
    try:
        detector = cv2.FaceDetectorYN.create(
            str(DETECTION_MODEL), "", (320, 320),
            CONFIDENCE_THRESHOLD, 0.3, 5000
        )
        recognizer = cv2.FaceRecognizerSF.create(str(RECOGNITION_MODEL), "")
        logger.info("AI models loaded successfully")
        return detector, recognizer
    except cv2.error as e:
        logger.error(f"OpenCV model error: {e}")
        raise
 
 
# ============================================================
#  FACE DATABASE
# ============================================================
def get_embedding(image, detector, recognizer):
    """Extract a face embedding from an image. Returns (embedding, reason)."""
    h, w = image.shape[:2]
    detector.setInputSize((w, h))
    _, faces = detector.detect(image)
    if faces is None or len(faces) == 0:
        return None, "no_face_detected"
    if len(faces) > 1:
        # Keep the highest-confidence detection.
        faces = sorted(faces, key=lambda f: f[14], reverse=True)
    try:
        aligned = recognizer.alignCrop(image, faces[0])
        return recognizer.feature(aligned), "ok"
    except cv2.error as e:
        return None, f"align_error:{e}"
 
 
def load_known_faces(detector, recognizer) -> dict:
    """Load all face images from FACES_DIR and build an embedding database."""
    known_faces = {}
    supported_ext = {".jpg", ".jpeg", ".png", ".bmp"}
    face_files = [f for f in FACES_DIR.iterdir() if f.suffix.lower() in supported_ext]
 
    if not face_files:
        logger.warning("No face images found in faces directory")
        return known_faces
 
    logger.info(f"Loading {len(face_files)} face image(s)...")
    for path in face_files:
        try:
            img = cv2.imread(str(path))
            if img is None:
                logger.warning(f"Cannot read image: {path.name}")
                continue
            h, w = img.shape[:2]
            emb, reason = get_embedding(img, detector, recognizer)
            if emb is not None:
                name = path.stem.capitalize()
                known_faces[name] = emb
                logger.info(f"Loaded: {path.name} -> '{name}' ({w}x{h})")
            else:
                logger.warning(f"Skipped {path.name}: {reason}")
        except Exception as e:
            logger.error(f"Error loading {path.name}: {e}")
            logger.debug(traceback.format_exc())
 
    logger.info(f"Face database: {len(known_faces)}/{len(face_files)} loaded")
    return known_faces
 
 
# ============================================================
#  CAMERA CAPTURE
# ============================================================
def capture_image():
    """
    Take a still photo and return (frame, error_reason).
    Possible error values: camera_exit_N / file_missing_timeout /
                           file_too_small_Nb / read_fail
    """
    if SNAP_PATH.exists():
        SNAP_PATH.unlink()
 
    ret = os.system(
        f"rpicam-still -o {SNAP_PATH} -t 1000 "
        f"--width {CAMERA_WIDTH} --height {CAMERA_HEIGHT} "
        f"--nopreview 2>/tmp/rpicam_err.log"
    )
    if ret != 0:
        err = ""
        try:
            err = Path("/tmp/rpicam_err.log").read_text().strip()[-200:]
        except Exception:
            pass
        logger.error(f"Camera command failed (exit={ret}): {err}")
        return None, f"camera_exit_{ret}"
 
    deadline = time.time() + CAMERA_TIMEOUT_SEC
    while not SNAP_PATH.exists():
        if time.time() > deadline:
            logger.error("Camera timeout: file never appeared")
            return None, "file_missing_timeout"
        time.sleep(0.05)
 
    size = SNAP_PATH.stat().st_size
    if size < 1024:
        logger.error(f"Captured file suspiciously small: {size} bytes")
        return None, f"file_too_small_{size}b"
 
    frame = cv2.imread(str(SNAP_PATH))
    if frame is None:
        logger.error(f"cv2.imread failed on {SNAP_PATH}")
        return None, "read_fail"
 
    return frame, "ok"
 
 
# ============================================================
#  FACE RECOGNITION
# ============================================================
def identify_face(feature, known_faces, recognizer):
    """Match a feature vector against the known face database."""
    best_score, best_name = 0.0, "Unknown"
    for name, known_emb in known_faces.items():
        try:
            score = recognizer.match(feature, known_emb, cv2.FaceRecognizerSF_FR_COSINE)
            if score > best_score:
                best_score, best_name = score, name
        except cv2.error as e:
            logger.warning(f"Match error for '{name}': {e}")
    return best_name, best_score
 
 
def draw_result(frame, face, name, score, granted):
    """Draw a bounding box and label on the frame."""
    box = face[0:4].astype(int)
    x, y, w, h = box
    color = (0, 255, 0) if granted else (0, 0, 255)
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
    cv2.putText(frame, f"{name} ({int(score * 100)}%)",
                (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
 
 
# ============================================================
#  SCAN CYCLE
# ============================================================
def scan_cycle(detector, recognizer, known_faces: dict) -> dict:
    """
    Run one full capture-detect-recognize cycle.
    Returns a result dict consumed by ScanWorker._build_mqtt_payload().
    """
    cycle_id = datetime.now().strftime("%H%M%S%f")[:9]
    result = {
        "cycle_id": cycle_id,
        "timestamp": datetime.now().isoformat(),
        "status": "unknown",
        "faces_detected": 0,
        "access_events": []
    }
 
    frame, err = capture_image()
    if frame is None:
        result.update({"status": "capture_failed", "error": err})
        logger.warning(f"[{cycle_id}] Capture failed: {err}")
        return result
 
    file_kb = SNAP_PATH.stat().st_size // 1024
    logger.debug(f"[{cycle_id}] Image captured: {file_kb} KB")
 
    h, w = frame.shape[:2]
    detector.setInputSize((w, h))
    _, faces = detector.detect(frame)
 
    if faces is None or len(faces) == 0:
        result["status"] = "no_face"
        logger.info(f"[{cycle_id}] No face detected")
        return result
 
    result["faces_detected"] = len(faces)
    logger.info(f"[{cycle_id}] Detected {len(faces)} face(s)")
 
    for i, face in enumerate(faces):
        confidence = float(face[14])
        try:
            aligned = recognizer.alignCrop(frame, face)
            feature = recognizer.feature(aligned)
        except cv2.error as e:
            logger.warning(f"[{cycle_id}] Face[{i}] align failed: {e}")
            continue
 
        if not known_faces:
            logger.warning(f"[{cycle_id}] Empty face database - denying all")
            name, score, granted = "Unknown", 0.0, False
        else:
            name, score = identify_face(feature, known_faces, recognizer)
            granted = score > MATCH_THRESHOLD
 
        event = {
            "face_index": i,
            "detection_confidence": round(confidence, 3),
            "name": name if granted else "Unknown",
            "match_score": round(score, 3),
            "granted": granted
        }
        result["access_events"].append(event)
 
        verdict = "ACCESS GRANTED" if granted else "ACCESS DENIED"
        logger.info(
            f"[{cycle_id}] {verdict} | {name} | score={score:.2f} | conf={confidence:.2f}"
        )
        draw_result(frame, face, name if granted else "Unknown", score, granted)
 
    result["status"] = "ok"
    cv2.imwrite(str(LOG_DIR / "last_result.jpg"), frame)
 
    try:
        cv2.imshow("Scanner Result", frame)
        cv2.waitKey(2000)
    except cv2.error:
        pass  # Headless environment - not an error.
 
    return result
 
 
# ============================================================
#  SCAN WORKER
# ============================================================
class ScanWorker(threading.Thread):
    """
    Dedicated thread for the scan loop.
    Puts results onto publish_queue after each cycle.
    """
 
    def __init__(self, detector, recognizer, known_faces):
        super().__init__(name="ScanWorker", daemon=True)
        self.detector           = detector
        self.recognizer         = recognizer
        self.known_faces        = known_faces
        self.cycle_count        = 0
        self.consecutive_errors = 0
 
    def run(self):
        logger.info("ScanWorker starting...")
 
        while not shutdown_event.is_set():
            self.cycle_count += 1
            logger.info(f"--- SCAN CYCLE #{self.cycle_count} ---")
 
            # Countdown, checking shutdown_event every second.
            for i in range(SCAN_INTERVAL_SEC, 0, -1):
                if shutdown_event.is_set():
                    return
                logger.debug(f"Scanning in {i}s...")
                time.sleep(1)
 
            try:
                result = scan_cycle(self.detector, self.recognizer, self.known_faces)
 
                if result["status"] == "capture_failed":
                    self.consecutive_errors += 1
                    logger.warning(
                        f"Consecutive hardware errors: "
                        f"{self.consecutive_errors}/{MAX_CONSECUTIVE_ERRORS}"
                    )
                    if self.consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
                        logger.critical(
                            "Too many consecutive hardware errors - requesting shutdown"
                        )
                        shutdown_event.set()
                        sys.exit(2)
                else:
                    self.consecutive_errors = 0
 
                payload = self._build_mqtt_payload(result)
                try:
                    publish_queue.put_nowait(payload)
                except queue.Full:
                    logger.warning("Publish queue full - dropping oldest entry")
                    try:
                        publish_queue.get_nowait()
                        publish_queue.put_nowait(payload)
                    except queue.Empty:
                        pass
 
            except Exception as e:
                logger.error(f"ScanWorker unhandled error in cycle #{self.cycle_count}: {e}")
                logger.error(traceback.format_exc())
                self.consecutive_errors += 1
                time.sleep(2)
 
        logger.info("ScanWorker stopped.")
 
    def _build_mqtt_payload(self, result: dict) -> dict:
        """
        Convert a scan_cycle result dict into a flat NETPIE publish payload.
        If at least one face was granted, the first granted event is used.
        Otherwise access is reported as denied.
        """
        base = {
            "timestamp":      result["timestamp"],
            "cycle_id":       result["cycle_id"],
            "faces_detected": result["faces_detected"],
            "status":         result["status"],
        }
 
        granted_events = [e for e in result.get("access_events", []) if e["granted"]]
 
        if granted_events:
            ev = granted_events[0]
            base.update({
                "access":               "granted",
                "name":                 ev["name"],
                "match_score":          ev["match_score"],
                "detection_confidence": ev["detection_confidence"],
            })
        else:
            base.update({
                "access":               "denied",
                "name":                 "Unknown",
                "match_score":          0.0,
                "detection_confidence": 0.0,
            })
 
        return base
 
 
# ============================================================
#  ENTRY POINT
# ============================================================
def main():
    if not run_startup_checks():
        logger.critical("Startup checks FAILED. Exiting.")
        sys.exit(1)
 
    try:
        detector, recognizer = init_models()
    except Exception as e:
        logger.critical(f"Cannot load models: {e}")
        sys.exit(1)
 
    known_faces = load_known_faces(detector, recognizer)
    if not known_faces:
        logger.warning("No faces loaded - system will deny all access.")
 
    logger.info(f"Known faces: {list(known_faces.keys())}")
 
    # Start MQTT thread first so the broker connection is attempted early.
    mqtt_worker = MqttWorker()
    mqtt_worker.start()
 
    logger.info("Waiting for MQTT connection (max 5s)...")
    mqtt_worker.connected.wait(timeout=5.0)
    if not mqtt_worker.connected.is_set():
        logger.warning("MQTT not connected yet - scanning will proceed, publishes will queue")
 
    scan_worker = ScanWorker(detector, recognizer, known_faces)
    scan_worker.start()
 
    # Main thread only waits for a shutdown signal.
    try:
        while not shutdown_event.is_set():
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Ctrl+C received - shutting down...")
        shutdown_event.set()
 
    scan_worker.join(timeout=10)
    publish_queue.join()  # Wait for any queued publishes to drain.
    logger.info("System shutdown complete.")
    cv2.destroyAllWindows()
 
 
if __name__ == "__main__":
    main()