/*
 * main_MotorDrive.h
 *
 *  Created on: Sep 10, 2025
 *      Author: kankanianapas
 */

#ifndef MAIN_MOTORDRIVE_H
#define MAIN_MOTORDRIVE_H

#include "stm32f4xx_hal.h"

// Function prototypes
void Motor_Init(TIM_HandleTypeDef *htim);
void Motor_Drive(int pwmVal);
void Motor_Stop(void);

#endif // MAIN_MOTORDRIVE_H
