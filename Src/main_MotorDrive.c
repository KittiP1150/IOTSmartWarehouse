/*
 * main_MotorDrive.c
 *
 *  Created on: Sep 10, 2025
 *      Author: kankanianapas
 */

#include "main_MotorDrive.h"

// ===== Define motor pins (edit if needed) ===== //
#define MOTOR_IN1_PORT     GPIOB
#define MOTOR_IN1_PIN      GPIO_PIN_10    // IN1 → PB10

#define MOTOR_IN2_PORT     GPIOB
#define MOTOR_IN2_PIN      GPIO_PIN_4     // IN2 → PA8

// ===== PWM channel ===== //
#define MOTOR_PWM_CHANNEL  TIM_CHANNEL_1  // TIM3_CH2 (PB5, D4)

// Timer handle (set from main.c)
static TIM_HandleTypeDef *htim_motor;

void Motor_Init(TIM_HandleTypeDef *htim)
{
    htim_motor = htim;

    // Start PWM channel
    HAL_TIM_PWM_Start(htim_motor, MOTOR_PWM_CHANNEL);

    // Ensure motor is stopped on boot
    Motor_Stop();
}

/*
 * pwmVal rules:
 *  >0  → forward
 *  <0  → backward
 *   0  → stop
 */


void Motor_Drive(int pwmVal)
{
    if (pwmVal > 0)
    {
        // Forward direction
        HAL_GPIO_WritePin(MOTOR_IN1_PORT, MOTOR_IN1_PIN, GPIO_PIN_SET);
        HAL_GPIO_WritePin(MOTOR_IN2_PORT, MOTOR_IN2_PIN, GPIO_PIN_RESET);

        __HAL_TIM_SET_COMPARE(htim_motor, MOTOR_PWM_CHANNEL, pwmVal);
    }
    else if (pwmVal < 0)
    {
        // Backward direction
        HAL_GPIO_WritePin(MOTOR_IN1_PORT, MOTOR_IN1_PIN, GPIO_PIN_RESET);
        HAL_GPIO_WritePin(MOTOR_IN2_PORT, MOTOR_IN2_PIN, GPIO_PIN_SET);

        __HAL_TIM_SET_COMPARE(htim_motor, MOTOR_PWM_CHANNEL, -pwmVal);
    }
    else
    {
        Motor_Stop();
    }
}

void Motor_Stop(void)
{
    HAL_GPIO_WritePin(MOTOR_IN1_PORT, MOTOR_IN1_PIN, GPIO_PIN_SET);
    HAL_GPIO_WritePin(MOTOR_IN2_PORT, MOTOR_IN2_PIN, GPIO_PIN_SET);

    __HAL_TIM_SET_COMPARE(htim_motor, MOTOR_PWM_CHANNEL, 0);
}
