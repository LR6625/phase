// Dual Stepper Motor Control with DRV8825 + TT Motor with L298N + MG90S Servos
// Stepper Motor 1:
//   DIR pin: 5
//   STEP pin: 4
// Stepper Motor 2:
//   DIR pin: 7
//   STEP pin: 6
// TT Motor (L298N):
//   ENA pin: 10 (PWM)
//   IN1 pin: 9
//   IN2 pin: 8
// 
// Serial commands:
//   SPIN_CW:M1:<steps>   - Rotate motor 1 clockwise
//   SPIN_CCW:M1:<steps>  - Rotate motor 1 counter-clockwise
//   SPIN_CW:M2:<steps>   - Rotate motor 2 clockwise
//   SPIN_CCW:M2:<steps>  - Rotate motor 2 counter-clockwise
//   SPIN_CW:BOTH:<steps> - Rotate both motors clockwise
//   SPIN_CCW:BOTH:<steps> - Rotate both motors counter-clockwise
//   SPIN_UP_TO_SWITCH:M1[:<maxSteps>] - Rotate motor 1 CCW until limit switch is pressed
//   TT_FORWARD:<SLOW|MEDIUM|FAST> - Run TT motor forward at specified speed
//   TT_REVERSE:<SLOW|MEDIUM|FAST> - Run TT motor reverse at specified speed
//   TT_STOP - Stop TT motor
//   GET_SWITCH - Query limit switch state (returns PRESSED or RELEASED)
//   SLEEP:<milliseconds> - Sleep
//   SERVO1:<angle> - Move servo 1 by signed offset from center (negative = down)
//   SERVO2:<angle> - Move servo 2 by signed offset from center (negative = down)
//   SERVO1_ABS:<angle> - Move servo 1 to absolute angle (0-180)
//   SERVO2_ABS:<angle> - Move servo 2 to absolute angle (0-180)
//   SERVO:<angle> - Alias for SERVO1_ABS:<angle>

#include <Servo.h>

// Stepper Motor 1 pins
const int M1_DIR_PIN = 5;
const int M1_STEP_PIN = 4;

// Stepper Motor 2 pins
const int M2_STEP_PIN = 6;
const int M2_DIR_PIN = 7;

// TT Motor pins (L298N)
const int TT_ENA_PIN = 3;
const int TT_IN1_PIN = 9;
const int TT_IN2_PIN = 8;

// TT Motor PWM speeds
const int TT_SPEED_SLOW = 100;
const int TT_SPEED_MEDIUM = 170;
const int TT_SPEED_FAST = 255;

// Servos
const int SERVO1_PIN = 11;
const int SERVO2_PIN = 12;
const int SERVO_MIN_ANGLE = 0;
const int SERVO_MAX_ANGLE = 180;
const int SERVO_CENTER_ANGLE = 90;
Servo mg90sServo1;
Servo mg90sServo2;

const int LIMIT_SWITCH_PIN = 13;
const long DEFAULT_MAX_STEPS = 200000;

const unsigned int STEP_PULSE_US_M1 = 250;  // Motor 1
const unsigned int STEP_PULSE_US_M2 = 500;  // Motor 2

String readLine() {
    String line = Serial.readStringUntil('\n');
    line.trim();
    return line;
}

void setup() {
    // Stepper Motor 1 pins
    pinMode(M1_DIR_PIN, OUTPUT);
    pinMode(M1_STEP_PIN, OUTPUT);
    
    // Stepper Motor 2 pins
    pinMode(M2_DIR_PIN, OUTPUT);
    pinMode(M2_STEP_PIN, OUTPUT);

    // TT Motor pins
    pinMode(TT_ENA_PIN, OUTPUT);
    pinMode(TT_IN1_PIN, OUTPUT);
    pinMode(TT_IN2_PIN, OUTPUT);

    pinMode(LIMIT_SWITCH_PIN, INPUT_PULLUP);
    
    // Force all pins low at startup (safe state)
    digitalWrite(M1_DIR_PIN, LOW);
    digitalWrite(M1_STEP_PIN, LOW);
    digitalWrite(M2_DIR_PIN, LOW);
    digitalWrite(M2_STEP_PIN, LOW);
    
    digitalWrite(TT_IN1_PIN, LOW);
    digitalWrite(TT_IN2_PIN, LOW);
    analogWrite(TT_ENA_PIN, 0);

    mg90sServo1.attach(SERVO1_PIN);
    mg90sServo2.attach(SERVO2_PIN);
    mg90sServo1.write(SERVO_CENTER_ANGLE);
    mg90sServo2.write(SERVO_CENTER_ANGLE);
    
    Serial.begin(9600);
    // Don't print startup message - it can pollute the serial buffer
}

void loop() {
    if (Serial.available() > 0) {
        String command = readLine();
        
        // Handle SPIN_CW commands
        if (command.startsWith("SPIN_CW:")) {
            int firstColon = command.indexOf(':');
            int secondColon = command.indexOf(':', firstColon + 1);
            
            if (secondColon == -1) {
                // Legacy format: SPIN_CW:steps (default to M1)
                long steps = command.substring(firstColon + 1).toInt();
                if (steps > 0) rotateMotor1(steps, true);
            } else {
                // New format: SPIN_CW:M1:steps
                String motor = command.substring(firstColon + 1, secondColon);
                long steps = command.substring(secondColon + 1).toInt();
                
                if (steps > 0) {
                    if (motor == "M1") {
                        rotateMotor1(steps, true);
                    } else if (motor == "M2") {
                        rotateMotor2(steps, true);
                    } else if (motor == "BOTH") {
                        rotateBothMotors(steps, true);
                    }
                }
            }
            Serial.println("OK");
        }
        // Handle SPIN_CCW commands
        else if (command.startsWith("SPIN_CCW:")) {
            int firstColon = command.indexOf(':');
            int secondColon = command.indexOf(':', firstColon + 1);
            
            if (secondColon == -1) {
                // Legacy format: SPIN_CCW:steps (default to M1)
                long steps = command.substring(firstColon + 1).toInt();
                if (steps > 0) rotateMotor1(steps, false);
            } else {
                // New format: SPIN_CCW:M1:steps
                String motor = command.substring(firstColon + 1, secondColon);
                long steps = command.substring(secondColon + 1).toInt();
                
                if (steps > 0) {
                    if (motor == "M1") {
                        rotateMotor1(steps, false);
                    } else if (motor == "M2") {
                        rotateMotor2(steps, false);
                    } else if (motor == "BOTH") {
                        rotateBothMotors(steps, false);
                    }
                }
            }
            Serial.println("OK");
        }
        // Handle SPIN_UP_TO_SWITCH commands
        else if (command.startsWith("SPIN_UP_TO_SWITCH:")) {
            int firstColon = command.indexOf(':');
            int secondColon = command.indexOf(':', firstColon + 1);

            String motor;
            long maxSteps = DEFAULT_MAX_STEPS;

            if (secondColon == -1) {
                motor = command.substring(firstColon + 1);
            } else {
                motor = command.substring(firstColon + 1, secondColon);
                maxSteps = command.substring(secondColon + 1).toInt();
                if (maxSteps <= 0) maxSteps = DEFAULT_MAX_STEPS;
            }

            if (motor == "M1") {
                bool hit = rotateMotor1UntilSwitch(maxSteps);
                Serial.println(hit ? "OK" : "ERR:LIMIT");
                Serial.flush();  // Ensure response is sent immediately
            } else {
                Serial.println("ERR:UNKNOWN");
                Serial.flush();
            }
        }
        // Handle SPIN_DOWN_TO_SWITCH commands
        else if (command.startsWith("SPIN_DOWN_TO_SWITCH:")) {
            int firstColon = command.indexOf(':');
            int secondColon = command.indexOf(':', firstColon + 1);

            String motor;
            long maxSteps = DEFAULT_MAX_STEPS;

            if (secondColon == -1) {
                motor = command.substring(firstColon + 1);
            } else {
                motor = command.substring(firstColon + 1, secondColon);
                maxSteps = command.substring(secondColon + 1).toInt();
                if (maxSteps <= 0) maxSteps = DEFAULT_MAX_STEPS;
            }

            if (motor == "M1") {
                bool hit = rotateMotor1UntilSwitchClockwise(maxSteps);
                Serial.println(hit ? "OK" : "ERR:LIMIT");
                Serial.flush();  // Ensure response is sent immediately
            } else {
                Serial.println("ERR:UNKNOWN");
                Serial.flush();
            }
        }
        // Handle SLEEP commands
        else if (command.startsWith("SLEEP:")) {
            long ms = command.substring(6).toInt();
            if (ms > 0) delay(ms);
            Serial.println("OK");
        }
        // Handle TT_FORWARD commands
        else if (command.startsWith("TT_FORWARD:")) {
            String speed = command.substring(11);
            speed.trim();
            int pwmValue = TT_SPEED_MEDIUM;
            
            if (speed == "SLOW") {
                pwmValue = TT_SPEED_SLOW;
            } else if (speed == "MEDIUM") {
                pwmValue = TT_SPEED_MEDIUM;
            } else if (speed == "FAST") {
                pwmValue = TT_SPEED_FAST;
            }
            
            digitalWrite(TT_IN1_PIN, HIGH);
            digitalWrite(TT_IN2_PIN, LOW);
            analogWrite(TT_ENA_PIN, pwmValue);
            Serial.println("OK");
        }
        // Handle TT_REVERSE commands
        else if (command.startsWith("TT_REVERSE:")) {
            String speed = command.substring(11);
            speed.trim();
            int pwmValue = TT_SPEED_MEDIUM;
            
            if (speed == "SLOW") {
                pwmValue = TT_SPEED_SLOW;
            } else if (speed == "MEDIUM") {
                pwmValue = TT_SPEED_MEDIUM;
            } else if (speed == "FAST") {
                pwmValue = TT_SPEED_FAST;
            }
            
            digitalWrite(TT_IN1_PIN, LOW);
            digitalWrite(TT_IN2_PIN, HIGH);
            analogWrite(TT_ENA_PIN, pwmValue);
            Serial.println("OK");
        }
        // Handle TT_STOP commands
        else if (command == "TT_STOP") {
            digitalWrite(TT_IN1_PIN, LOW);
            digitalWrite(TT_IN2_PIN, LOW);
            analogWrite(TT_ENA_PIN, 0);
            Serial.println("OK");
        }
        // Handle GET_SWITCH commands
        else if (command == "GET_SWITCH") {
            if (digitalRead(LIMIT_SWITCH_PIN) == LOW) {
                Serial.println("PRESSED");
            } else {
                Serial.println("RELEASED");
            }
        }
        // Handle SERVO commands
        else if (command.startsWith("SERVO1_ABS:")) {
            int angle = command.substring(11).toInt();
            mg90sServo1.write(clampServoAngle(angle));
            Serial.println("OK");
        }
        else if (command.startsWith("SERVO2_ABS:")) {
            int angle = command.substring(11).toInt();
            mg90sServo2.write(clampServoAngle(angle));
            Serial.println("OK");
        }
        else if (command.startsWith("SERVO1:")) {
            int offset = command.substring(7).toInt();
            mg90sServo1.write(relativeServoAngle(offset));
            Serial.println("OK");
        }
        else if (command.startsWith("SERVO2:")) {
            int offset = command.substring(7).toInt();
            mg90sServo2.write(relativeServoAngle(offset));
            Serial.println("OK");
        }
        else if (command.startsWith("SERVO:")) {
            int angle = command.substring(6).toInt();
            mg90sServo1.write(clampServoAngle(angle));
            Serial.println("OK");
        }
        // Unknown command
        else if (command.length() > 0) {
            Serial.println("ERR:UNKNOWN");
        }
    }
}

void rotateMotor1(long steps, bool clockwise) {
    if (steps < 0) steps = -steps;
    digitalWrite(M1_DIR_PIN, clockwise ? HIGH : LOW);

    for (long i = 0; i < steps; i++) {
        digitalWrite(M1_STEP_PIN, HIGH);
        delayMicroseconds(STEP_PULSE_US_M1);
        digitalWrite(M1_STEP_PIN, LOW);
        delayMicroseconds(STEP_PULSE_US_M1);
    }
}

void rotateMotor2(long steps, bool clockwise) {
    if (steps < 0) steps = -steps;
    digitalWrite(M2_DIR_PIN, clockwise ? HIGH : LOW);

    // Ramp up speed from 2000µs to 500µs over first 1500 steps
    const long RAMP_STEPS = 1500;
    const unsigned int START_DELAY = 2000;
    const unsigned int END_DELAY = STEP_PULSE_US_M2;  // 500µs
    
    for (long i = 0; i < steps; i++) {
        // Calculate current delay with linear interpolation during ramp phase
        unsigned int currentDelay = START_DELAY;
        if (i < RAMP_STEPS) {
            currentDelay = START_DELAY - ((START_DELAY - END_DELAY) * i / RAMP_STEPS);
        } else {
            currentDelay = END_DELAY;
        }
        
        digitalWrite(M2_STEP_PIN, HIGH);
        delayMicroseconds(currentDelay);
        digitalWrite(M2_STEP_PIN, LOW);
        delayMicroseconds(currentDelay);
    }
}

void rotateBothMotors(long steps, bool clockwise) {
    if (steps < 0) steps = -steps;
    
    // Set direction for both motors
    digitalWrite(M1_DIR_PIN, clockwise ? HIGH : LOW);
    digitalWrite(M2_DIR_PIN, clockwise ? HIGH : LOW);

    // Step both motors simultaneously (use M1's faster speed)
    for (long i = 0; i < steps; i++) {
        digitalWrite(M1_STEP_PIN, HIGH);
        digitalWrite(M2_STEP_PIN, HIGH);
        delayMicroseconds(STEP_PULSE_US_M1);
        
        digitalWrite(M1_STEP_PIN, LOW);
        digitalWrite(M2_STEP_PIN, LOW);
        delayMicroseconds(STEP_PULSE_US_M1);
    }
}

bool rotateMotor1UntilSwitch(long maxSteps) {
    if (maxSteps < 0) maxSteps = -maxSteps;
    digitalWrite(M1_DIR_PIN, LOW); // CCW for "spin up"

    for (long i = 0; i < maxSteps; i++) {
        if (digitalRead(LIMIT_SWITCH_PIN) == LOW) {
            delay(10);  // Small delay to let motor settle
            return true;
        }
        digitalWrite(M1_STEP_PIN, HIGH);
        delayMicroseconds(STEP_PULSE_US_M1);
        digitalWrite(M1_STEP_PIN, LOW);
        delayMicroseconds(STEP_PULSE_US_M1);
    }
    return false;
}

bool rotateMotor1UntilSwitchClockwise(long maxSteps) {
    if (maxSteps < 0) maxSteps = -maxSteps;
    digitalWrite(M1_DIR_PIN, HIGH); // CW for "spin down"

    for (long i = 0; i < maxSteps; i++) {
        if (digitalRead(LIMIT_SWITCH_PIN) == LOW) {
            delay(10);  // Small delay to let motor settle
            return true;
        }
        digitalWrite(M1_STEP_PIN, HIGH);
        delayMicroseconds(STEP_PULSE_US_M1);
        digitalWrite(M1_STEP_PIN, LOW);
        delayMicroseconds(STEP_PULSE_US_M1);
    }
    return false;
}

int clampServoAngle(int angle) {
    if (angle < SERVO_MIN_ANGLE) return SERVO_MIN_ANGLE;
    if (angle > SERVO_MAX_ANGLE) return SERVO_MAX_ANGLE;
    return angle;
}

int relativeServoAngle(int offset) {
    int angle = SERVO_CENTER_ANGLE + offset;
    return clampServoAngle(angle);
}