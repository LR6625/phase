// Dual Stepper Motor Control with DRV8825
// Motor 1:
//   DIR pin: 5
//   STEP pin: 4
// Motor 2:
//   DIR pin: 7
//   STEP pin: 6
// 
// Serial commands:
//   SPIN_CW:M1:<steps>   - Rotate motor 1 clockwise
//   SPIN_CCW:M1:<steps>  - Rotate motor 1 counter-clockwise
//   SPIN_CW:M2:<steps>   - Rotate motor 2 clockwise
//   SPIN_CCW:M2:<steps>  - Rotate motor 2 counter-clockwise
//   SPIN_CW:BOTH:<steps> - Rotate both motors clockwise
//   SPIN_CCW:BOTH:<steps> - Rotate both motors counter-clockwise
//   SPIN_UP_TO_SWITCH:M1[:<maxSteps>] - Rotate motor 1 CCW until limit switch is pressed
//   SLEEP:<milliseconds> - Sleep

// Motor 1 pins
const int M1_DIR_PIN = 5;
const int M1_STEP_PIN = 4;

// Motor 2 pins
const int M2_STEP_PIN = 6;
const int M2_DIR_PIN = 7;

const int LIMIT_SWITCH_PIN = 13;
const long DEFAULT_MAX_STEPS = 200000;

const unsigned int STEP_PULSE_US = 125;
// Motor 1 now runs 4x faster with 1/32nd microstepping

String readLine() {
    String line = Serial.readStringUntil('\n');
    line.trim();
    return line;
}

void setup() {
    // Motor 1 pins
    pinMode(M1_DIR_PIN, OUTPUT);
    pinMode(M1_STEP_PIN, OUTPUT);
    
    // Motor 2 pins
    pinMode(M2_DIR_PIN, OUTPUT);
    pinMode(M2_STEP_PIN, OUTPUT);

    pinMode(LIMIT_SWITCH_PIN, INPUT_PULLUP);
    
    // Force all pins low at startup (safe state)
    digitalWrite(M1_DIR_PIN, LOW);
    digitalWrite(M1_STEP_PIN, LOW);
    digitalWrite(M2_DIR_PIN, LOW);
    digitalWrite(M2_STEP_PIN, LOW);
    
    Serial.begin(9600);
    Serial.println("Dual Stepper Motor Ready");
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
            } else {
                Serial.println("ERR:UNKNOWN");
            }
        }
        // Handle SLEEP commands
        else if (command.startsWith("SLEEP:")) {
            long ms = command.substring(6).toInt();
            if (ms > 0) delay(ms);
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
        delayMicroseconds(STEP_PULSE_US);
        digitalWrite(M1_STEP_PIN, LOW);
        delayMicroseconds(STEP_PULSE_US);
    }
    // Motor 1 now steps 4x faster for 1/32nd microstepping mode
}

void rotateMotor2(long steps, bool clockwise) {
    if (steps < 0) steps = -steps;
    digitalWrite(M2_DIR_PIN, clockwise ? HIGH : LOW);

    for (long i = 0; i < steps; i++) {
        digitalWrite(M2_STEP_PIN, HIGH);
        delayMicroseconds(STEP_PULSE_US);
        digitalWrite(M2_STEP_PIN, LOW);
        delayMicroseconds(STEP_PULSE_US);
    }
}

void rotateBothMotors(long steps, bool clockwise) {
    if (steps < 0) steps = -steps;
    
    // Set direction for both motors
    digitalWrite(M1_DIR_PIN, clockwise ? HIGH : LOW);
    digitalWrite(M2_DIR_PIN, clockwise ? HIGH : LOW);

    // Step both motors simultaneously
    for (long i = 0; i < steps; i++) {
        digitalWrite(M1_STEP_PIN, HIGH);
        digitalWrite(M2_STEP_PIN, HIGH);
        delayMicroseconds(STEP_PULSE_US);
        
        digitalWrite(M1_STEP_PIN, LOW);
        digitalWrite(M2_STEP_PIN, LOW);
        delayMicroseconds(STEP_PULSE_US);
    }
}

bool rotateMotor1UntilSwitch(long maxSteps) {
    if (maxSteps < 0) maxSteps = -maxSteps;
    digitalWrite(M1_DIR_PIN, LOW); // CCW for "spin up"

    for (long i = 0; i < maxSteps; i++) {
        if (digitalRead(LIMIT_SWITCH_PIN) == LOW) {
            return true;
        }
        digitalWrite(M1_STEP_PIN, HIGH);
        delayMicroseconds(STEP_PULSE_US);
        digitalWrite(M1_STEP_PIN, LOW);
        delayMicroseconds(STEP_PULSE_US);
    }
    return false;
}