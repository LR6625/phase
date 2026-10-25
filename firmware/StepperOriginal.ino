// Stepper Motor Control with DRV8825
// DIR pin: 5
// STEP pin: 4
// Serial commands:
//   SPIN_CW:<steps>
//   SPIN_CCW:<steps>
//   SLEEP:<milliseconds>

const int DIR_PIN = 5;
const int STEP_PIN = 4;
const unsigned int STEP_PULSE_US = 500;

String readLine() {
    String line = Serial.readStringUntil('\n');
    line.trim();
    return line;
}

void setup() {
    pinMode(DIR_PIN, OUTPUT);
    pinMode(STEP_PIN, OUTPUT);
    Serial.begin(9600);
    Serial.println("Stepper Motor Ready");
}

void loop() {
    if (Serial.available() > 0) {
        String command = readLine();

        if (command.startsWith("SPIN_CW:")) {
            long steps = command.substring(8).toInt();
            rotateMotor(steps, true);
            Serial.println("OK");
        } else if (command.startsWith("SPIN_CCW:")) {
            long steps = command.substring(9).toInt();
            rotateMotor(steps, false);
            Serial.println("OK");
        } else if (command.startsWith("SLEEP:")) {
            long ms = command.substring(6).toInt();
            if (ms < 0) ms = 0;
            delay(ms);
            Serial.println("OK");
        } else {
            Serial.println("ERR:UNKNOWN");
        }
    }
}

void rotateMotor(long steps, bool clockwise) {
    if (steps < 0) steps = -steps;
    digitalWrite(DIR_PIN, clockwise ? HIGH : LOW);

    for (long i = 0; i < steps; i++) {
        digitalWrite(STEP_PIN, HIGH);
        delayMicroseconds(STEP_PULSE_US);
        digitalWrite(STEP_PIN, LOW);
        delayMicroseconds(STEP_PULSE_US);
    }
}
