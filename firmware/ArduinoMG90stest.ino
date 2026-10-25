#include <Servo.h>

// Create servo object
Servo myServo;

// Pin where servo is connected
const int servoPin = 9;

int currentAngle = 90;  // Start at center position

void setup() {
  // Initialize serial communication
  Serial.begin(9600);
  
  // Attach the servo to pin 9
  myServo.attach(servoPin);
  
  // Move to starting position
  myServo.write(currentAngle);
  Serial.println("Servo initialized on pin 9 - Ready for commands");
}

void loop() {
  // Listen for serial commands
  if (Serial.available() > 0) {
    String input = Serial.readStringUntil('\n');
    int angle = input.toInt();
    
    // Constrain angle between 0 and 180 degrees
    angle = constrain(angle, 0, 180);
    currentAngle = angle;
    
    // Move servo to the specified angle
    myServo.write(angle);
    
    // Send confirmation
    Serial.print("Servo moved to: ");
    Serial.print(angle);
    Serial.println(" degrees");
  }
}
