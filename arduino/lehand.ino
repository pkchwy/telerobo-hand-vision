

#include <Servo.h>

// Servo control
#define SERVO_COUNT 5
int servoPins[SERVO_COUNT] = {3, 5, 6, 9, 10}; // PWM pins for servos
Servo servos[SERVO_COUNT];

// Servo positions (PWM values 500-2500)
int servoPositions[SERVO_COUNT] = {1500, 1500, 1500, 1500, 1500};
String inputString = "";
boolean stringComplete = false;



void setup() {
  Serial.begin(9600);
  
  // Initialize servos
  for(int i = 0; i < SERVO_COUNT; i++) {
    servos[i].attach(servoPins[i]);
    servos[i].writeMicroseconds(servoPositions[i]);
  }
  
  // Configure status LED
  pinMode(13, OUTPUT);
  digitalWrite(13, HIGH); // Ready indicator
  
  Serial.println("Robot hand ready - waiting for servo commands");
  inputString.reserve(200);
}




void loop() {
  // Check for incoming serial data
  if (Serial.available()) {
    char inChar = (char)Serial.read();
    if (inChar == '\n') {
      stringComplete = true;
    } else {
      inputString += inChar;
    }
  }
  
  // Process complete command
  if (stringComplete) {
    parseServoCommand(inputString);
    inputString = "";
    stringComplete = false;
  }
}

// Parse servo command from server.py (format: "1500,1800,1200,2000,900")
void parseServoCommand(String command) {
  int values[SERVO_COUNT];
  int valueIndex = 0;
  int lastIndex = 0;
  
  // Parse comma-separated values
  for (int i = 0; i <= command.length() && valueIndex < SERVO_COUNT; i++) {
    if (command.charAt(i) == ',' || i == command.length()) {
      String valueStr = command.substring(lastIndex, i);
      values[valueIndex] = valueStr.toInt();
      
      // Constrain servo values (500-2500)
      values[valueIndex] = constrain(values[valueIndex], 500, 2500);
      valueIndex++;
      lastIndex = i + 1;
    }
  }
  
  // Update servo positions if we got all 5 values
  if (valueIndex == SERVO_COUNT) {
    for (int i = 0; i < SERVO_COUNT; i++) {
      servoPositions[i] = values[i];
      servos[i].writeMicroseconds(servoPositions[i]);
    }
    
    // Send confirmation
    Serial.print("Servos updated: ");
    for (int i = 0; i < SERVO_COUNT; i++) {
      Serial.print(servoPositions[i]);
      if (i < SERVO_COUNT - 1) Serial.print(",");
    }
    Serial.println();
  }
}

