#include <Servo.h>

Servo servo1;
const int motorPin = 11;

void setup() {
  Serial.begin(9600);
  servo1.attach(9);
  servo1.write(0);      

  pinMode(motorPin, OUTPUT);
  digitalWrite(motorPin, HIGH);  
}

void loop() {
  if (Serial.available()) {
    String cmd = Serial.readStringUntil('\n');
    cmd.trim();

    if (cmd == "OPEN") {
      servo1.write(90);  
    }
    else if (cmd == "CLOSE") {
      servo1.write(0);  
    }
    else if (cmd == "MOTOR_ON") {
      digitalWrite(motorPin, LOW); 
      delay(2000);      
      digitalWrite(motorPin, HIGH);
    }
  }
}