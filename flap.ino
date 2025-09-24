#include <Servo.h>

Servo servo1;

void setup() {
  Serial.begin(9600);
  servo1.attach(9); // Servo signal wire on pin 9
  servo1.write(0);  // Start at 0 degrees
}

void loop() {
  if (Serial.available()) {
    String cmd = Serial.readStringUntil('\n');
    cmd.trim();

    if (cmd == "OPEN") {
      servo1.write(90);  // Move to 90 degrees
    }
    else if (cmd == "CLOSE") {
      servo1.write(0);   // Move back to 0 degrees
    }
  }
}
