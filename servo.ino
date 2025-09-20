#include <Servo.h>
Servo s1;
int servoPin=9;

void setup() {
 s1.attach(servoPin);
 Serial.begin(9600);
}

void loop() {
s1.write(45);
for(int i=0;i<90;i++){
  s1.write(i+45);
  delay(50);
}


}