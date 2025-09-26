#include<Servo.h>
#include<LiquidCrystal_I2C.h>

Servo s_amazon;



void setup(){

    s_amazon.attach(9);
    Serial.begin(9600);


}
void loop() {
  if (Serial.available()) {
    String cmd = Serial.readStringUntil('\n');
    if (cmd == "OPEN") s_amazon.write(90);   // open servo
    if (cmd == "CLOSE") s_amazon.write(0);   // close servo
    if (cmd.startsWith("Amazon Boxes:")) lcd.print(cmd);
  }
}
