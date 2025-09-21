#include <Servo.h>
#include <LiquidCrystal_I2C.h>

Servo s1, s2, s3, s4;
LiquidCrystal_I2C lcd(0x20, 16, 2);
String input;

void setup() {
  s1.attach(11);
  s2.attach(10);
  s3.attach(9);
  s4.attach(6);

  Serial.begin(9600);

  lcd.init();
  lcd.backlight();
  lcd.clear();

  s1.write(0);
  s2.write(0);
  s3.write(0);
  s4.write(0);

  Serial.println("Enter company name: ");
  lcd.setCursor(0, 0);
  lcd.print("Enter company:");
}

void loop() {
  if (Serial.available()) {
    input = Serial.readStringUntil('\n');

    if (input == "Amazon") {
      for (int i = 0; i <= 90; i++) {
        s1.write(i);
        delay(15);
      }
      lcd.clear();
      lcd.print("Amazon box");
    } 
    else if (input == "Flipkart") {
      for (int i = 0; i <= 90; i++) {
        s2.write(i);
        delay(15);
      }
      lcd.clear();
      lcd.print("Flipkart box");
    } 
    else if (input == "Blinkit") {
      for (int i = 0; i <= 90; i++) {
        s3.write(i);
        delay(15);
      }
      lcd.clear();
      lcd.print("Blinkit box");
    } 
    else if (input == "Myntra") {
      for (int i = 0; i <= 90; i++) {
        s4.write(i);
        delay(15);
      }
      lcd.clear();
      lcd.print("Myntra box");
    } 
    else {
      Serial.println("Invalid Input");
      lcd.clear();
      lcd.print("Invalid Input");
    }
   
  }
}
