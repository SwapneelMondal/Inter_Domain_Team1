#include <Servo.h>
#include <LiquidCrystal_I2C.h>

LiquidCrystal_I2C lcd(0x20, 16, 2);

Servo s1;
Servo s2;
Servo s3;

int amazonBox = 0;
int flipkartBox = 0;
int myntraBox = 0;

int in1 = 12;
int in2 = 13;
int en  = 10;

String input;

void setup() {
  s1.attach(6);
  s2.attach(5);
  s3.attach(3);

  s1.write(0);
  s2.write(0);
  s3.write(0);

  pinMode(in1, OUTPUT);
  pinMode(in2, OUTPUT);
  pinMode(en, OUTPUT);

  digitalWrite(in1, HIGH);
  digitalWrite(in2, LOW);
  analogWrite(en, 255);

  lcd.init();
  lcd.backlight();
  lcd.setCursor(0,0);
  lcd.print("Robot Ready");

  Serial.begin(9600);
  Serial.println("Enter company name: ");
}

void loop() {
  if (Serial.available()) {
    input = Serial.readStringUntil('\n');
    lcd.clear();

    if (input == "Amazon") {
      s1.write(90);
      lcd.setCursor(0,0);
      lcd.print("Amazon Box");
      amazonBox++;
      Serial.print(amazonBox);
      Serial.println(" Amazon boxes detected");

      delay(7000); 
      s1.write(0); 
    }
    else if (input == "Flipkart") {
      s2.write(90); 
      lcd.setCursor(0,0);
      lcd.print("Flipkart Box");
      flipkartBox++;
      Serial.print(flipkartBox);
      Serial.println(" Flipkart boxes detected");

      delay(5000); 
      s2.write(0);   
    }
    else if (input == "Myntra") {
      s3.write(90);  
      lcd.setCursor(0,0);
      lcd.print("Myntra Box");
      myntraBox++;
      Serial.print(myntraBox);
      Serial.println(" Myntra boxes detected");

      delay(5000);   
      s3.write(0);  
    }
    else {
      lcd.setCursor(0,0);
      lcd.print("Invalid Input");
      Serial.println("Invalid Input");
    }

    Serial.println("Enter company name: ");
  }
}
