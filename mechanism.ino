#include<Servo.h>

Servo s1;
Servo s2;
Servo s3;
Servo s4;
String input;

void setup(){
  s1.attach(11);
  s2.attach(10);
  s3.attach(9);
  
  s4.attach(6);
  Serial.begin(9600);
  
  s1.write(0);
  s2.write(0);
  s3.write(0);
  s4.write(0);
  Serial.println("Enter company name: ");
}
  
  
   void loop() {
     if (Serial.available()) {
    input = Serial.readStringUntil('\n');
    

    if (input == "Amazon") {
      for(int i=0;i<=90;i++){
       s1.write(i);
      }
    }else if (input == "Flipkart") {
      for(int i=0;i<=90;i++){
       s2.write(i);
      }
    }else if (input == "Blinkit") {
      for(int i=0;i<=90;i++){
       s3.write(i);
      }
    }else if (input == "Myntra") {
      for(int i=0;i<=90;i++){
       s4.write(i);
      }
    }else{
        Serial.println("Invalid Input");

     }
  }
}