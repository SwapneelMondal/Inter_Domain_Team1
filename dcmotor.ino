int pwmPin = 9;
int dirPin = 8;

void setup() {
  pinMode(pwmPin, OUTPUT);
  pinMode(dirPin, OUTPUT);
  digitalWrite(dirPin, HIGH);
}

void loop() {
  int speed = 255;
  analogWrite(pwmPin, speed);
  delay(5000);
  analogWrite(pwmPin, 0);
}
