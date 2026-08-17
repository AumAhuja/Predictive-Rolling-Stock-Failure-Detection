#include <WiFi.h>
#include <esp_now.h>
#include "esp_wifi.h"

// =====================================
// L298N MOTOR PINS
// =====================================

// MOTOR 1
#define IN1 25
#define IN2 26

// MOTOR 2
#define IN3 27
#define IN4 14


// =====================================
// MOTOR 1
// =====================================

void motor1Stop() {
  digitalWrite(IN1, LOW);
  digitalWrite(IN2, LOW);
}

void motor1Forward() {

  // Stop first
  motor1Stop();
  delay(10);

  digitalWrite(IN1, HIGH);
  digitalWrite(IN2, LOW);
}

void motor1Backward() {

  // Stop first
  motor1Stop();
  delay(10);

  digitalWrite(IN1, LOW);
  digitalWrite(IN2, HIGH);
}


// =====================================
// MOTOR 2
// =====================================

void motor2Stop() {
  digitalWrite(IN3, LOW);
  digitalWrite(IN4, LOW);
}

void motor2Forward() {

  // Stop first
  motor2Stop();
  delay(10);

  digitalWrite(IN3, HIGH);
  digitalWrite(IN4, LOW);
}

void motor2Backward() {

  // Stop first
  motor2Stop();
  delay(10);

  digitalWrite(IN3, LOW);
  digitalWrite(IN4, HIGH);
}


// =====================================
// STOP BOTH MOTORS
// =====================================

void stopAllMotors() {

  motor1Stop();
  motor2Stop();
}


// =====================================
// RECEIVE ESP-NOW DATA
// =====================================

void receiveData(
  const esp_now_recv_info_t *info,
  const uint8_t *data,
  int len
) {

  // We expect exactly 1 byte
  if (len != 1) {
    Serial.println("INVALID DATA");
    return;
  }

  char command = data[0];

  Serial.print("RECEIVED: ");
  Serial.println(command);


  // ===================================
  // MOTOR 1 FORWARD
  // A
  // ===================================

  if (command == 'A' || command == 'a') {

    Serial.println("MOTOR 1 FORWARD");

    // Keep motor 2 stopped
    motor2Stop();

    // Motor 1 forward
    motor1Forward();
  }


  // ===================================
  // MOTOR 1 BACKWARD
  // B
  // ===================================

  else if (command == 'B' || command == 'b') {

    Serial.println("MOTOR 1 BACKWARD");

    // Keep motor 2 stopped
    motor2Stop();

    // Motor 1 backward
    motor1Backward();
  }


  // ===================================
  // MOTOR 2 FORWARD
  // C
  // ===================================

  else if (command == 'C' || command == 'c') {

    Serial.println("MOTOR 2 FORWARD");

    // Keep motor 1 stopped
    motor1Stop();

    // Motor 2 forward
    motor2Forward();
  }


  // ===================================
  // MOTOR 2 BACKWARD
  // D
  // ===================================

  else if (command == 'D' || command == 'd') {

    Serial.println("MOTOR 2 BACKWARD");

    // Keep motor 1 stopped
    motor1Stop();

    // Motor 2 backward
    motor2Backward();
  }


  // ===================================
  // STOP
  // =====================================

  else if (command == 'S' || command == 's') {

    Serial.println("STOP");

    stopAllMotors();
  }


  // ===================================
  // UNKNOWN COMMAND
  // ===================================

  else {

    Serial.println("UNKNOWN COMMAND");

    stopAllMotors();
  }
}


// =====================================
// SETUP
// =====================================

void setup() {

  Serial.begin(115200);


  // ===================================
  // L298N PINS
  // ===================================

  pinMode(IN1, OUTPUT);
  pinMode(IN2, OUTPUT);

  pinMode(IN3, OUTPUT);
  pinMode(IN4, OUTPUT);


  // ===================================
  // MOTORS OFF
  // ===================================

  stopAllMotors();


  // ===================================
  // WIFI
  // ===================================

  WiFi.mode(WIFI_STA);

  // Same channel as transmitter
  esp_wifi_set_channel(1, WIFI_SECOND_CHAN_NONE);


  // ===================================
  // INFORMATION
  // ===================================

  Serial.println();
  Serial.println("==============================");
  Serial.println("ESP32 #2 MOTOR RECEIVER");
  Serial.println("==============================");

  Serial.print("MAC: ");
  Serial.println(WiFi.macAddress());


  // ===================================
  // ESP-NOW
  // ===================================

  if (esp_now_init() != ESP_OK) {

    Serial.println("ESP-NOW INIT FAILED!");

    stopAllMotors();

    while (true) {
      delay(1000);
    }
  }


  // ===================================
  // REGISTER RECEIVE CALLBACK
  // ===================================

  esp_now_register_recv_cb(receiveData);


  Serial.println("ESP-NOW READY");
  Serial.println("WAITING FOR COMMANDS...");
  Serial.println();
}


// =====================================
// LOOP
// =====================================

void loop() {

  // NOTHING HERE

  // Motors continue running until
  // another command is received.

  delay(10);
}
