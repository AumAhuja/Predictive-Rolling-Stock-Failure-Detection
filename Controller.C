#include <WiFi.h>
#include <esp_now.h>
#include "esp_wifi.h"

// =====================================
// BUTTON PINS
// =====================================

#define RED1    32
#define BLACK1  33
#define RED2    18
#define BLACK2  19


// =====================================
// BROADCAST ADDRESS
// =====================================

uint8_t receiverAddress[] = {
  0xFF,
  0xFF,
  0xFF,
  0xFF,
  0xFF,
  0xFF
};


// =====================================
// SEND COMMAND
// =====================================

void sendCommand(char command) {

  esp_err_t result = esp_now_send(
    receiverAddress,
    (uint8_t *)&command,
    1
  );

  if (result == ESP_OK) {

    Serial.print("SENT: ");
    Serial.println(command);

  } else {

    Serial.print("SEND ERROR: ");
    Serial.println(result);
  }
}


// =====================================
// SETUP
// =====================================

void setup() {

  Serial.begin(115200);


  // ===================================
  // BUTTONS
  // ===================================

  pinMode(RED1, INPUT_PULLUP);
  pinMode(BLACK1, INPUT_PULLUP);
  pinMode(RED2, INPUT_PULLUP);
  pinMode(BLACK2, INPUT_PULLUP);


  // ===================================
  // WIFI
  // ===================================

  WiFi.mode(WIFI_STA);

  esp_wifi_set_channel(
    1,
    WIFI_SECOND_CHAN_NONE
  );


  // ===================================
  // MAC
  // ===================================

  Serial.println();
  Serial.println("==============================");
  Serial.println("ESP32 #1 CONTROLLER");
  Serial.println("==============================");

  Serial.print("MAC: ");
  Serial.println(WiFi.macAddress());


  // ===================================
  // ESP-NOW
  // ===================================

  if (esp_now_init() != ESP_OK) {

    Serial.println("ESP-NOW INIT FAILED!");

    while (true) {
      delay(1000);
    }
  }


  // ===================================
  // ADD BROADCAST PEER
  // ===================================

  esp_now_peer_info_t peerInfo = {};

  memcpy(
    peerInfo.peer_addr,
    receiverAddress,
    6
  );

  peerInfo.channel = 1;
  peerInfo.encrypt = false;


  if (esp_now_add_peer(&peerInfo) != ESP_OK) {

    Serial.println("FAILED TO ADD PEER!");

    while (true) {
      delay(1000);
    }
  }


  Serial.println("ESP-NOW READY");
  Serial.println("CONTROLLER READY");
}


// =====================================
// LOOP
// =====================================

void loop() {

  // ===================================
  // RED 1
  // ===================================

  if (digitalRead(RED1) == LOW) {

    sendCommand('A');
  }


  // ===================================
  // BLACK 1
  // ===================================

  else if (digitalRead(BLACK1) == LOW) {

    sendCommand('B');
  }


  // ===================================
  // RED 2
  // ===================================

  else if (digitalRead(RED2) == LOW) {

    sendCommand('C');
  }


  // ===================================
  // BLACK 2
  // ===================================

  else if (digitalRead(BLACK2) == LOW) {

    sendCommand('D');
  }


  // ===================================
  // NO BUTTON
  // ===================================

  else {

    sendCommand('S');
  }


  // Send every 100 ms
  delay(100);
}
