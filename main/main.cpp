#include "Arduino.h"
#include "esp_log.h"
#include "esp_camera.h"
#include <WiFi.h>
#include <WiFiClientSecure.h>
#include <ESP_Mail_Client.h>
#include <LittleFS.h>
#include <Wire.h>
#include <Adafruit_GFX.h>
// #include <Adafruit_SSD1306.h>
#include <SPI.h>
#include <MFRC522.h>
#include <Adafruit_MCP23X17.h>
#include <Adafruit_AHTX0.h>
#include <ld2410.h>
#include "camera_pins.h"
#include "door_events.h"
#include "antispoof_task.h"

static const char* TAG = "MAIN";

// Forward declarations từ code Arduino cũ
void setup();
void loop();
void startCameraServer();

// FreeRTOS Handles (Định nghĩa lại ở đây)
// No app_main here - the Arduino-ESP-IDF component provides it and calls setup()/loop().
