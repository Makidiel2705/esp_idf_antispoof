#include "Arduino.h"
#include "esp_task_wdt.h"
#include "esp_camera.h"
#include "esp_wifi.h"
#include <WiFi.h>
#include <WiFiClientSecure.h>
#include <ESP_Mail_Client.h>
#include <LittleFS.h>
#include <Wire.h>
#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>
#include <SPI.h>
#include <MFRC522.h>
#include <Adafruit_MCP23X17.h>
#include <Adafruit_AHTX0.h>
#include <LD2410.h>
#include <VL53L0X.h>
#include <string.h>
#include <sys/time.h>
#include "camera_pins.h"
#include "door_events.h"
#include "antispoof_task.h"

// Forward declaration của app_httpd.cpp
extern "C" void startCameraServer();
extern "C" bool buildTelegramJpegWithOverlay(camera_fb_t *fb, uint8_t **out_buf, size_t *out_len);
#include <Preferences.h>
Preferences prefs;

// FreeRTOS Handles
QueueHandle_t    doorQueue;
SemaphoreHandle_t i2cMutex;
TaskHandle_t rfidTaskHandle;
TaskHandle_t doorControlTaskHandle;

// RFID #1 (bên ngoài - xác thực)
#define SS_PIN   38
#define RST_PIN  42
#define SPI_SCK   39
#define SPI_MISO  41
#define SPI_MISO_2 20
#define SPI_MOSI  40
MFRC522 mfrc522(SS_PIN, RST_PIN);

// RFID #2 (bên trong - thay nút bấm, mở cửa ra)
#define SS_PIN_2  19
#define RST_PIN_2 48
MFRC522 mfrc522_inner(SS_PIN_2, RST_PIN_2);
static constexpr uint8_t RFID_READER_NONE = 0;
static constexpr uint8_t RFID_READER_OUTER = 1;
static constexpr uint8_t RFID_READER_INNER = 2;
static constexpr uint32_t RFID_READER_SLICE_MS = 250;
static volatile uint8_t active_rfid_reader = RFID_READER_NONE;
static bool rfid_outer_version_logged = false;
static bool rfid_inner_version_logged = false;

const String UID_ON  = "14:B3:27:16";
const String UID_OFF = "C4:8D:80:16";

// Time
const char* ntpServer = "pool.ntp.org";
const char* ntpServer2 = "time.google.com";
const char* ntpServer3 = "time.cloudflare.com";
const long  gmtOffset_sec = 25200;
const int   daylightOffset_sec = 0;

// Screen
#define SCREEN_WIDTH 128
#define SCREEN_HEIGHT 64
Adafruit_SSD1306 display(SCREEN_WIDTH, SCREEN_HEIGHT, &Wire, -1);

// I2C Pins
#define SDA_PIN 21
#define SCL_PIN 47

// MCP23017 Pin Mapping (Address 0x20)
#define MCP23017_ADDR   0x20
#define MCP_LOCK_PIN    3   // Port A, pin 3 (OUTPUT - Solenoid Lock)
#define MCP_LED_PIN     4   // Port A, pin 4 (OUTPUT - LED for face recognition)
#define MCP_BUZZER_PIN  5   // Port A, pin 5 (OUTPUT - Buzzer)
#define MCP_PUMP_PIN    7   // Port A, pin 7 (OUTPUT - Water Pump)
#define MCP_FAN_PIN     15  // Port B, pin 7 (OUTPUT - Fan)

Adafruit_MCP23X17 mcp;

// AHT20 (I2C address 0x38)
Adafruit_AHTX0 aht;
volatile float dht_temp = NAN;
volatile float dht_hum  = NAN;

// MQ2 Smoke Sensor (Analog on GPIO 14)
#define MQ2_PIN 14
volatile int mq2_value = 0;
volatile bool smoke_detected = false;
volatile bool pump_state = false;
volatile bool pump_auto_mode = true;
volatile float smoke_alarm_threshold = 10.0;

// LD2410 mmWave Radar (UART1: TX=2, RX=1)
#define LD2410_TX_PIN 2
#define LD2410_RX_PIN 1
LD2410 radar;
volatile bool presence_detected = false;
volatile int presence_distance = 0;
static constexpr bool kEnableLd2410 = true;
static SemaphoreHandle_t ld2410Mutex = nullptr;
static LD2410::BasicData ld2410_basic{};
static LD2410::EngineeringData ld2410_engineering{};
static LD2410::ConfigurationData ld2410_config{};
static bool ld2410_started = false;
static bool ld2410_engineering_enabled = false;
static bool ld2410_config_loaded = false;

// Fan Control
volatile bool fan_state = false;
volatile bool fan_auto_mode = false;
volatile float fan_humidity_threshold = 70.0;
static bool humidity_telegram_alert_sent = false;

// VL53L0X ToF Sensor (I2C address 0x29)
VL53L0X tof_sensor;
bool vl53l0x_found = false;
volatile int vl53l0x_distance_mm = 0;
volatile bool door_waiting_passthrough = false;
static volatile uint8_t door_passthrough_direction = 0; // 0=none, 1=entering(face), 2=exiting(button)
static volatile uint32_t door_unlocked_at_ms = 0;
static constexpr uint16_t DOOR_CLOSE_DISTANCE_MM = 60;
static constexpr uint32_t DOOR_UNLOCK_GRACE_MS = 10000UL;
static constexpr uint32_t DOOR_CLOSE_LOCK_HOLD_MS = 4000UL;
static constexpr uint32_t DOOR_PASS_PULSE_MIN_MS = 100UL;

// State Variables
bool RecognitionFace = false;
bool LockState = false;
bool mcp_found = false;
bool aht_found = false;
bool oled_found = false;
String passkey = "1111";
static char face_guidance_text[32] = "";
static volatile uint32_t face_guidance_until_ms = 0;
static portMUX_TYPE face_guidance_mux = portMUX_INITIALIZER_UNLOCKED;
static int oled_unlock_anim_frame = 0;
volatile int occupant_count = 0;
volatile bool rfid_auth_window_active = false;
volatile int rfid_expected_face_id = 0;
volatile int pending_rfid_enroll_face_id = 0;
volatile uint8_t ai_auth_flush_frames_remaining = 0;
volatile uint32_t ai_auth_warmup_until_ms = 0;
static unsigned long rfid_last_face_seen_ms = 0;
static bool mmwave_alert_sent = false;
static unsigned long mmwave_presence_started_ms = 0;
static unsigned long mmwave_absence_started_ms = 0;
static unsigned long mmwave_cooldown_until_ms = 0;
static bool smoke_telegram_alert_sent = false;

static constexpr int MAX_ROOM_OCCUPANTS = 12;
struct RoomOccupant {
  int face_id;
  char name[32];
  char role[24];
  char uid[24];
  uint32_t entry_ts;
};
static RoomOccupant room_occupants[MAX_ROOM_OCCUPANTS];
static int room_occupant_count = 0;
static SemaphoreHandle_t occupantMutex = nullptr;
static RoomOccupant pending_entry = {};
static RoomOccupant pending_exit = {};
static bool pending_entry_valid = false;
static bool pending_exit_valid = false;

// Telegram & Email Config
String BOT_TOKEN = "";
String CHAT_ID = "";
bool sendTelegramFlag = false;
String telegramActor = "";
String currentOTP = "";
static volatile bool telegramUploadInProgress = false;
static constexpr uint32_t TELEGRAM_CONNECT_TIMEOUT_MS = 15000;
static constexpr uint32_t TELEGRAM_RESPONSE_TIMEOUT_MS = 8000;
static constexpr uint32_t TELEGRAM_PHOTO_TASK_STACK = 8192;
static constexpr uint32_t TELEGRAM_TEXT_TASK_STACK = 6144;

struct TelegramTaskData {
  String caption;
  uint8_t *jpg_buf;
  size_t jpg_len;
};

#define SMTP_HOST "smtp.gmail.com"
#define SMTP_PORT 587
#define AUTHOR_EMAIL ""
#define AUTHOR_PASSWORD ""
#define RECIPIENT_EMAIL "tranchikien27052003@gmail.com"
SMTPSession smtp;

void sendOTPEmail(String otp) {
  Session_Config config;
  config.server.host_name = SMTP_HOST;
  config.server.port = SMTP_PORT;
  config.login.email = AUTHOR_EMAIL;
  config.login.password = AUTHOR_PASSWORD;
  config.login.user_domain = "";

  SMTP_Message message;
  message.sender.name = "ESP32-S3 Camera Guard";
  message.sender.email = AUTHOR_EMAIL;
  message.subject = "ESP32 Verification Code";
  message.addRecipient("Admin", RECIPIENT_EMAIL);

  String htmlMsg =
      "<div style='font-family:Arial;padding:20px;border:1px solid #ddd;border-radius:10px;max-width:400px;'>"
      "<h2>ESP32 verification code</h2>"
      "<p>Your OTP code is:</p>"
      "<h1 style='color:#2563eb;letter-spacing:5px;text-align:center;'>" + otp + "</h1>"
      "<p style='font-size:12px;color:#888;'>This code will expire soon.</p>"
      "</div>";
  message.html.content = htmlMsg.c_str();
  message.text.content = ("Your OTP code is: " + otp).c_str();

  if (!smtp.connect(&config)) {
    Serial.printf("SMTP connect failed: %s\n", smtp.errorReason().c_str());
    return;
  }

  if (!MailClient.sendMail(&smtp, &message)) {
    Serial.printf("SMTP send failed: %s\n", smtp.errorReason().c_str());
  }
}

void emailTask(void *pvParameters) {
  sendOTPEmail(currentOTP);
  vTaskDelete(NULL);
}

void triggerSendOTP() {
  xTaskCreate(emailTask, "emailTask", 16384, NULL, 1, NULL);
}

#include "bitmaps.h" 

// --- Helper Functions ---
void unlockDoor();
void lockDoor();
void playWebURL();
void playSuccess();
void playFail();
void playUnlock();
void add_event(uint8_t method, const char* actor);
void sendTelegramPhoto(String caption);
void sendTelegramMessage(String message);
void telegramTask(void *pvParameters);
void telegramMessageTask(void *pvParameters);
static void logNetworkEndpoints();
static void loadOccupantCount();
static void refreshOledStatus();
static String face_rfid_key(int id);
static String face_name_key(int id);
static String face_role_key(int id);
static String face_fail_key(int id);
static String face_lock_key(int id);
static String occupant_id_key(int slot);
static String occupant_uid_key(int slot);
static String occupant_ts_key(int slot);
static void saveOccupants();
static void loadOccupants();
static String formatTimeTs(uint32_t ts);
static String formatDuration(uint32_t seconds);
static bool findOccupantByFace(int face_id, int *index);
static bool findOccupantByUid(const String &uid, int *index);
static int findFaceIdByRfid(const String &uid);
static void startRfidAuthWindow(int face_id);
static void clearRfidAuthWindow();
void oledStatusTask(void *pvParameters);
static void disableAllRfidReaders();
static void selectRfidReader(uint8_t reader);
static bool readOuterRfidOnce();
static bool readInnerRfidOnce();
static void commitPendingPassthrough();
void vl53l0xTask(void *pvParameters);
void buzzBeep(unsigned long duration_ms = 200);

// --- Task Implementations ---
void setCameraStatusLed(bool enabled) {
  if (!mcp_found || !i2cMutex) return;
  if (xSemaphoreTake(i2cMutex, pdMS_TO_TICKS(300)) == pdTRUE) {
    mcp.digitalWrite(MCP_LED_PIN, enabled ? HIGH : LOW);
    xSemaphoreGive(i2cMutex);
  } else {
    Serial.printf("[WARN] Failed to %s camera LED: I2C busy\r\n", enabled ? "enable" : "disable");
  }
}

static void loadOccupantCount() {
  occupant_count = prefs.getInt("occupant_count", 0);
  if (occupant_count < 0) occupant_count = 0;
}

static String face_rfid_key(int id) {
  return "rfid_" + String(id);
}

static String face_name_key(int id) {
  return "id_" + String(id);
}

static String face_role_key(int id) {
  return "role_" + String(id);
}

static String face_fail_key(int id) {
  return "fail_" + String(id);
}

static String face_lock_key(int id) {
  return "lock_" + String(id);
}

static String occupant_id_key(int slot) {
  return "occ_id_" + String(slot);
}

static String occupant_uid_key(int slot) {
  return "occ_uid_" + String(slot);
}

static String occupant_ts_key(int slot) {
  return "occ_ts_" + String(slot);
}

static String jsonEscapeLocal(const String &value) {
  String escaped;
  escaped.reserve(value.length() + 8);
  for (size_t i = 0; i < value.length(); ++i) {
    char c = value[i];
    if (c == '\\') escaped += "\\\\";
    else if (c == '"') escaped += "\\\"";
    else if (c == '\n') escaped += "\\n";
    else if (c == '\r') escaped += "\\r";
    else if (c == '\t') escaped += "\\t";
    else escaped += c;
  }
  return escaped;
}

static String formatTimeTs(uint32_t ts) {
  if (ts == 0) return "";
  time_t t = (time_t)ts;
  struct tm timeinfo;
  localtime_r(&t, &timeinfo);
  char buf[24];
  strftime(buf, sizeof(buf), "%Y-%m-%d %H:%M:%S", &timeinfo);
  return String(buf);
}

static String formatDuration(uint32_t seconds) {
  uint32_t h = seconds / 3600;
  uint32_t m = (seconds % 3600) / 60;
  uint32_t s = seconds % 60;
  char buf[24];
  if (h > 0) snprintf(buf, sizeof(buf), "%luh %lum %lus", (unsigned long)h, (unsigned long)m, (unsigned long)s);
  else if (m > 0) snprintf(buf, sizeof(buf), "%lum %lus", (unsigned long)m, (unsigned long)s);
  else snprintf(buf, sizeof(buf), "%lus", (unsigned long)s);
  return String(buf);
}

static bool isValidUnixTime(time_t ts) {
  return ts >= 1704067200; // 2024-01-01, avoids falling back to 1970.
}

static void setSystemUnixTime(uint32_t unix_ts) {
  struct timeval tv = {};
  tv.tv_sec = static_cast<time_t>(unix_ts);
  tv.tv_usec = 0;
  settimeofday(&tv, nullptr);
}

static void restoreLastKnownTime() {
  time_t now;
  time(&now);
  if (isValidUnixTime(now)) return;
  uint32_t last = prefs.getUInt("last_unix", 0);
  if (isValidUnixTime(last)) {
    setSystemUnixTime(last);
    Serial.printf("[TIME] Restored last known time from NVS: %lu\n", static_cast<unsigned long>(last));
  }
}

static void persistCurrentTimeIfValid() {
  time_t now;
  time(&now);
  if (isValidUnixTime(now)) {
    prefs.putUInt("last_unix", static_cast<uint32_t>(now));
  }
}

void timeKeeperTask(void *pvParameters) {
  (void)pvParameters;
  while (true) {
    persistCurrentTimeIfValid();
    vTaskDelay(pdMS_TO_TICKS(60000));
  }
}

static bool findOccupantByFace(int face_id, int *index) {
  for (int i = 0; i < room_occupant_count; ++i) {
    if (room_occupants[i].face_id == face_id) {
      if (index) *index = i;
      return true;
    }
  }
  return false;
}

static bool findOccupantByUid(const String &uid, int *index) {
  for (int i = 0; i < room_occupant_count; ++i) {
    if (uid.equalsIgnoreCase(String(room_occupants[i].uid))) {
      if (index) *index = i;
      return true;
    }
  }
  return false;
}

static void saveOccupants() {
  prefs.putInt("occupant_count", room_occupant_count);
  for (int i = 0; i < MAX_ROOM_OCCUPANTS; ++i) {
    String id_key = occupant_id_key(i);
    String uid_key = occupant_uid_key(i);
    String ts_key = occupant_ts_key(i);
    if (i < room_occupant_count) {
      prefs.putInt(id_key.c_str(), room_occupants[i].face_id);
      prefs.putString(uid_key.c_str(), room_occupants[i].uid);
      prefs.putUInt(ts_key.c_str(), room_occupants[i].entry_ts);
    } else {
      if (prefs.isKey(id_key.c_str())) prefs.remove(id_key.c_str());
      if (prefs.isKey(uid_key.c_str())) prefs.remove(uid_key.c_str());
      if (prefs.isKey(ts_key.c_str())) prefs.remove(ts_key.c_str());
    }
  }
  occupant_count = room_occupant_count;
}

static void loadOccupants() {
  if (!occupantMutex) occupantMutex = xSemaphoreCreateMutex();
  if (xSemaphoreTake(occupantMutex, pdMS_TO_TICKS(500)) != pdTRUE) {
    loadOccupantCount();
    return;
  }
  int saved_count = prefs.getInt("occupant_count", 0);
  if (saved_count < 0) saved_count = 0;
  if (saved_count > MAX_ROOM_OCCUPANTS) saved_count = MAX_ROOM_OCCUPANTS;
  room_occupant_count = 0;
  for (int i = 0; i < saved_count; ++i) {
    int face_id = prefs.getInt(occupant_id_key(i).c_str(), 0);
    String uid = prefs.getString(occupant_uid_key(i).c_str(), "");
    uint32_t ts = prefs.getUInt(occupant_ts_key(i).c_str(), 0);
    if (face_id <= 0 || uid.length() == 0) continue;
    RoomOccupant &occ = room_occupants[room_occupant_count++];
    memset(&occ, 0, sizeof(occ));
    occ.face_id = face_id;
    String name = prefs.getString(face_name_key(face_id).c_str(), "Face " + String(face_id));
    String role = prefs.getString(face_role_key(face_id).c_str(), "");
    snprintf(occ.name, sizeof(occ.name), "%s", name.c_str());
    snprintf(occ.role, sizeof(occ.role), "%s", role.c_str());
    snprintf(occ.uid, sizeof(occ.uid), "%s", uid.c_str());
    occ.entry_ts = ts;
  }
  occupant_count = room_occupant_count;
  xSemaphoreGive(occupantMutex);
}

bool is_occupant_inside_by_face(int face_id) {
  bool inside = false;
  if (occupantMutex && xSemaphoreTake(occupantMutex, pdMS_TO_TICKS(100)) == pdTRUE) {
    inside = findOccupantByFace(face_id, nullptr);
    xSemaphoreGive(occupantMutex);
  }
  return inside;
}

bool is_occupant_inside_by_uid(const String &uid) {
  bool inside = false;
  if (occupantMutex && xSemaphoreTake(occupantMutex, pdMS_TO_TICKS(100)) == pdTRUE) {
    inside = findOccupantByUid(uid, nullptr);
    xSemaphoreGive(occupantMutex);
  }
  return inside;
}

bool prepare_pending_entry(int face_id) {
  if (!occupantMutex || xSemaphoreTake(occupantMutex, pdMS_TO_TICKS(200)) != pdTRUE) return false;
  if (findOccupantByFace(face_id, nullptr) || room_occupant_count >= MAX_ROOM_OCCUPANTS) {
    xSemaphoreGive(occupantMutex);
    return false;
  }
  memset(&pending_entry, 0, sizeof(pending_entry));
  pending_entry.face_id = face_id;
  String name = prefs.getString(face_name_key(face_id).c_str(), "Face " + String(face_id));
  String role = prefs.getString(face_role_key(face_id).c_str(), "");
  String uid = prefs.getString(face_rfid_key(face_id).c_str(), "");
  time_t now;
  time(&now);
  pending_entry.entry_ts = (uint32_t)now;
  snprintf(pending_entry.name, sizeof(pending_entry.name), "%s", name.c_str());
  snprintf(pending_entry.role, sizeof(pending_entry.role), "%s", role.c_str());
  snprintf(pending_entry.uid, sizeof(pending_entry.uid), "%s", uid.c_str());
  pending_entry_valid = true;
  xSemaphoreGive(occupantMutex);
  return true;
}

bool prepare_pending_exit_by_uid(const String &uid) {
  if (!occupantMutex || xSemaphoreTake(occupantMutex, pdMS_TO_TICKS(200)) != pdTRUE) return false;
  int index = -1;
  if (!findOccupantByUid(uid, &index)) {
    xSemaphoreGive(occupantMutex);
    return false;
  }
  pending_exit = room_occupants[index];
  pending_exit_valid = true;
  xSemaphoreGive(occupantMutex);
  return true;
}

String get_occupants_json() {
  String json = "[";
  time_t now;
  time(&now);
  if (occupantMutex && xSemaphoreTake(occupantMutex, pdMS_TO_TICKS(200)) == pdTRUE) {
    for (int i = 0; i < room_occupant_count; ++i) {
      if (i > 0) json += ",";
      uint32_t elapsed = (now > room_occupants[i].entry_ts) ? (uint32_t)(now - room_occupants[i].entry_ts) : 0;
      json += "{\"id\":" + String(room_occupants[i].face_id) +
              ",\"name\":\"" + jsonEscapeLocal(String(room_occupants[i].name)) +
              "\",\"role\":\"" + jsonEscapeLocal(String(room_occupants[i].role)) +
              "\",\"rfid\":\"" + jsonEscapeLocal(String(room_occupants[i].uid)) +
              "\",\"entry_time\":\"" + jsonEscapeLocal(formatTimeTs(room_occupants[i].entry_ts)) +
              "\",\"duration\":\"" + jsonEscapeLocal(formatDuration(elapsed)) + "\"}";
    }
    xSemaphoreGive(occupantMutex);
  }
  json += "]";
  return json;
}

void clear_occupants() {
  if (!occupantMutex || xSemaphoreTake(occupantMutex, pdMS_TO_TICKS(500)) != pdTRUE) return;
  room_occupant_count = 0;
  pending_entry_valid = false;
  pending_exit_valid = false;
  saveOccupants();
  xSemaphoreGive(occupantMutex);
}

void remove_occupant_by_face_id(int face_id) {
  if (!occupantMutex || xSemaphoreTake(occupantMutex, pdMS_TO_TICKS(500)) != pdTRUE) return;
  int index = -1;
  if (findOccupantByFace(face_id, &index)) {
    for (int i = index; i < room_occupant_count - 1; ++i) {
      room_occupants[i] = room_occupants[i + 1];
    }
    room_occupant_count--;
    saveOccupants();
  }
  xSemaphoreGive(occupantMutex);
}

void shift_occupants_after_face_delete(int deleted_face_id) {
  if (!occupantMutex || xSemaphoreTake(occupantMutex, pdMS_TO_TICKS(500)) != pdTRUE) return;
  int write_idx = 0;
  for (int i = 0; i < room_occupant_count; ++i) {
    if (room_occupants[i].face_id == deleted_face_id) continue;
    if (room_occupants[i].face_id > deleted_face_id) room_occupants[i].face_id--;
    room_occupants[write_idx++] = room_occupants[i];
  }
  room_occupant_count = write_idx;
  saveOccupants();
  xSemaphoreGive(occupantMutex);
}

static void disableAllRfidReaders() {
  static bool rfid_gpio_configured = false;
  if (!rfid_gpio_configured) {
    pinMode(SS_PIN, OUTPUT);
    pinMode(SS_PIN_2, OUTPUT);
    pinMode(RST_PIN, OUTPUT);
    pinMode(RST_PIN_2, OUTPUT);
    rfid_gpio_configured = true;
  }
  digitalWrite(SS_PIN, HIGH);
  digitalWrite(SS_PIN_2, HIGH);
  digitalWrite(RST_PIN, LOW);
  digitalWrite(RST_PIN_2, LOW);
  active_rfid_reader = RFID_READER_NONE;
}

static void selectRfidReader(uint8_t reader) {
  if (reader == active_rfid_reader) return;

  disableAllRfidReaders();
  delay(2);

  if (reader == RFID_READER_OUTER) {
    SPI.end();
    SPI.begin(SPI_SCK, SPI_MISO, SPI_MOSI, SS_PIN);
    digitalWrite(RST_PIN, HIGH);
    delay(5);
    mfrc522.PCD_Init();
    if (!rfid_outer_version_logged) {
      byte version = mfrc522.PCD_ReadRegister(MFRC522::VersionReg);
      Serial.printf("[RFID1] VersionReg=0x%02X (SS=%d RST=%d MISO=%d)\n", version, SS_PIN, RST_PIN, SPI_MISO);
      rfid_outer_version_logged = true;
    }
    active_rfid_reader = RFID_READER_OUTER;
  } else if (reader == RFID_READER_INNER) {
    SPI.end();
    SPI.begin(SPI_SCK, SPI_MISO_2, SPI_MOSI, SS_PIN_2);
    digitalWrite(RST_PIN_2, HIGH);
    delay(5);
    mfrc522_inner.PCD_Init();
    if (!rfid_inner_version_logged) {
      byte version = mfrc522_inner.PCD_ReadRegister(MFRC522::VersionReg);
      Serial.printf("[RFID2] VersionReg=0x%02X (SS=%d RST=%d MISO=%d)\n", version, SS_PIN_2, RST_PIN_2, SPI_MISO_2);
      rfid_inner_version_logged = true;
    }
    active_rfid_reader = RFID_READER_INNER;
  }
}

static bool readOuterRfidOnce() {
  static uint32_t last_scan_time = 0;
  if (!(mfrc522.PICC_IsNewCardPresent() && mfrc522.PICC_ReadCardSerial())) {
    return false;
  }
  if (millis() - last_scan_time < 2000) {
    mfrc522.PICC_HaltA();
    mfrc522.PCD_StopCrypto1();
    return true; // Ignore rapid consecutive scans
  }
  last_scan_time = millis();

  String uidString = "";
  for (byte i = 0; i < mfrc522.uid.size; i++) {
    if (mfrc522.uid.uidByte[i] < 0x10) uidString += "0";
    uidString += String(mfrc522.uid.uidByte[i], HEX);
    if (i < mfrc522.uid.size - 1) uidString += ":";
  }
  uidString.toUpperCase();

  DoorMessage msg = {};
  if (pending_rfid_enroll_face_id > 0) {
    int existing_face_id = findFaceIdByRfid(uidString);
    if (existing_face_id != 0 && existing_face_id != pending_rfid_enroll_face_id) {
      msg.cmd = CMD_PLAY_FAIL;
      msg.method = DM_RFID;
      strncpy(msg.actor, "rfid_already_bound", sizeof(msg.actor) - 1);
    } else {
      prefs.putString(face_rfid_key(pending_rfid_enroll_face_id).c_str(), uidString);
      append_log("RFID linked to Face " + String(pending_rfid_enroll_face_id) + ": " + uidString);
      msg.cmd = CMD_PLAY_UNLOCK;
      msg.method = DM_RFID;
      strncpy(msg.actor, "rfid_linked", sizeof(msg.actor) - 1);
      pending_rfid_enroll_face_id = 0;
    }
  } else {
    int face_id = findFaceIdByRfid(uidString);
    if (face_id > 0) {
      if (prefs.getBool(face_lock_key(face_id).c_str(), false)) {
        append_log("Tu choi RFID bi khoa: Face " + String(face_id) + ", RFID: " + uidString);
        msg.cmd = CMD_PLAY_FAIL;
        msg.method = DM_RFID;
        snprintf(msg.actor, sizeof(msg.actor), "locked_face_%d", face_id);
      } else if (is_occupant_inside_by_face(face_id)) {
        append_log("Tu choi vao lai: Face " + String(face_id) + " dang o trong phong, RFID: " + uidString);
        msg.cmd = CMD_PLAY_FAIL;
        msg.method = DM_RFID;
        strncpy(msg.actor, "already_inside", sizeof(msg.actor) - 1);
      } else {
        startRfidAuthWindow(face_id);
        setCameraStatusLed(true);
        append_log("RFID accepted, waiting face verification: " + uidString);
        msg.cmd = CMD_PLAY_UNLOCK;
        msg.method = DM_RFID;
        snprintf(msg.actor, sizeof(msg.actor), "rfid_wait_face_%d", face_id);
      }
    } else {
      msg.cmd = CMD_PLAY_FAIL;
      msg.method = DM_RFID;
      strncpy(msg.actor, "unknown_rfid", sizeof(msg.actor) - 1);
    }
  }
  msg.actor[sizeof(msg.actor) - 1] = '\0';
  xQueueSend(doorQueue, &msg, 0);
  mfrc522.PICC_HaltA();
  mfrc522.PCD_StopCrypto1();
  return true;
}

static bool readInnerRfidOnce() {
  if (!(mfrc522_inner.PICC_IsNewCardPresent() && mfrc522_inner.PICC_ReadCardSerial())) {
    return false;
  }

  String uidString = "";
  for (byte i = 0; i < mfrc522_inner.uid.size; i++) {
    if (mfrc522_inner.uid.uidByte[i] < 0x10) uidString += "0";
    uidString += String(mfrc522_inner.uid.uidByte[i], HEX);
    if (i < mfrc522_inner.uid.size - 1) uidString += ":";
  }
  uidString.toUpperCase();

  DoorMessage msg = {};
  msg.method = DM_BUTTON;
  int face_id = findFaceIdByRfid(uidString);
  if (face_id <= 0) {
    msg.cmd = CMD_PLAY_FAIL;
    strncpy(msg.actor, "unknown_exit_rfid", sizeof(msg.actor) - 1);
    append_log("Tu choi ra: the RFID chua dang ky: " + uidString);
    Serial.printf("[RFID2] Unknown card: %s\n", uidString.c_str());
  } else if (!prepare_pending_exit_by_uid(uidString)) {
    msg.cmd = CMD_PLAY_FAIL;
    snprintf(msg.actor, sizeof(msg.actor), "not_inside_%d", face_id);
    append_log("Tu choi ra: Face " + String(face_id) + " khong co trong danh sach trong phong, RFID: " + uidString);
    Serial.printf("[RFID2] Card %s belongs to Face %d but is not inside\n", uidString.c_str(), face_id);
  } else {
    msg.cmd = CMD_OPEN_DOOR;
    snprintf(msg.actor, sizeof(msg.actor), "exit_face_%d", face_id);
    Serial.printf("[RFID2] Card read: %s -> request exit door open for Face %d\n", uidString.c_str(), face_id);
  }
  if (xQueueSend(doorQueue, &msg, 0) != pdTRUE) {
    Serial.println("[RFID2] Failed to enqueue door event");
  }

  mfrc522_inner.PICC_HaltA();
  mfrc522_inner.PCD_StopCrypto1();
  return true;
}

static int findFaceIdByRfid(const String &uid) {
  int raw_count = prefs.getUInt("raw_count", 0);
  for (int id = 1; id <= raw_count; ++id) {
    if (prefs.getString(face_rfid_key(id).c_str(), "") == uid) {
      return id;
    }
  }
  return 0;
}

static void startRfidAuthWindow(int face_id) {
  rfid_expected_face_id = face_id;
  rfid_auth_window_active = true;
  rfid_last_face_seen_ms = millis();
  ai_auth_flush_frames_remaining = 2;
  ai_auth_warmup_until_ms = 0;
  notifyAiFrameEvent();
}

static void clearRfidAuthWindow() {
  rfid_expected_face_id = 0;
  rfid_auth_window_active = false;
  rfid_last_face_seen_ms = 0;
  ai_auth_flush_frames_remaining = 0;
  ai_auth_warmup_until_ms = 0;
  setCameraStatusLed(false);
}

void notifyRfidFaceSeen() {
  if (rfid_auth_window_active) {
    rfid_last_face_seen_ms = millis();
  }
}

void recordAuthFailureForFace(int face_id, const String &reason) {
  if (face_id <= 0) return;
  if (prefs.getBool(face_lock_key(face_id).c_str(), false)) return;

  int failures = prefs.getInt(face_fail_key(face_id).c_str(), 0) + 1;
  prefs.putInt(face_fail_key(face_id).c_str(), failures);
  String name = prefs.getString(face_name_key(face_id).c_str(), "Face " + String(face_id));
  String uid = prefs.getString(face_rfid_key(face_id).c_str(), "Unknown");
  append_log("Loi xac thuc lien tiep: " + name + " (" + String(failures) + "/3), ly do: " + reason);

  if (failures >= 3) {
    prefs.putBool(face_lock_key(face_id).c_str(), true);
    append_log("Da khoa nguoi dung: " + name + ", RFID: " + uid);
    sendTelegramMessage("Da khoa nguoi dung sau 3 lan xac thuc sai: " + name +
                        " (ID=" + String(face_id) + ", RFID=" + uid + ")");
  }
}

void resetAuthFailureForFace(int face_id) {
  if (face_id <= 0) return;
  if (prefs.getInt(face_fail_key(face_id).c_str(), 0) != 0) {
    prefs.putInt(face_fail_key(face_id).c_str(), 0);
  }
}

static void commitPendingPassthrough() {
  if (door_passthrough_direction == 1) {
    if (occupantMutex && xSemaphoreTake(occupantMutex, pdMS_TO_TICKS(500)) == pdTRUE) {
      if (pending_entry_valid && !findOccupantByFace(pending_entry.face_id, nullptr) && room_occupant_count < MAX_ROOM_OCCUPANTS) {
        room_occupants[room_occupant_count++] = pending_entry;
        pending_entry_valid = false;
        saveOccupants();
        append_log("Vao phong: ID=" + String(room_occupants[room_occupant_count - 1].face_id) +
                   ", Ten=" + String(room_occupants[room_occupant_count - 1].name) +
                   ", RFID=" + String(room_occupants[room_occupant_count - 1].uid) +
                   ", Gio vao=" + formatTimeTs(room_occupants[room_occupant_count - 1].entry_ts));
      }
      xSemaphoreGive(occupantMutex);
    }
    ESP_LOGI("VL53L0X", "Person entered. Occupants: %d", occupant_count);
  } else if (door_passthrough_direction == 2) {
    if (occupantMutex && xSemaphoreTake(occupantMutex, pdMS_TO_TICKS(500)) == pdTRUE) {
      int exit_index = -1;
      if (pending_exit_valid && findOccupantByUid(String(pending_exit.uid), &exit_index)) {
        time_t now;
        time(&now);
        uint32_t elapsed = (now > pending_exit.entry_ts) ? (uint32_t)(now - pending_exit.entry_ts) : 0;
        append_log("Ra khoi phong: ID=" + String(pending_exit.face_id) +
                   ", Ten=" + String(pending_exit.name) +
                   ", RFID=" + String(pending_exit.uid) +
                   ", Gio vao=" + formatTimeTs(pending_exit.entry_ts) +
                   ", Gio ra=" + formatTimeTs((uint32_t)now) +
                   ", Thoi gian=" + formatDuration(elapsed));
        sendTelegramMessage("Ra khoi phong: " + String(pending_exit.name) +
                            " (ID=" + String(pending_exit.face_id) +
                            ", RFID=" + String(pending_exit.uid) +
                            "), thoi gian trong phong: " + formatDuration(elapsed));
        for (int i = exit_index; i < room_occupant_count - 1; ++i) {
          room_occupants[i] = room_occupants[i + 1];
        }
        room_occupant_count--;
        pending_exit_valid = false;
        saveOccupants();
        if (occupant_count == 0) {
          mmwave_cooldown_until_ms = millis() + 30000UL;
        }
      }
      xSemaphoreGive(occupantMutex);
    }
    ESP_LOGI("VL53L0X", "Person exited. Occupants: %d", occupant_count);
  }
}

// AHT20 Task (replaces DHT11 task) + Auto Fan Logic
void ahtTask(void *pvParameters) {
  while (true) {
    if (aht_found) {
      sensors_event_t humidity, temp;
      if (xSemaphoreTake(i2cMutex, pdMS_TO_TICKS(1000)) == pdTRUE) {
        aht.getEvent(&humidity, &temp);
        xSemaphoreGive(i2cMutex);
        
        if (!isnan(temp.temperature) && !isnan(humidity.relative_humidity)) {
          dht_temp = temp.temperature;
          dht_hum  = humidity.relative_humidity;
          // Serial.printf("[AHT20] Nhiet do: %.1f*C | Do am: %.1f%%\n", (float)dht_temp, (float)dht_hum);

          const float humidity_threshold = (float)fan_humidity_threshold;
          const float humidity_recover_threshold = humidity_threshold - 5.0f;
          const bool humidity_recovered = humidity_telegram_alert_sent && dht_hum < humidity_recover_threshold;
          bool fan_turned_off = false;
          if (dht_hum >= humidity_threshold) {
            if (!humidity_telegram_alert_sent) {
              sendTelegramMessage("Canh bao do am cao: AHT20 = " + String((float)dht_hum, 1) +
                                  "%, nguong = " + String(humidity_threshold, 1) + "%");
              humidity_telegram_alert_sent = true;
            }
          }

          // Auto fan logic
          if (fan_auto_mode && mcp_found) {
            if (dht_hum >= humidity_threshold && !fan_state) {
              if (xSemaphoreTake(i2cMutex, pdMS_TO_TICKS(100)) == pdTRUE) {
                fan_state = true;
                mcp.digitalWrite(MCP_FAN_PIN, HIGH);
                xSemaphoreGive(i2cMutex);
                Serial.println("[FAN] Auto ON (do am cao)");
              }
            } else if (dht_hum < humidity_recover_threshold && fan_state) {
              if (xSemaphoreTake(i2cMutex, pdMS_TO_TICKS(100)) == pdTRUE) {
                fan_state = false;
                mcp.digitalWrite(MCP_FAN_PIN, LOW);
                xSemaphoreGive(i2cMutex);
                Serial.println("[FAN] Auto OFF (do am giam)");
                fan_turned_off = true;
              }
            }
          }

          if (humidity_recovered) {
            String msg = "Do am da giam: AHT20 = " + String((float)dht_hum, 1) + "%";
            if (fan_turned_off || !fan_state) {
              msg += ", quat da tat.";
            } else {
              msg += ".";
            }
            sendTelegramMessage(msg);
            humidity_telegram_alert_sent = false;
          }
        }
      }
    }
    vTaskDelay(pdMS_TO_TICKS(3000));
  }
}

// MQ2 Smoke Sensor Task
void mq2Task(void *pvParameters) {
  while (true) {
    int raw_mq2 = analogRead(MQ2_PIN);
    mq2_value = (raw_mq2 * 100) / 4095; // Convert 0-4095 to 0-100%
    smoke_detected = (mq2_value >= smoke_alarm_threshold);
    
    if (mcp_found && i2cMutex) {
      if (xSemaphoreTake(i2cMutex, pdMS_TO_TICKS(100)) == pdTRUE) {
        if (pump_auto_mode) {
          pump_state = smoke_detected;
        }
        if (pump_state) {
          mcp.digitalWrite(MCP_PUMP_PIN, HIGH);
          mcp.digitalWrite(MCP_BUZZER_PIN, HIGH);
        } else {
          mcp.digitalWrite(MCP_PUMP_PIN, LOW);
          mcp.digitalWrite(MCP_BUZZER_PIN, LOW);
        }
        xSemaphoreGive(i2cMutex);
      }
    }

    if (smoke_detected) {
      ESP_LOGW("MQ2", "SMOKE DETECTED! Value: %d%%", mq2_value);
      if (!smoke_telegram_alert_sent) {
        sendTelegramMessage("Canh bao chay: MQ2 = " + String(mq2_value) +
                            "%, nguong = " + String((float)smoke_alarm_threshold, 1) + "%");
        smoke_telegram_alert_sent = true;
      }
    } else {
      smoke_telegram_alert_sent = false;
    }
    vTaskDelay(pdMS_TO_TICKS(2000));
  }
}

static void appendLd2410Array(String &out, const uint8_t *values, int count) {
  out += '[';
  for (int i = 0; i < count; ++i) {
    if (i) out += ',';
    out += values[i];
  }
  out += ']';
}

String getLd2410DebugJson() {
  LD2410::BasicData basic{};
  LD2410::EngineeringData engineering{};
  LD2410::ConfigurationData config{};
  bool started = false;
  bool eng_enabled = false;
  bool cfg_loaded = false;

  if (ld2410Mutex && xSemaphoreTake(ld2410Mutex, pdMS_TO_TICKS(100)) == pdTRUE) {
    basic = ld2410_basic;
    engineering = ld2410_engineering;
    config = ld2410_config;
    started = ld2410_started;
    eng_enabled = ld2410_engineering_enabled;
    cfg_loaded = ld2410_config_loaded;
    xSemaphoreGive(ld2410Mutex);
  }

  String json;
  json.reserve(1600);
  json += "{\"ok\":";
  json += started ? "true" : "false";
  json += ",\"engineering_enabled\":";
  json += eng_enabled ? "true" : "false";
  json += ",\"config_loaded\":";
  json += cfg_loaded ? "true" : "false";
  json += ",\"basic_current\":";
  json += basic.isBasicDataCurrent() ? "true" : "false";
  json += ",\"engineering_current\":";
  json += engineering.isEngineeringDataCurrent() ? "true" : "false";
  json += ",\"target_state\":";
  json += static_cast<int>(basic.targetState);
  json += ",\"moving_distance\":";
  json += basic.movingTargetDistance;
  json += ",\"moving_energy\":";
  json += basic.movingTargetEnergy;
  json += ",\"stationary_distance\":";
  json += basic.stationaryTargetDistance;
  json += ",\"stationary_energy\":";
  json += basic.stationaryTargetEnergy;
  json += ",\"detection_distance\":";
  json += basic.detectionDistance;
  json += ",\"max_moving_gate\":";
  json += engineering.maxMovingGate;
  json += ",\"max_stationary_gate\":";
  json += engineering.maxStationaryGate;
  json += ",\"moving_gate_energy\":";
  appendLd2410Array(json, engineering.movingEnergyGates, LD2410_MAX_GATES);
  json += ",\"stationary_gate_energy\":";
  appendLd2410Array(json, engineering.stationaryEnergyGates, LD2410_MAX_GATES);
  json += ",\"moving_sensitivity\":";
  appendLd2410Array(json, config.motionSensitivity, 9);
  json += ",\"stationary_sensitivity\":";
  appendLd2410Array(json, config.stationarySensitivity, 9);
  json += "}";
  return json;
}

bool setLd2410GateSensitivity(uint8_t gate, uint8_t moving, uint8_t stationary) {
  if (!ld2410Mutex || gate > 8 || moving > 100 || stationary > 100) return false;
  bool ok = false;
  if (xSemaphoreTake(ld2410Mutex, pdMS_TO_TICKS(2500)) == pdTRUE) {
    ok = radar.setGateSensitivityThreshold(gate, moving, stationary);
    if (ok) {
      ld2410_config.motionSensitivity[gate] = moving;
      ld2410_config.stationarySensitivity[gate] = stationary;
      ld2410_config_loaded = true;
    }
    xSemaphoreGive(ld2410Mutex);
  }
  return ok;
}

bool refreshLd2410Configuration() {
  if (!ld2410Mutex) return false;
  bool ok = false;
  if (xSemaphoreTake(ld2410Mutex, pdMS_TO_TICKS(2500)) == pdTRUE) {
    ok = radar.readConfiguration();
    ld2410_config = radar.getCurrentConfiguration();
    ld2410_config_loaded = ok;
    xSemaphoreGive(ld2410Mutex);
  }
  return ok;
}

bool enableLd2410EngineeringMode() {
  if (!ld2410Mutex) return false;
  bool ok = false;
  if (xSemaphoreTake(ld2410Mutex, pdMS_TO_TICKS(2500)) == pdTRUE) {
    ok = radar.enableEngineeringMode();
    ld2410_engineering_enabled = ok;
    xSemaphoreGive(ld2410Mutex);
  }
  return ok;
}

// LD2410 mmWave Radar Task
void ld2410Task(void *pvParameters) {
  ld2410Mutex = xSemaphoreCreateMutex();
  if (!ld2410Mutex) {
    Serial.println("[LD2410] Mutex create FAILED!");
    vTaskDelete(NULL);
    return;
  }

  if (xSemaphoreTake(ld2410Mutex, pdMS_TO_TICKS(1000)) == pdTRUE) {
    ld2410_started = radar.beginUART(LD2410_RX_PIN, LD2410_TX_PIN, Serial1, 256000);
    Serial.printf("[LD2410] beginUART: %s\n", ld2410_started ? "OK" : "FAILED");
    ld2410_engineering_enabled = radar.enableEngineeringMode();
    Serial.printf("[LD2410] enableEngineeringMode: %s\n", ld2410_engineering_enabled ? "OK" : "FAILED");
    ld2410_config_loaded = radar.readConfiguration();
    ld2410_config = radar.getCurrentConfiguration();
    Serial.printf("[LD2410] readConfiguration: %s\n", ld2410_config_loaded ? "OK" : "FAILED");
    xSemaphoreGive(ld2410Mutex);
  }

  while (true) {
    LD2410::BasicData basic{};
    bool basic_current = false;

    if (xSemaphoreTake(ld2410Mutex, pdMS_TO_TICKS(20)) == pdTRUE) {
      radar.readSensorData(64);
      ld2410_basic = radar.getBasicData();
      ld2410_engineering = radar.getEngineeringData();
      basic = ld2410_basic;
      basic_current = basic.isBasicDataCurrent();
      xSemaphoreGive(ld2410Mutex);
    }

    if (basic_current) {
      presence_detected = basic.targetState != LD2410::TargetState::NO_TARGET;
      if (presence_detected) {
        mmwave_absence_started_ms = 0;
        presence_distance = (basic.movingTargetDistance > 0) ? basic.movingTargetDistance : basic.stationaryTargetDistance;
        if (occupant_count == 0 && millis() >= mmwave_cooldown_until_ms) {
          if (mmwave_presence_started_ms == 0) {
            mmwave_presence_started_ms = millis();
          } else if (!mmwave_alert_sent && millis() - mmwave_presence_started_ms >= 10000UL) {
            sendTelegramMessage("Canh bao: ke la dot nhap. Cam bien mmWave phat hien nguoi trong phong lien tuc qua 10 giay trong khi so nguoi trong phong = 0.");
            mmwave_alert_sent = true;
          }
        } else if (occupant_count > 0) {
          mmwave_presence_started_ms = 0;
          mmwave_alert_sent = false;
          mmwave_absence_started_ms = 0;
          mmwave_cooldown_until_ms = 0;
        } else {
          // occupant_count == 0 but still in cooldown period
          mmwave_presence_started_ms = 0;
        }
      } else {
        presence_distance = 0;
        if (mmwave_absence_started_ms == 0) {
          mmwave_absence_started_ms = millis();
        } else if (millis() - mmwave_absence_started_ms >= 2000UL) {
          mmwave_presence_started_ms = 0;
          mmwave_alert_sent = false;
          mmwave_absence_started_ms = 0;
        }
      }
    } else {
      presence_detected = false;
      presence_distance = 0;
      mmwave_presence_started_ms = 0;
      mmwave_alert_sent = false;
      mmwave_absence_started_ms = 0;
    }
    vTaskDelay(pdMS_TO_TICKS(50));
  }
}

void doorControlTask(void *pvParameters) {
  DoorMessage msg;
  while (true) {
    if (xQueueReceive(doorQueue, &msg, pdMS_TO_TICKS(100)) == pdTRUE) {
      if (msg.cmd == CMD_OPEN_DOOR) {
        Serial.printf("[DOOR] Open command received. method=%d actor=%s\n", msg.method, msg.actor);
        unlockDoor();
        if (msg.method == DM_FACE) {
          // Defer occupant_count++ until VL53L0X detects passthrough
          door_waiting_passthrough = true;
          door_passthrough_direction = 1; // entering
          buzzBeep(200);
          clearRfidAuthWindow();
        } else if (msg.method == DM_BUTTON) {
          // Defer occupant_count-- until VL53L0X detects passthrough
          door_waiting_passthrough = true;
          door_passthrough_direction = 2; // exiting
          buzzBeep(200);
        }
        add_event(msg.method, msg.actor);
        if (xSemaphoreTake(i2cMutex, portMAX_DELAY) == pdTRUE) {
          playSuccess();
          playWebURL();
          xSemaphoreGive(i2cMutex);
        }
      } else if (msg.cmd == CMD_PLAY_FAIL) {
        buzzBeep(100);
        vTaskDelay(pdMS_TO_TICKS(50));
        buzzBeep(100);
        if (xSemaphoreTake(i2cMutex, portMAX_DELAY) == pdTRUE) {
          playFail();
          playWebURL();
          xSemaphoreGive(i2cMutex);
        }
      } else if (msg.cmd == CMD_PLAY_UNLOCK) {
        buzzBeep(200);
        if (xSemaphoreTake(i2cMutex, portMAX_DELAY) == pdTRUE) {
          playUnlock();
          playWebURL();
          xSemaphoreGive(i2cMutex);
        }
      }
    }
    if (sendTelegramFlag) {
      sendTelegramFlag = false;
      sendTelegramPhoto(telegramActor);
    }
    if (rfid_auth_window_active && rfid_last_face_seen_ms > 0 &&
        (millis() - rfid_last_face_seen_ms) > 10000UL) {
      clearRfidAuthWindow();
      ESP_LOGW("RFID", "Authentication window expired: no face detected for 10s");
    }
    // Auto-lock is handled by VL53L0X task: <6cm for 4s after the door has opened.
  }
}

// VL53L0X ToF Sensor Task
// - Ignores the still-closed door for 10s immediately after unlock.
// - After the door is opened, <6cm for <4s is treated as a passthrough pulse.
// - <6cm held for 4s is treated as the door being closed and triggers auto-lock.
void vl53l0xTask(void *pvParameters) {
  bool close_range_active = false;
  bool door_open_seen = false;
  uint32_t close_range_start_ms = 0;

  // Periodic logging
  unsigned long last_log_ms = 0;

  while (true) {
    if (!vl53l0x_found) {
      vTaskDelay(pdMS_TO_TICKS(1000));
      continue;
    }

    // Read distance with I2C mutex
    uint16_t dist = 8190;
    if (xSemaphoreTake(i2cMutex, pdMS_TO_TICKS(200)) == pdTRUE) {
      dist = tof_sensor.readRangeSingleMillimeters();
      if (tof_sensor.timeoutOccurred()) {
        dist = 8190;
        ESP_LOGW("VL53L0X", "Timeout reading sensor");
      }
      xSemaphoreGive(i2cMutex);
    }
    vl53l0x_distance_mm = dist;

    // Log distance every 5 seconds
    if (millis() - last_log_ms >= 5000) {
      last_log_ms = millis();
      Serial.printf("[VL53L0X] Distance: %d mm | Door open: %s | Waiting passthrough: %s\n",
               dist, LockState ? "yes" : "no", door_waiting_passthrough ? "yes" : "no");
    }

    const uint32_t now = millis();
    const bool door_close_detected = (dist > 0 && dist < DOOR_CLOSE_DISTANCE_MM);

    if (!LockState) {
      close_range_active = false;
      close_range_start_ms = 0;
      door_open_seen = false;
      vTaskDelay(pdMS_TO_TICKS(50));
      continue;
    }

    if (!door_close_detected) {
      if (!door_open_seen) {
        ESP_LOGI("VL53L0X", "Door opened after unlock: %dmm", dist);
      }
      door_open_seen = true;

      if (close_range_active) {
        const uint32_t duration = now - close_range_start_ms;
        if (door_waiting_passthrough &&
            duration >= DOOR_PASS_PULSE_MIN_MS &&
            duration < DOOR_CLOSE_LOCK_HOLD_MS) {
          ESP_LOGI("VL53L0X", "Short close pulse %lums -> commit passthrough direction=%d",
                   (unsigned long)duration,
                   door_passthrough_direction);
          commitPendingPassthrough();
          door_waiting_passthrough = false;
          door_passthrough_direction = 0;
        }
        close_range_active = false;
        close_range_start_ms = 0;
      }

      vTaskDelay(pdMS_TO_TICKS(50));
      continue;
    }

    const bool still_in_unlock_grace =
        !door_open_seen && door_unlocked_at_ms > 0 && (now - door_unlocked_at_ms) < DOOR_UNLOCK_GRACE_MS;
    if (still_in_unlock_grace) {
      close_range_active = false;
      close_range_start_ms = 0;
      vTaskDelay(pdMS_TO_TICKS(50));
      continue;
    }

    if (!close_range_active) {
      close_range_active = true;
      close_range_start_ms = now;
      ESP_LOGI("VL53L0X", "Close range started: %dmm (threshold=%umm)",
               dist,
               (unsigned)DOOR_CLOSE_DISTANCE_MM);
    } else if (now - close_range_start_ms >= DOOR_CLOSE_LOCK_HOLD_MS) {
      Serial.println("[VL53L0X] Door close detected for 4s (<6cm) -> locking door");
      lockDoor();
      if (xSemaphoreTake(i2cMutex, portMAX_DELAY) == pdTRUE) {
        playWebURL();
        xSemaphoreGive(i2cMutex);
      }
      pending_entry_valid = false;
      pending_exit_valid = false;
      door_waiting_passthrough = false;
      door_passthrough_direction = 0;
      close_range_active = false;
      close_range_start_ms = 0;
      door_open_seen = false;
    }

    vTaskDelay(pdMS_TO_TICKS(50));
  }
}

void rfidTask(void *pvParameters) {
  uint8_t current_reader = RFID_READER_OUTER;
  unsigned long next_switch_ms = 0;
  selectRfidReader(current_reader);
  next_switch_ms = millis() + RFID_READER_SLICE_MS;

  while (true) {
    unsigned long now = millis();
    if (now >= next_switch_ms) {
      current_reader = (current_reader == RFID_READER_OUTER) ? RFID_READER_INNER : RFID_READER_OUTER;
      selectRfidReader(current_reader);
      next_switch_ms = now + RFID_READER_SLICE_MS;
    }

    if (current_reader == RFID_READER_OUTER) {
      readOuterRfidOnce();
    } else {
      readInnerRfidOnce();
    }

    vTaskDelay(pdMS_TO_TICKS(25));
  }
}

// --- Setup & Loop Arduino ---
void setup() {
  delay(1000); 
  Serial.begin(115200);
  Serial.println("\n--- He thong dang khoi dong ---");
  randomSeed(esp_random());
  esp_log_level_set("FbsLoader", ESP_LOG_ERROR);
  esp_log_level_set("dl::Model", ESP_LOG_ERROR);
  esp_log_level_set("httpd_txrx", ESP_LOG_ERROR);
  esp_log_level_set("APP_HTTPD", ESP_LOG_INFO);
  esp_log_level_set("ANTISPOOF", ESP_LOG_INFO);
  esp_log_level_set("HumanFaceFeat", ESP_LOG_INFO);
  esp_log_level_set("HumanFaceRecognizer", ESP_LOG_INFO);
  esp_log_level_set("CAMERA", ESP_LOG_INFO);
  esp_log_level_set("MEM", ESP_LOG_INFO);
  esp_log_level_set("MQ2", ESP_LOG_INFO);
  esp_log_level_set("RFID", ESP_LOG_INFO);
  esp_log_level_set("TELEGRAM", ESP_LOG_INFO);
  esp_log_level_set("VL53L0X", ESP_LOG_INFO);
  esp_log_level_set("NetworkClient", ESP_LOG_NONE);
  esp_log_level_set("NetworkClientSecure", ESP_LOG_ERROR);
  esp_log_level_set("gpio", ESP_LOG_NONE);
  prefs.begin("door_names", false);
  passkey = prefs.getString("login_pass", passkey);
  restoreLastKnownTime();
  occupantMutex = xSemaphoreCreateMutex();
  loadOccupants();
  
  // Queue & Mutex
  doorQueue = xQueueCreate(10, sizeof(DoorMessage));
  i2cMutex = xSemaphoreCreateMutex();

  // Watchdog timeout 15s for AI
  esp_task_wdt_config_t twdt_config = {
      .timeout_ms = 15000,
      .idle_core_mask = 0,
      .trigger_panic = true,
  };
  esp_task_wdt_reconfigure(&twdt_config);
  Serial.println("[WDT] Watchdog timeout increased to 15s");

  // AI Model Init
  ESP_LOGI("MEM", "Free Heap: %d bytes, PSRAM: %d bytes", xPortGetFreeHeapSize(), heap_caps_get_free_size(MALLOC_CAP_SPIRAM));
  if (antispoof_init("/spiffs/model/antispoof.espdl")) {
      Serial.println("AI Antispoof initialized successfully!");
  } else {
      Serial.println("AI Antispoof init FAILED!");
  }

  // RFID SPI
  SPI.begin(SPI_SCK, SPI_MISO, SPI_MOSI, SS_PIN);
  disableAllRfidReaders();
  selectRfidReader(RFID_READER_OUTER);

  // LittleFS
  if (!LittleFS.begin(false)) {
      Serial.println("[FS] LittleFS mount failed. Formatting...");
      LittleFS.format();
      if (!LittleFS.begin(false)) {
          Serial.println("[FS] LittleFS FATAL ERROR!");
      }
  } else {
      Serial.println("[FS] LittleFS mounted.");
      if (!LittleFS.exists("/db")) {
          Serial.println("[FS] Creating /db directory...");
          LittleFS.mkdir("/db");
      }
  }
  if (LittleFS.begin(false) && !LittleFS.exists("/db")) {
      Serial.println("[FS] Creating /db directory after recovery...");
      LittleFS.mkdir("/db");
  }

  // I2C Bus
  Wire.begin(SDA_PIN, SCL_PIN);
  Wire.setClock(100000);

  // OLED SSD1306
  if (xSemaphoreTake(i2cMutex, portMAX_DELAY) == pdTRUE) {
    if (!display.begin(SSD1306_SWITCHCAPVCC, 0x3C, false, false)) {
        Serial.println("[ERR] SSD1306 allocation failed");
        oled_found = false;
    } else {
        oled_found = true;
        Serial.println("[OK] SSD1306 initialized");
        display.clearDisplay();
        display.setTextSize(1);
        display.setTextColor(SSD1306_WHITE);
        display.setCursor(0, 0);
        display.println("Door Security");
        display.println("Dang khoi dong...");
        display.display();
    }
    xSemaphoreGive(i2cMutex);
  }

  // MCP23017
  if (xSemaphoreTake(i2cMutex, portMAX_DELAY) == pdTRUE) {
    if (!mcp.begin_I2C(MCP23017_ADDR, &Wire)) {
        Serial.println("[ERR] MCP23017 not found at 0x20!");
        mcp_found = false;
    } else {
        Serial.println("[OK] MCP23017 initialized");
        mcp_found = true;
        mcp.pinMode(MCP_LOCK_PIN, OUTPUT);    // A3 - Solenoid Lock
        mcp.pinMode(MCP_LED_PIN, OUTPUT);     // A4 - LED (AI face recognition)
        mcp.pinMode(MCP_BUZZER_PIN, OUTPUT);  // A5 - Buzzer
        mcp.pinMode(MCP_PUMP_PIN, OUTPUT);    // A7 - Water Pump
        mcp.pinMode(MCP_FAN_PIN, OUTPUT);     // B7 - Fan
        mcp.digitalWrite(MCP_LOCK_PIN, LOW);
        mcp.digitalWrite(MCP_LED_PIN, LOW);
        mcp.digitalWrite(MCP_BUZZER_PIN, LOW);
        mcp.digitalWrite(MCP_PUMP_PIN, LOW);
        mcp.digitalWrite(MCP_FAN_PIN, LOW);
    }
    xSemaphoreGive(i2cMutex);
  }

  // AHT20
  if (xSemaphoreTake(i2cMutex, portMAX_DELAY) == pdTRUE) {
    if (!aht.begin()) {
        Serial.println("[ERR] AHT20 not found at 0x38!");
        aht_found = false;
    } else {
        Serial.println("[OK] AHT20 initialized");
        aht_found = true;
    }
    xSemaphoreGive(i2cMutex);
  }

  // VL53L0X ToF Sensor
  if (xSemaphoreTake(i2cMutex, portMAX_DELAY) == pdTRUE) {
    tof_sensor.setBus(&Wire);
    tof_sensor.setTimeout(500);
    if (!tof_sensor.init()) {
        Serial.println("[ERR] VL53L0X not found at 0x29!");
        vl53l0x_found = false;
    } else {
        Serial.println("[OK] VL53L0X initialized");
        vl53l0x_found = true;
        tof_sensor.setMeasurementTimingBudget(20000); // 20ms for fast reads
    }
    xSemaphoreGive(i2cMutex);
  }

  // MQ2 Analog Pin
  pinMode(MQ2_PIN, INPUT);
  Serial.println("[OK] MQ2 pin configured (GPIO 14)");

  // Camera
  camera_config_t config;
  config.ledc_channel = LEDC_CHANNEL_0;
  config.ledc_timer = LEDC_TIMER_0;
  config.pin_d0 = Y2_GPIO_NUM;
  config.pin_d1 = Y3_GPIO_NUM;
  config.pin_d2 = Y4_GPIO_NUM;
  config.pin_d3 = Y5_GPIO_NUM;
  config.pin_d4 = Y6_GPIO_NUM;
  config.pin_d5 = Y7_GPIO_NUM;
  config.pin_d6 = Y8_GPIO_NUM;
  config.pin_d7 = Y9_GPIO_NUM;
  config.pin_xclk = XCLK_GPIO_NUM;
  config.pin_pclk = PCLK_GPIO_NUM;
  config.pin_vsync = VSYNC_GPIO_NUM;
  config.pin_href = HREF_GPIO_NUM;
  config.pin_sscb_sda = SIOD_GPIO_NUM;
  config.pin_sscb_scl = SIOC_GPIO_NUM;
  config.pin_pwdn = PWDN_GPIO_NUM;
  config.pin_reset = RESET_GPIO_NUM;
  config.xclk_freq_hz = 20000000;
  config.pixel_format = PIXFORMAT_JPEG;
  
  if(psramFound()){
    config.frame_size = FRAMESIZE_QVGA;
    config.jpeg_quality = 12;
    config.fb_count = 2;
  } else {
    config.frame_size = FRAMESIZE_QVGA;
    config.jpeg_quality = 12;
    config.fb_count = 1;
  }

  esp_err_t err = esp_camera_init(&config);
  if (err != ESP_OK) {
    Serial.printf("Camera init failed with error 0x%x", err);
    return;
  }
  camera_fb_t *boot_fb = esp_camera_fb_get();
  if (!boot_fb) {
    ESP_LOGE("CAMERA", "Boot self-test failed: esp_camera_fb_get() returned null");
  } else {
    ESP_LOGI("CAMERA", "Boot self-test OK: %ux%u len=%u format=%d",
             boot_fb->width,
             boot_fb->height,
             (unsigned)boot_fb->len,
             (int)boot_fb->format);
    esp_camera_fb_return(boot_fb);
  }
  
  // WiFi
  Serial.println("Dang ket noi WiFi...");
  // WiFi.begin("ThanhTrung", "61baumac19@"); 
  WiFi.begin("Leo", "61baumac19@");
  // WiFi.begin("C120-TN-DIENTU", "999999999"); 
  // WiFi.begin("Phu Cuong", "0913080875"); 
  
  int retry_count = 0;
  while (WiFi.status() != WL_CONNECTED && retry_count < 60) {
    delay(500);
    Serial.print(".");
    retry_count++;
  }

  if (WiFi.status() == WL_CONNECTED) {
    WiFi.setSleep(false);
    esp_wifi_set_ps(WIFI_PS_NONE);
    Serial.println("\nWiFi connected!");
    Serial.print("IP Address: ");
    Serial.println(WiFi.localIP());
    configTime(gmtOffset_sec, daylightOffset_sec, ntpServer, ntpServer2, ntpServer3);

    if (oled_found && xSemaphoreTake(i2cMutex, pdMS_TO_TICKS(500)) == pdTRUE) {
      display.clearDisplay();
      display.setTextSize(1);
      display.setTextColor(SSD1306_WHITE);
      display.setCursor(0, 0);
      display.println("Door Security");
      display.setCursor(0, 16);
      display.println("WiFi Connected!");
      display.setCursor(0, 32);
      display.print("IP: ");
      display.println(WiFi.localIP());
      display.display();
      xSemaphoreGive(i2cMutex);
    }
  } else {
    Serial.println("\nWiFi connection FAILED (Timeout). Running in OFFLINE mode.");
    if (oled_found && xSemaphoreTake(i2cMutex, pdMS_TO_TICKS(500)) == pdTRUE) {
      display.clearDisplay();
      display.setTextSize(1);
      display.setTextColor(SSD1306_WHITE);
      display.setCursor(0, 0);
      display.println("Door Security");
      display.setCursor(0, 16);
      display.println("WiFi FAILED!");
      display.display();
      xSemaphoreGive(i2cMutex);
    }
  }
  
  startCameraServer();
  logNetworkEndpoints();
  
  // Create FreeRTOS Tasks
  xTaskCreatePinnedToCore(doorControlTask, "doorCtrl", 8192, NULL, 3, &doorControlTaskHandle, 0);
  xTaskCreatePinnedToCore(rfidTask, "rfidTask", 4096, NULL, 2, &rfidTaskHandle, 0);
  xTaskCreatePinnedToCore(ahtTask, "ahtTask", 4096, NULL, 1, NULL, 0);
  xTaskCreatePinnedToCore(mq2Task, "mq2Task", 4096, NULL, 1, NULL, 0);
  if (kEnableLd2410) {
    xTaskCreatePinnedToCore(ld2410Task, "ld2410Task", 4096, NULL, 1, NULL, 0);
  }
  if (oled_found) {
    xTaskCreatePinnedToCore(oledStatusTask, "oledTask", 4096, NULL, 1, NULL, 0);
  }
  xTaskCreatePinnedToCore(timeKeeperTask, "timeKeep", 3072, NULL, 1, NULL, 0);
  if (vl53l0x_found) {
    xTaskCreatePinnedToCore(vl53l0xTask, "vl53l0xTask", 4096, NULL, 2, NULL, 0);
  }
  
  if (xSemaphoreTake(i2cMutex, portMAX_DELAY) == pdTRUE) {
      playWebURL();
      xSemaphoreGive(i2cMutex);
  }

  Serial.println("Setup complete!");
}

void loop() {
  static unsigned long lastIpLogMs = 0;

  // Log IP address every 10 seconds
  if (millis() - lastIpLogMs >= 10000) {
    lastIpLogMs = millis();
    if (WiFi.status() == WL_CONNECTED) {
      Serial.printf("[IP] %s\n", WiFi.localIP().toString().c_str());
    } else {
      Serial.println("[IP] WiFi disconnected");
    }
  }

  vTaskDelay(pdMS_TO_TICKS(50));
}

static void refreshOledStatus() {
  if (!oled_found) {
    return;
  }

  if (LockState && FRAME_COUNT_UNLOCK > 0) {
    display.clearDisplay();
    display.drawBitmap(32, 0, frames_unlock[oled_unlock_anim_frame], FRAME_WIDTH_UNLOCK, FRAME_HEIGHT_UNLOCK, SSD1306_WHITE);
    display.display();
    oled_unlock_anim_frame = (oled_unlock_anim_frame + 1) % FRAME_COUNT_UNLOCK;
    return;
  }
  oled_unlock_anim_frame = 0;

  char guidance[32] = "";
  uint32_t guidance_until = 0;
  portENTER_CRITICAL(&face_guidance_mux);
  snprintf(guidance, sizeof(guidance), "%s", face_guidance_text);
  guidance_until = face_guidance_until_ms;
  portEXIT_CRITICAL(&face_guidance_mux);

  display.clearDisplay();
  display.setTextSize(1);
  display.setTextColor(SSD1306_WHITE);
  display.setCursor(0, 0);
  display.println("Door Security");

  if (guidance[0] != '\0' && millis() < guidance_until) {
    display.setCursor(0, 16);
    display.println("Huong dan AI:");
    display.setCursor(0, 32);
    display.println(guidance);
    display.setCursor(0, 48);
    display.println("Nhin thang camera");
    display.display();
    return;
  }

  display.setCursor(0, 16);
  if (WiFi.status() == WL_CONNECTED) {
    display.print("IP: ");
    display.println(WiFi.localIP());
  } else {
    display.println("IP: offline");
  }

  display.setCursor(0, 32);
  display.print("T: ");
  if (isnan(dht_temp)) display.print("--.-");
  else display.print(dht_temp, 1);
  display.print("C H: ");
  if (isnan(dht_hum)) display.print("--");
  else display.print(dht_hum, 0);
  display.println("%");

  display.setCursor(0, 48);
  display.print("Smoke: ");
  display.print(mq2_value);
  display.print("%");
  if (smoke_detected) display.print(" !");

  display.display();
}

void oledStatusTask(void *pvParameters) {
  while (true) {
    if (oled_found && xSemaphoreTake(i2cMutex, pdMS_TO_TICKS(200)) == pdTRUE) {
      refreshOledStatus();
      xSemaphoreGive(i2cMutex);
    }
    vTaskDelay(pdMS_TO_TICKS(LockState ? FRAME_DELAY : 1000));
  }
}

void updateFaceGuidanceText(const char *text, uint32_t ttl_ms) {
  if (!text || text[0] == '\0') {
    return;
  }
  portENTER_CRITICAL(&face_guidance_mux);
  snprintf(face_guidance_text, sizeof(face_guidance_text), "%s", text);
  face_guidance_until_ms = millis() + ttl_ms;
  portEXIT_CRITICAL(&face_guidance_mux);
}

// Helper implementations
void unlockDoor() { 
  if (mcp_found) {
    if (xSemaphoreTake(i2cMutex, pdMS_TO_TICKS(100)) == pdTRUE) {
      mcp.digitalWrite(MCP_LOCK_PIN, HIGH); 
      xSemaphoreGive(i2cMutex);
    }
  }
  door_unlocked_at_ms = millis();
  LockState = true; 
}

void lockDoor() { 
  if (mcp_found) {
    if (xSemaphoreTake(i2cMutex, pdMS_TO_TICKS(100)) == pdTRUE) {
      mcp.digitalWrite(MCP_LOCK_PIN, LOW); 
      xSemaphoreGive(i2cMutex);
    }
  }
  door_unlocked_at_ms = 0;
  LockState = false; 
}

void playWebURL() { 
  if (!oled_found) return;
  refreshOledStatus();
}
void playSuccess() {
  if (!oled_found || FRAME_COUNT_SUCCESS <= 0) return;
  for (int frame = 0; frame < FRAME_COUNT_SUCCESS; ++frame) {
    display.clearDisplay();
    display.drawBitmap(32, 0, frames_success[frame], FRAME_WIDTH_SUCCESS, FRAME_HEIGHT_SUCCESS, SSD1306_WHITE);
    display.display();
    vTaskDelay(pdMS_TO_TICKS(FRAME_DELAY));
  }
}
void playFail() { /* OLED Fail Anim */ }
void playUnlock() {
  if (!oled_found || FRAME_COUNT_UNLOCK <= 0) return;
  display.clearDisplay();
  display.drawBitmap(32, 0, frames_unlock[0], FRAME_WIDTH_UNLOCK, FRAME_HEIGHT_UNLOCK, SSD1306_WHITE);
  display.display();
}

void buzzBeep(unsigned long duration_ms) {
  if (!mcp_found || !i2cMutex) return;
  if (xSemaphoreTake(i2cMutex, pdMS_TO_TICKS(300)) == pdTRUE) {
    mcp.digitalWrite(MCP_BUZZER_PIN, HIGH);
    xSemaphoreGive(i2cMutex);
  }
  vTaskDelay(pdMS_TO_TICKS(duration_ms));
  if (xSemaphoreTake(i2cMutex, pdMS_TO_TICKS(300)) == pdTRUE) {
    mcp.digitalWrite(MCP_BUZZER_PIN, LOW);
    xSemaphoreGive(i2cMutex);
  }
}

static void logNetworkEndpoints() {
  if (WiFi.status() != WL_CONNECTED) {
    return;
  }
  IPAddress ip = WiFi.localIP();
  Serial.printf("Dashboard URL: http://%u.%u.%u.%u/\n", ip[0], ip[1], ip[2], ip[3]);
  Serial.printf("Stream URL: http://%u.%u.%u.%u:81/stream\n", ip[0], ip[1], ip[2], ip[3]);
}

// add_event() is implemented in app_httpd.cpp (used for logs/dashboard).

void telegramTask(void *pvParameters) {
  TelegramTaskData *data = static_cast<TelegramTaskData *>(pvParameters);
  const String caption = data->caption;
  const uint8_t *fbBuf = data->jpg_buf;
  const size_t fbLen = data->jpg_len;

  WiFiClientSecure client;
  client.setInsecure();
  client.setHandshakeTimeout(TELEGRAM_CONNECT_TIMEOUT_MS / 1000);

  const size_t free_internal_before = heap_caps_get_free_size(MALLOC_CAP_INTERNAL);
  const size_t largest_internal_before = heap_caps_get_largest_free_block(MALLOC_CAP_INTERNAL);
  ESP_LOGI("TELEGRAM",
           "Upload start. free_internal=%u largest_internal=%u jpg_len=%u",
           static_cast<unsigned>(free_internal_before),
           static_cast<unsigned>(largest_internal_before),
           static_cast<unsigned>(fbLen));

  ESP_LOGI("TELEGRAM", "Connecting to Telegram API for photo upload...");
  if (!client.connect("api.telegram.org", 443)) {
    ESP_LOGE("TELEGRAM",
             "Connection to Telegram failed. free_internal=%u largest_internal=%u",
             static_cast<unsigned>(heap_caps_get_free_size(MALLOC_CAP_INTERNAL)),
             static_cast<unsigned>(heap_caps_get_largest_free_block(MALLOC_CAP_INTERNAL)));
    client.stop();
    free(data->jpg_buf);
    delete data;
    telegramUploadInProgress = false;
    vTaskDelete(NULL);
    return;
  }

  const char *part_chat =
      "--ESP32Boundary\r\nContent-Disposition: form-data; name=\"chat_id\"\r\n\r\n";
  const char *part_caption =
      "\r\n--ESP32Boundary\r\nContent-Disposition: form-data; name=\"caption\"\r\n\r\n";
  const char *part_photo =
      "\r\n--ESP32Boundary\r\nContent-Disposition: form-data; name=\"photo\"; filename=\"esp32-cam.jpg\"\r\n"
      "Content-Type: image/jpeg\r\n\r\n";
  const char *tail = "\r\n--ESP32Boundary--\r\n";
  const size_t totalLen =
      strlen(part_chat) + CHAT_ID.length() +
      strlen(part_caption) + caption.length() +
      strlen(part_photo) + fbLen + strlen(tail);

  client.println("POST /bot" + BOT_TOKEN + "/sendPhoto HTTP/1.1");
  client.println("Host: api.telegram.org");
  client.println("Content-Length: " + String(totalLen));
  client.println("Content-Type: multipart/form-data; boundary=ESP32Boundary");
  client.println("Connection: close");
  client.println();
  client.print(part_chat);
  client.print(CHAT_ID);
  client.print(part_caption);
  client.print(caption);
  client.print(part_photo);

  for (size_t offset = 0; offset < fbLen; offset += 512) {
    const size_t chunk = (offset + 512 < fbLen) ? 512 : (fbLen - offset);
    client.write(fbBuf + offset, chunk);
    vTaskDelay(pdMS_TO_TICKS(1));
  }
  client.print(tail);

  bool http_ok = false;
  bool telegram_ok = false;
  String response_tail;
  const uint32_t start = millis();
  response_tail.reserve(256);
  while ((client.connected() || client.available()) && (millis() - start < TELEGRAM_RESPONSE_TIMEOUT_MS)) {
    while (client.available()) {
      char c = static_cast<char>(client.read());
      response_tail += c;
      if (response_tail.length() > 256) {
        response_tail.remove(0, response_tail.length() - 256);
      }
      if (!http_ok && response_tail.indexOf("HTTP/1.1 200 OK") >= 0) {
        http_ok = true;
      }
      if (!telegram_ok && response_tail.indexOf("\"ok\":true") >= 0) {
        telegram_ok = true;
      }
    }
    vTaskDelay(pdMS_TO_TICKS(10));
  }
  client.stop();

  if (http_ok && telegram_ok) {
    ESP_LOGI("TELEGRAM", "Photo upload completed successfully.");
  } else {
    ESP_LOGE("TELEGRAM",
             "Photo upload failed. free_internal=%u largest_internal=%u tail=%s",
             static_cast<unsigned>(heap_caps_get_free_size(MALLOC_CAP_INTERNAL)),
             static_cast<unsigned>(heap_caps_get_largest_free_block(MALLOC_CAP_INTERNAL)),
             response_tail.c_str());
  }

  free(data->jpg_buf);
  delete data;
  telegramUploadInProgress = false;
  vTaskDelete(NULL);
}

void sendTelegramPhoto(String caption) {
  if (telegramUploadInProgress) {
    ESP_LOGW("TELEGRAM", "Skipping upload because previous Telegram task is still running");
    return;
  }

  camera_fb_t *fb = esp_camera_fb_get();
  if (!fb) {
    ESP_LOGE("TELEGRAM", "Camera capture failed for Telegram");
    return;
  }

  TelegramTaskData *data = new TelegramTaskData();
  data->caption = caption;

  uint8_t *annotated_buf = nullptr;
  size_t annotated_len = 0;
  if (buildTelegramJpegWithOverlay(fb, &annotated_buf, &annotated_len)) {
    data->jpg_buf = annotated_buf;
    data->jpg_len = annotated_len;
  } else {
    if (fb->format != PIXFORMAT_JPEG) {
      ESP_LOGE("TELEGRAM", "Telegram upload requires JPEG frame, got format=%d", fb->format);
      delete data;
      esp_camera_fb_return(fb);
      return;
    }
    data->jpg_len = fb->len;
    data->jpg_buf = static_cast<uint8_t *>(ps_malloc(fb->len));
    if (data->jpg_buf == NULL) {
      ESP_LOGE("TELEGRAM", "Failed to allocate PSRAM buffer for Telegram photo (%u bytes)", static_cast<unsigned>(fb->len));
      delete data;
      esp_camera_fb_return(fb);
      return;
    }
    memcpy(data->jpg_buf, fb->buf, fb->len);
  }

  esp_camera_fb_return(fb);

  telegramUploadInProgress = true;
  BaseType_t task_ok = xTaskCreatePinnedToCore(telegramTask, "telegramTask", TELEGRAM_PHOTO_TASK_STACK, data, 1, NULL, 0);
  if (task_ok != pdPASS) {
    ESP_LOGE("TELEGRAM", "Failed to start telegramTask");
    free(data->jpg_buf);
    delete data;
    telegramUploadInProgress = false;
    return;
  }

  ESP_LOGI("TELEGRAM", "Telegram background upload task launched");
}

static String telegramUrlEncode(const String &input) {
  String encoded;
  encoded.reserve(input.length() * 3);
  const char *hex = "0123456789ABCDEF";
  for (size_t i = 0; i < input.length(); ++i) {
    const unsigned char c = static_cast<unsigned char>(input[i]);
    if ((c >= 'a' && c <= 'z') ||
        (c >= 'A' && c <= 'Z') ||
        (c >= '0' && c <= '9') ||
        c == '-' || c == '_' || c == '.' || c == '~') {
      encoded += static_cast<char>(c);
    } else if (c == ' ') {
      encoded += "%20";
    } else {
      encoded += '%';
      encoded += hex[(c >> 4) & 0x0F];
      encoded += hex[c & 0x0F];
    }
  }
  return encoded;
}

void telegramMessageTask(void *pvParameters) {
  String *message = static_cast<String *>(pvParameters);
  if (!message) {
    vTaskDelete(NULL);
    return;
  }

  WiFiClientSecure client;
  client.setInsecure();
  client.setHandshakeTimeout(TELEGRAM_CONNECT_TIMEOUT_MS / 1000);

  ESP_LOGI("TELEGRAM", "Connecting to Telegram API for text message...");
  if (!client.connect("api.telegram.org", 443)) {
    ESP_LOGE("TELEGRAM",
             "Connection to Telegram failed for text message. free_internal=%u largest_internal=%u",
             static_cast<unsigned>(heap_caps_get_free_size(MALLOC_CAP_INTERNAL)),
             static_cast<unsigned>(heap_caps_get_largest_free_block(MALLOC_CAP_INTERNAL)));
    client.stop();
    delete message;
    vTaskDelete(NULL);
    return;
  }

  const String path = "/bot" + BOT_TOKEN + "/sendMessage?chat_id=" + CHAT_ID + "&text=" + telegramUrlEncode(*message);
  client.println("GET " + path + " HTTP/1.1");
  client.println("Host: api.telegram.org");
  client.println("Connection: close");
  client.println();

  unsigned long start = millis();
  while (client.connected() && !client.available() && millis() - start < 8000UL) {
    delay(10);
  }

  String response;
  while (client.available()) {
    response += client.readStringUntil('\n');
    response += '\n';
  }
  client.stop();
  ESP_LOGI("TELEGRAM", "Text message response:\n%s", response.c_str());

  delete message;
  vTaskDelete(NULL);
}

void sendTelegramMessage(String message) {
  String *payload = new String(message);
  if (!payload) {
    ESP_LOGE("TELEGRAM", "Failed to allocate Telegram text payload");
    return;
  }

  BaseType_t task_ok = xTaskCreatePinnedToCore(telegramMessageTask, "telegramMsgTask", TELEGRAM_TEXT_TASK_STACK, payload, 1, NULL, 0);
  if (task_ok != pdPASS) {
    ESP_LOGE("TELEGRAM", "Failed to start telegramMessageTask");
    delete payload;
    return;
  }
}
