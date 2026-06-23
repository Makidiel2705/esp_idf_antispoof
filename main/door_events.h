#ifndef DOOR_EVENTS_H
#define DOOR_EVENTS_H

#include <Arduino.h>
#include "freertos/FreeRTOS.h"
#include "freertos/queue.h"
#include "freertos/semphr.h"
#include "freertos/task.h"

// Door Control Commands
typedef enum {
    CMD_NONE,
    CMD_OPEN_DOOR,
    CMD_PLAY_FAIL,
    CMD_PLAY_UNLOCK,
    CMD_PLAY_SUCCESS
} DoorCommand;

// Door Methods (for logging & actor identification)
typedef enum {
    DM_BUTTON,
    DM_RFID,
    DM_KEYPAD,
    DM_FACE,
    DM_WEB
} DoorMethod;

// Message structure for the door queue
typedef struct {
    DoorCommand cmd;
    DoorMethod method;
    char actor[32];
} DoorMessage;

// Global Handles (extern-ed for other components)
extern QueueHandle_t doorQueue;
extern SemaphoreHandle_t i2cMutex;
extern TaskHandle_t rfidTaskHandle;
extern TaskHandle_t doorControlTaskHandle;

// Shared variables (extern-ed)
extern volatile float dht_temp;
extern volatile float dht_hum;
extern bool RecognitionFace;
extern bool LockState;
extern bool sendTelegramFlag;
extern String telegramActor;
void append_log(const String &msg);
void notifyAiFrameEvent();
void updateFaceGuidanceText(const char *text, uint32_t ttl_ms = 2500);
void notifyRfidFaceSeen();
void recordAuthFailureForFace(int face_id, const String &reason);
void resetAuthFailureForFace(int face_id);

// New sensor variables
extern volatile int mq2_value;
extern volatile bool smoke_detected;
extern volatile bool pump_state;
extern volatile bool pump_auto_mode;
extern volatile float smoke_alarm_threshold;
extern volatile bool presence_detected;
extern volatile int presence_distance;
extern volatile bool fan_state;
extern volatile bool fan_auto_mode;
extern volatile float fan_humidity_threshold;
extern volatile int occupant_count;
extern volatile bool rfid_auth_window_active;
extern volatile int rfid_expected_face_id;
extern volatile int pending_rfid_enroll_face_id;
extern volatile uint8_t ai_auth_flush_frames_remaining;
extern volatile uint32_t ai_auth_warmup_until_ms;
extern volatile int vl53l0x_distance_mm;
extern bool vl53l0x_found;
extern volatile bool door_waiting_passthrough;

bool is_occupant_inside_by_face(int face_id);
bool is_occupant_inside_by_uid(const String &uid);
bool prepare_pending_entry(int face_id);
bool prepare_pending_exit_by_uid(const String &uid);
String get_occupants_json();
void clear_occupants();
void remove_occupant_by_face_id(int face_id);
void shift_occupants_after_face_delete(int deleted_face_id);

#endif
