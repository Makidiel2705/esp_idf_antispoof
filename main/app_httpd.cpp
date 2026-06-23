#include <Arduino.h>
#include "esp_task_wdt.h"
#include "esp_http_server.h"
#include "esp_timer.h"
#include "esp_camera.h"
#include "img_converters.h"
#include "fb_gfx.h"
#include "esp_log.h"
#include "sdkconfig.h"
#include <sys/time.h>

#include "door_events.h"
#include "antispoof_task.h"
#include "camera_index.h"
#include <algorithm>
#include <math.h>
#include <vector>
#include <list>
#include <Preferences.h>
#include <LittleFS.h>
#include <Adafruit_MCP23X17.h>

#include "human_face_detect.hpp"
#include "human_face_recognition.hpp"

static const char *TAG = "APP_HTTPD";
static const char *kFaceDbPathFs = "/db/faces.db";
static const char *kFaceDbPathVfs = "/littlefs/db/faces.db";

extern Preferences prefs;
extern String passkey;
extern bool LockState;
extern bool sendTelegramFlag;
extern String telegramActor;
extern String currentOTP;
extern volatile float dht_temp;
extern volatile float dht_hum;
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
extern bool mcp_found;
extern Adafruit_MCP23X17 mcp;
extern SemaphoreHandle_t i2cMutex;
extern String getLd2410DebugJson();
extern bool setLd2410GateSensitivity(uint8_t gate, uint8_t moving, uint8_t stationary);
extern bool refreshLd2410Configuration();
extern bool enableLd2410EngineeringMode();
#define MCP_FAN_PIN 15
#define MCP_LED_PIN 4
#define MCP_PUMP_PIN 7
void triggerSendOTP();
void setCameraStatusLed(bool enabled);

static constexpr uint32_t OTP_VALID_MS = 180000;
static uint32_t otp_deadline_ms = 0;

static int8_t is_enrolling = 0;
static int8_t detection_enabled = 1;
static int8_t recognition_enabled = 1;
static String pending_enroll_name = "";
static String pending_enroll_role = "";

static HumanFaceDetect *detector = nullptr;
static HumanFaceRecognizer *recognizer = nullptr;
static SemaphoreHandle_t recognizer_mutex = nullptr;

static TaskHandle_t ai_task_handle = nullptr;
static SemaphoreHandle_t ai_frame_mutex = nullptr;
static uint8_t *ai_frame_buffer = nullptr;
static size_t ai_frame_size = 0;
static bool ai_ready_to_process = false;
static int ai_frame_w = 0;
static int ai_frame_h = 0;
static SemaphoreHandle_t overlay_mutex = nullptr;

#define MAX_EVENTS 50
struct DoorEvent {
    uint32_t ts;
    uint8_t method;
    char actor[32];
};

static DoorEvent events[MAX_EVENTS];
static int events_count = 0;
static int events_next = 0;

#define PART_BOUNDARY "123456789000000000000987654321"
static const char *_STREAM_CONTENT_TYPE = "multipart/x-mixed-replace;boundary=" PART_BOUNDARY;
static const char *_STREAM_BOUNDARY = "\r\n--" PART_BOUNDARY "\r\n";
static const char *_STREAM_PART =
    "Content-Type: image/jpeg\r\nContent-Length: %u\r\nX-Timestamp: %ld.%06ld\r\n\r\n";

struct OverlayState;
struct RecognitionOutcome;

static esp_err_t parse_get(httpd_req_t *req, char **obuf);
static String urlDecode(const String &str);
static HumanFaceRecognizer *ensure_recognizer();
static void ensure_ai_background();
static void ai_analyzer_task(void *pvParameters);
static int current_ai_cadence_ms();
static void notify_ai_task();
static bool decode_to_rgb888(const uint8_t *src, size_t src_len, pixformat_t format, uint8_t *rgb_out);
static bool is_authenticated(httpd_req_t *req);
static bool is_step1_authenticated(httpd_req_t *req);
static esp_err_t redirect_to(httpd_req_t *req, const char *location);
static esp_err_t require_auth(httpd_req_t *req);
void append_log(const String &msg);
void sendTelegramPhoto(String caption);
static void add_event_internal(uint8_t method, const char *actor);
static String face_name_key(int id);
static String face_time_key(int id);
static String face_raw_key(int id);
static String face_rfid_key(int id);
static String face_role_key(int id);
static String face_fail_key(int id);
static String face_lock_key(int id);
static void sync_face_raw_map();
static void shift_face_metadata_left(int deleted_id, int old_count);
static esp_err_t setname_handler(httpd_req_t *req);
static esp_err_t arm_rfid_link_handler(httpd_req_t *req);
static esp_err_t cancel_rfid_link_handler(httpd_req_t *req);
static esp_err_t rfid_state_handler(httpd_req_t *req);
static esp_err_t clear_logs_handler(httpd_req_t *req);
static esp_err_t occupants_handler(httpd_req_t *req);
static esp_err_t sync_time_handler(httpd_req_t *req);
static esp_err_t time_status_handler(httpd_req_t *req);
static esp_err_t login_handler(httpd_req_t *req);
static esp_err_t do_login_handler(httpd_req_t *req);
static esp_err_t verify_2fa_handler(httpd_req_t *req);
static esp_err_t do_verify_2fa_handler(httpd_req_t *req);
static esp_err_t forgot_password_handler(httpd_req_t *req);
static esp_err_t reset_password_handler(httpd_req_t *req);
static esp_err_t logout_handler(httpd_req_t *req);
static esp_err_t setpass_handler(httpd_req_t *req);
static esp_err_t xclk_handler(httpd_req_t *req);
static esp_err_t reg_handler(httpd_req_t *req);
static esp_err_t greg_handler(httpd_req_t *req);
static esp_err_t pll_handler(httpd_req_t *req);
static esp_err_t win_handler(httpd_req_t *req);
static int parse_get_var(char *buf, const char *key, int def);
static int print_reg(char *p, sensor_t *s, uint16_t reg, uint32_t mask);
static void update_overlay_state(const std::list<dl::detect::result_t> *results, const RecognitionOutcome &outcome);
static void clear_overlay_state();
static bool snapshot_overlay_state(OverlayState *out);
static void draw_overlay(fb_data_t *fb, const OverlayState &state);
static String jsonEscape(const String &value);
static bool readJpegSize(const uint8_t *data, size_t len, int *width, int *height);
static bool run_liveness_for_face(dl::image::img_t *img, const dl::detect::result_t &face, RecognitionOutcome *outcome);

struct CropRegion {
    int left;
    int top;
    int right;
    int bottom;
    float requested_scale;
    float applied_scale;
};

enum OverlayStatus : uint8_t {
    OVERLAY_NONE = 0,
    OVERLAY_FAKE,
    OVERLAY_REAL_UNREGISTERED,
    OVERLAY_REAL_SUCCESS,
};

enum FaceGuideCode : uint8_t {
    FACE_GUIDE_OK = 0,
    FACE_GUIDE_TOO_FAR,
    FACE_GUIDE_TOO_CLOSE,
    FACE_GUIDE_MOVE_CENTER,
    FACE_GUIDE_HOLD_STILL,
};

struct FaceGuidance {
    FaceGuideCode code;
    const char *text;
    float center_dx_ratio;
    float center_dy_ratio;
};

struct OverlayState {
    bool active;
    OverlayStatus status;
    float real_score;
    int face_id;
    int box[4];
    int keypoint[10];
    char label[96];
    int64_t updated_us;
};

struct RecognitionOutcome {
    int id;
    float liveness_score;
    bool anti_ran;
    OverlayStatus overlay_status;
    char label[96];
};

static OverlayState g_overlay_state = {};

static constexpr float FACE_GUIDE_MIN_APPLIED_SCALE = 2.30f;
static constexpr int FACE_GUIDE_MIN_FACE_W = 50;
static constexpr int FACE_GUIDE_MIN_FACE_H = 70;
static constexpr int FACE_GUIDE_CLOSE_FACE_H = 140;
static constexpr float FACE_GUIDE_MAX_CENTER_OFFSET_RATIO_X = 0.20f;
static constexpr float FACE_GUIDE_MAX_CENTER_OFFSET_RATIO_Y = 0.20f;
static constexpr int64_t FACE_GUIDE_STABLE_US = 300000;

static FaceGuidance evaluate_face_guidance(
    int frame_w,
    int frame_h,
    int box_x,
    int box_y,
    int box_w,
    int box_h,
    const CropRegion &crop)
{
    FaceGuidance guide = {FACE_GUIDE_OK, "Giu yen", 0.0f, 0.0f};
    if (frame_w <= 0 || frame_h <= 0 || box_w <= 0 || box_h <= 0) {
        guide.code = FACE_GUIDE_MOVE_CENTER;
        guide.text = "Can mat vao khung";
        return guide;
    }

    const float face_cx = box_x + box_w / 2.0f;
    const float face_cy = box_y + box_h / 2.0f;
    guide.center_dx_ratio = (face_cx - frame_w / 2.0f) / frame_w;
    guide.center_dy_ratio = (face_cy - frame_h / 2.0f) / frame_h;
    const float abs_dx = fabsf(guide.center_dx_ratio);
    const float abs_dy = fabsf(guide.center_dy_ratio);

    // Distance first: small faces reduce both recognition detail and anti-spoof context quality.
    if (box_w < FACE_GUIDE_MIN_FACE_W || box_h < FACE_GUIDE_MIN_FACE_H) {
        guide.code = FACE_GUIDE_TOO_FAR;
        guide.text = "Lai gan hon";
        return guide;
    }

    // If 2.7x context cannot be kept and the face is already large, distance is the main issue.
    if (crop.applied_scale < FACE_GUIDE_MIN_APPLIED_SCALE && box_h >= FACE_GUIDE_CLOSE_FACE_H) {
        guide.code = FACE_GUIDE_TOO_CLOSE;
        guide.text = "Lui xa hon";
        return guide;
    }

    // Otherwise, clipped 2.7x context is usually caused by off-center placement.
    if (abs_dx > FACE_GUIDE_MAX_CENTER_OFFSET_RATIO_X || abs_dy > FACE_GUIDE_MAX_CENTER_OFFSET_RATIO_Y ||
        crop.applied_scale < FACE_GUIDE_MIN_APPLIED_SCALE) {
        guide.code = FACE_GUIDE_MOVE_CENTER;
        guide.text = "Can mat vao giua";
        return guide;
    }

    return guide;
}

static CropRegion compute_scaled_crop_region(int src_w, int src_h, int box_x, int box_y, int box_w, int box_h, float scale)
{
    CropRegion region = {box_x, box_y, box_x + box_w - 1, box_y + box_h - 1, scale, 1.0f};
    if (box_w <= 0 || box_h <= 0 || src_w <= 1 || src_h <= 1) {
        return region;
    }

    float safe_scale = std::min((src_h - 1.0f) / box_h, std::min((src_w - 1.0f) / box_w, scale));
    float new_width = box_w * safe_scale;
    float new_height = box_h * safe_scale;
    float center_x = box_x + box_w / 2.0f;
    float center_y = box_y + box_h / 2.0f;

    float left = center_x - new_width / 2.0f;
    float top = center_y - new_height / 2.0f;
    float right = center_x + new_width / 2.0f;
    float bottom = center_y + new_height / 2.0f;

    if (left < 0.0f) {
        right -= left;
        left = 0.0f;
    }
    if (top < 0.0f) {
        bottom -= top;
        top = 0.0f;
    }
    if (right > src_w - 1.0f) {
        left -= right - src_w + 1.0f;
        right = src_w - 1.0f;
    }
    if (bottom > src_h - 1.0f) {
        top -= bottom - src_h + 1.0f;
        bottom = src_h - 1.0f;
    }

    region.left = std::max(0, (int)left);
    region.top = std::max(0, (int)top);
    region.right = std::min(src_w - 1, (int)right);
    region.bottom = std::min(src_h - 1, (int)bottom);
    region.applied_scale = safe_scale;
    return region;
}

static uint32_t overlay_color_for_status(OverlayStatus status)
{
    switch (status) {
    case OVERLAY_FAKE:
        return 0xFF0000;
    case OVERLAY_REAL_SUCCESS:
        return 0x00FF00;
    case OVERLAY_REAL_UNREGISTERED:
        return 0xFFFF00;
    case OVERLAY_NONE:
    default:
        return 0xFFFF00;
    }
}

static uint32_t fb_color(fb_data_t *fb, uint32_t color)
{
    if (fb->bytes_per_pixel == 2) {
        return ((color >> 16) & 0x001F) | ((color >> 3) & 0x07E0) | ((color << 8) & 0xF800);
    }
    // fb_gfx on RGB888 expects byte order B,G,R in memory-backed color values.
    return ((color & 0x0000FF) << 16) | (color & 0x00FF00) | ((color & 0xFF0000) >> 16);
}

static void clear_overlay_state()
{
    if (!overlay_mutex) {
        return;
    }
    if (xSemaphoreTake(overlay_mutex, pdMS_TO_TICKS(50)) == pdTRUE) {
        g_overlay_state = {};
        xSemaphoreGive(overlay_mutex);
    }
}

static void update_overlay_state(const std::list<dl::detect::result_t> *results, const RecognitionOutcome &outcome)
{
    if (!overlay_mutex) {
        return;
    }
    if (!results || results->empty()) {
        clear_overlay_state();
        return;
    }

    const auto &face = results->front();
    OverlayState next = {};
    next.active = true;
    next.status = outcome.overlay_status;
    next.real_score = outcome.liveness_score;
    next.face_id = outcome.id;
    next.updated_us = esp_timer_get_time();
    next.box[0] = (int)face.box[0];
    next.box[1] = (int)face.box[1];
    next.box[2] = (int)face.box[2];
    next.box[3] = (int)face.box[3];
    for (int i = 0; i < 10; ++i) {
        next.keypoint[i] = (i < (int)face.keypoint.size()) ? (int)face.keypoint[i] : -1;
    }
    snprintf(next.label, sizeof(next.label), "%s", outcome.label);

    if (xSemaphoreTake(overlay_mutex, pdMS_TO_TICKS(50)) == pdTRUE) {
        g_overlay_state = next;
        xSemaphoreGive(overlay_mutex);
    }
}

static bool snapshot_overlay_state(OverlayState *out)
{
    if (!out || !overlay_mutex) {
        return false;
    }
    if (xSemaphoreTake(overlay_mutex, pdMS_TO_TICKS(20)) != pdTRUE) {
        return false;
    }
    *out = g_overlay_state;
    xSemaphoreGive(overlay_mutex);
    if (!out->active) {
        return false;
    }
    // Avoid drawing stale boxes forever if AI has not refreshed.
    if ((esp_timer_get_time() - out->updated_us) > 3000000LL) {
        return false;
    }
    return true;
}

static void draw_overlay(fb_data_t *fb, const OverlayState &state)
{
    if (!fb || !state.active) {
        return;
    }

    uint32_t color = fb_color(fb, overlay_color_for_status(state.status));
    int x = state.box[0];
    int y = state.box[1];
    int w = state.box[2] - state.box[0] + 1;
    int h = state.box[3] - state.box[1] + 1;
    if (x < 0 || y < 0 || w <= 0 || h <= 0) {
        return;
    }
    if ((x + w) > fb->width) {
        w = fb->width - x;
    }
    if ((y + h) > fb->height) {
        h = fb->height - y;
    }
    if (w <= 0 || h <= 0) {
        return;
    }

    fb_gfx_drawFastHLine(fb, x, y, w, color);
    fb_gfx_drawFastHLine(fb, x, y + h - 1, w, color);
    fb_gfx_drawFastVLine(fb, x, y, h, color);
    fb_gfx_drawFastVLine(fb, x + w - 1, y, h, color);

    for (int i = 0; i < 10; i += 2) {
        if (state.keypoint[i] >= 0 && state.keypoint[i + 1] >= 0) {
            fb_gfx_fillRect(fb, state.keypoint[i], state.keypoint[i + 1], 3, 3, color);
        }
    }

    int text_x = x;
    int text_y = y - 16;
    if (text_y < 4) {
        text_y = y + h + 4;
        if (text_y > fb->height - 16) {
            text_y = fb->height - 16;
        }
    }
    fb_gfx_print(fb, text_x, text_y, color, state.label);
}

extern "C" bool buildTelegramJpegWithOverlay(camera_fb_t *fb, uint8_t **out_buf, size_t *out_len)
{
    if (!fb || !out_buf || !out_len) {
        return false;
    }
    *out_buf = nullptr;
    *out_len = 0;

    OverlayState overlay = {};
    if (!snapshot_overlay_state(&overlay)) {
        return false;
    }

    if (ai_frame_w > 0 && ai_frame_h > 0 && (fb->width != ai_frame_w || fb->height != ai_frame_h)) {
        const float sx = (float)fb->width / (float)ai_frame_w;
        const float sy = (float)fb->height / (float)ai_frame_h;
        overlay.box[0] = (int)lroundf(overlay.box[0] * sx);
        overlay.box[1] = (int)lroundf(overlay.box[1] * sy);
        overlay.box[2] = (int)lroundf(overlay.box[2] * sx);
        overlay.box[3] = (int)lroundf(overlay.box[3] * sy);
        for (int i = 0; i < 10; i += 2) {
            if (overlay.keypoint[i] >= 0 && overlay.keypoint[i + 1] >= 0) {
                overlay.keypoint[i] = (int)lroundf(overlay.keypoint[i] * sx);
                overlay.keypoint[i + 1] = (int)lroundf(overlay.keypoint[i + 1] * sy);
            }
        }
    }

    const size_t rgb_len = (size_t)fb->width * (size_t)fb->height * 3;
    uint8_t *rgb_buf = static_cast<uint8_t *>(heap_caps_malloc(rgb_len, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT));
    if (!rgb_buf) {
        ESP_LOGW(TAG, "TELEGRAM_OVERLAY: Failed to allocate RGB buffer (%u bytes)", (unsigned)rgb_len);
        return false;
    }

    bool converted = fmt2rgb888(fb->buf, fb->len, fb->format, rgb_buf);
    if (!converted) {
        ESP_LOGW(TAG, "TELEGRAM_OVERLAY: Failed to convert frame for overlay (format=%d)", (int)fb->format);
        free(rgb_buf);
        return false;
    }

    fb_data_t rfb;
    rfb.width = fb->width;
    rfb.height = fb->height;
    rfb.bytes_per_pixel = 3;
    rfb.format = FB_RGB888;
    rfb.data = rgb_buf;
    draw_overlay(&rfb, overlay);

    bool encoded = fmt2jpg(rgb_buf, rgb_len, fb->width, fb->height, PIXFORMAT_RGB888, 90, out_buf, out_len);
    free(rgb_buf);
    if (!encoded || !*out_buf || *out_len == 0) {
        if (*out_buf) {
            free(*out_buf);
            *out_buf = nullptr;
        }
        *out_len = 0;
        ESP_LOGW(TAG, "TELEGRAM_OVERLAY: Failed to encode annotated JPEG");
        return false;
    }
    return true;
}

static esp_err_t parse_get(httpd_req_t *req, char **obuf)
{
    size_t buf_len = httpd_req_get_url_query_len(req) + 1;
    if (buf_len <= 1) return ESP_FAIL;
    char *buf = static_cast<char *>(malloc(buf_len));
    if (!buf) return ESP_ERR_NO_MEM;
    if (httpd_req_get_url_query_str(req, buf, buf_len) != ESP_OK) {
        free(buf);
        return ESP_FAIL;
    }
    *obuf = buf;
    return ESP_OK;
}

static String urlDecode(const String &str)
{
    String out;
    char c = 0;
    int j = 0;
    for (int i = 0; i < (int)str.length(); ++i) {
        if (str[i] == '%') {
            sscanf(str.substring(i + 1, i + 3).c_str(), "%x", &j);
            c = (char)j;
            out += c;
            i += 2;
        } else if (str[i] == '+') {
            out += ' ';
        } else {
            out += str[i];
        }
    }
    return out;
}

static String jsonEscape(const String &value)
{
    String escaped;
    escaped.reserve(value.length() + 8);
    for (size_t i = 0; i < value.length(); ++i) {
        char c = value[i];
        switch (c) {
        case '\\': escaped += "\\\\"; break;
        case '"': escaped += "\\\""; break;
        case '\b': escaped += "\\b"; break;
        case '\f': escaped += "\\f"; break;
        case '\n': escaped += "\\n"; break;
        case '\r': escaped += "\\r"; break;
        case '\t': escaped += "\\t"; break;
        default:
            if ((uint8_t)c < 0x20) {
                char buf[7];
                snprintf(buf, sizeof(buf), "\\u%04x", (unsigned char)c);
                escaped += buf;
            } else {
                escaped += c;
            }
            break;
        }
    }
    return escaped;
}

static bool readJpegSize(const uint8_t *data, size_t len, int *width, int *height)
{
    if (!data || len < 4 || data[0] != 0xFF || data[1] != 0xD8) return false;
    size_t pos = 2;
    while (pos + 9 < len) {
        while (pos < len && data[pos] != 0xFF) pos++;
        while (pos < len && data[pos] == 0xFF) pos++;
        if (pos >= len) return false;
        uint8_t marker = data[pos++];
        if (marker == 0xD9 || marker == 0xDA) return false;
        if (pos + 2 > len) return false;
        uint16_t seg_len = ((uint16_t)data[pos] << 8) | data[pos + 1];
        if (seg_len < 2 || pos + seg_len > len) return false;
        bool is_sof = (marker >= 0xC0 && marker <= 0xCF && marker != 0xC4 && marker != 0xC8 && marker != 0xCC);
        if (is_sof) {
            if (seg_len < 7) return false;
            *height = ((uint16_t)data[pos + 3] << 8) | data[pos + 4];
            *width = ((uint16_t)data[pos + 5] << 8) | data[pos + 6];
            return *width > 0 && *height > 0;
        }
        pos += seg_len;
    }
    return false;
}

static String face_name_key(int id) { return "id_" + String(id); }
static String face_time_key(int id) { return "time_" + String(id); }
static String face_raw_key(int id) { return "raw_" + String(id); }
static String face_rfid_key(int id) { return "rfid_" + String(id); }
static String face_role_key(int id) { return "role_" + String(id); }
static String face_fail_key(int id) { return "fail_" + String(id); }
static String face_lock_key(int id) { return "lock_" + String(id); }

static HumanFaceRecognizer *ensure_recognizer()
{
    if (!recognizer) {
        recognizer = new HumanFaceRecognizer(kFaceDbPathVfs);
    }
    if (!recognizer_mutex) recognizer_mutex = xSemaphoreCreateMutex();
    return recognizer;
}

static void ensure_ai_background()
{
    if (!detector) detector = new HumanFaceDetect();
    if (!overlay_mutex) overlay_mutex = xSemaphoreCreateMutex();
    if (!ai_frame_buffer) {
        ai_frame_size = 320 * 240 * 3;
        ai_frame_buffer = static_cast<uint8_t *>(heap_caps_malloc(ai_frame_size, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT));
        ai_frame_mutex = xSemaphoreCreateMutex();
        xTaskCreatePinnedToCore(ai_analyzer_task, "ai_analyzer", 16384, nullptr, 5, &ai_task_handle, 1);
    }
}

void notifyAiFrameEvent()
{
    notify_ai_task();
}

static void notify_ai_task()
{
    if (ai_task_handle) {
        xTaskNotifyGive(ai_task_handle);
    }
}

static int current_ai_cadence_ms()
{
    if (rfid_auth_window_active) return 150;
    if (is_enrolling) return 200;
    return 500;
}

static bool decode_to_rgb888(const uint8_t *src, size_t src_len, pixformat_t format, uint8_t *rgb_out)
{
    if (!src || !src_len || !rgb_out) return false;
    // fmt2rgb888 already decodes JPEG directly to RGB888, avoiding the old
    // JPEG -> RGB565 -> RGB888 expansion pass in the AI hot path.
    return fmt2rgb888(src, src_len, format, rgb_out);
}

static String current_log_time_string()
{
    time_t now;
    struct tm timeinfo;
    time(&now);
    if (now < 1704067200) {
        return "Chua dong bo (+" + String(millis() / 1000) + "s)";
    }
    localtime_r(&now, &timeinfo);
    char timeStr[64];
    strftime(timeStr, sizeof(timeStr), "%Y-%m-%d %H:%M:%S", &timeinfo);
    return String(timeStr);
}

static bool is_valid_unix_time(time_t ts)
{
    return ts >= 1704067200; // 2024-01-01
}

static void set_system_unix_time(uint32_t unix_ts)
{
    struct timeval tv = {};
    tv.tv_sec = static_cast<time_t>(unix_ts);
    tv.tv_usec = 0;
    settimeofday(&tv, nullptr);
    prefs.putUInt("last_unix", unix_ts);
}

static bool get_cookie_flag(httpd_req_t *req, const char *needle)
{
    size_t len = httpd_req_get_hdr_value_len(req, "Cookie");
    if (len == 0) return false;
    char *cookie = static_cast<char *>(malloc(len + 1));
    if (!cookie) return false;
    bool ok = false;
    if (httpd_req_get_hdr_value_str(req, "Cookie", cookie, len + 1) == ESP_OK) ok = strstr(cookie, needle) != nullptr;
    free(cookie);
    return ok;
}

static bool is_authenticated(httpd_req_t *req) { return get_cookie_flag(req, "auth=2"); }
static bool is_step1_authenticated(httpd_req_t *req) { return get_cookie_flag(req, "auth=1") || get_cookie_flag(req, "auth=2"); }

static void issue_otp()
{
    currentOTP = String(random(100000, 999999));
    otp_deadline_ms = millis() + OTP_VALID_MS;
    triggerSendOTP();
}

static int otp_remaining_seconds()
{
    if (currentOTP.length() == 0 || otp_deadline_ms == 0) return 0;
    int32_t remaining_ms = static_cast<int32_t>(otp_deadline_ms - millis());
    if (remaining_ms <= 0) return 0;
    return (remaining_ms + 999) / 1000;
}

static bool otp_is_valid(const char *otp)
{
    return otp_remaining_seconds() > 0 && currentOTP.length() > 0 && strcmp(otp, currentOTP.c_str()) == 0;
}

static void clear_otp()
{
    currentOTP = "";
    otp_deadline_ms = 0;
}

static esp_err_t redirect_to(httpd_req_t *req, const char *location)
{
    httpd_resp_set_status(req, "302 Found");
    httpd_resp_set_hdr(req, "Location", location);
    return httpd_resp_send(req, nullptr, 0);
}

static esp_err_t require_auth(httpd_req_t *req)
{
    if (is_authenticated(req)) return ESP_OK;
    if (is_step1_authenticated(req)) {
        redirect_to(req, "/verify_2fa");
        return ESP_FAIL;
    }
    redirect_to(req, "/login");
    return ESP_FAIL;
}


void append_log(const String &msg)
{
    if (!LittleFS.begin(true)) return;
    String timeStr = current_log_time_string();
    File file = LittleFS.open("/logs.json", "a");
    if (!file) return;
    String logEntry = "{\"time\":\"" + timeStr + "\",\"actor\":\"" + msg + "\"},\n";
    file.print(logEntry);
    file.close();
}

void add_event(uint8_t method, const char *actor)
{
    add_event_internal(method, actor);
}

static void add_event_internal(uint8_t method, const char *actor)
{
    time_t now; time(&now);
    int idx = events_next % MAX_EVENTS;
    events[idx].ts = (uint32_t)now;
    events[idx].method = method;
    strncpy(events[idx].actor, actor ? actor : "", sizeof(events[idx].actor) - 1);
    events[idx].actor[sizeof(events[idx].actor) - 1] = '\0';
    events_next = (events_next + 1) % MAX_EVENTS;
    if (events_count < MAX_EVENTS) events_count++;
    append_log(String(actor ? actor : ""));
}

static void shift_face_metadata_left(int deleted_id, int old_count)
{
    for (int id = deleted_id; id < old_count; ++id) {
        String nextName = prefs.getString(face_name_key(id + 1).c_str(), "");
        String nextTime = prefs.getString(face_time_key(id + 1).c_str(), "");
        String nextRfid = prefs.getString(face_rfid_key(id + 1).c_str(), "");
        String nextRole = prefs.getString(face_role_key(id + 1).c_str(), "");
        int nextFail = prefs.getInt(face_fail_key(id + 1).c_str(), 0);
        bool nextLock = prefs.getBool(face_lock_key(id + 1).c_str(), false);
        if (nextName.length() > 0) prefs.putString(face_name_key(id).c_str(), nextName);
        else prefs.remove(face_name_key(id).c_str());
        if (nextTime.length() > 0) prefs.putString(face_time_key(id).c_str(), nextTime);
        else prefs.remove(face_time_key(id).c_str());
        if (nextRfid.length() > 0) prefs.putString(face_rfid_key(id).c_str(), nextRfid);
        else prefs.remove(face_rfid_key(id).c_str());
        if (nextRole.length() > 0) prefs.putString(face_role_key(id).c_str(), nextRole);
        else prefs.remove(face_role_key(id).c_str());
        if (nextFail > 0) prefs.putInt(face_fail_key(id).c_str(), nextFail);
        else prefs.remove(face_fail_key(id).c_str());
        if (nextLock) prefs.putBool(face_lock_key(id).c_str(), nextLock);
        else prefs.remove(face_lock_key(id).c_str());
    }
    prefs.remove(face_name_key(old_count).c_str());
    prefs.remove(face_time_key(old_count).c_str());
    prefs.remove(face_rfid_key(old_count).c_str());
    prefs.remove(face_role_key(old_count).c_str());
    prefs.remove(face_fail_key(old_count).c_str());
    prefs.remove(face_lock_key(old_count).c_str());
}

static void sync_face_raw_map()
{
    if (!LittleFS.begin(true)) return;
    struct FaceDbMeta { uint16_t num_feats_total; uint16_t num_feats_valid; uint16_t feat_len; } meta = {};
    std::vector<uint16_t> raw_ids;
    File file = LittleFS.open(kFaceDbPathFs, "r");
    if (file) {
        if (file.read(reinterpret_cast<uint8_t *>(&meta), sizeof(meta)) == sizeof(meta) && meta.feat_len > 0) {
            size_t feat_bytes = meta.feat_len * sizeof(float);
            for (uint16_t i = 0; i < meta.num_feats_total; ++i) {
                uint16_t raw_id = 0;
                if (file.read(reinterpret_cast<uint8_t *>(&raw_id), sizeof(raw_id)) != sizeof(raw_id)) break;
                if (raw_id != 0) raw_ids.push_back(raw_id);
                if (!file.seek(file.position() + feat_bytes)) break;
            }
        }
        file.close();
    }
    int old_count = prefs.getUInt("raw_count", 0);
    int new_count = (int)raw_ids.size();
    for (int id = 1; id <= new_count; ++id) prefs.putUInt(face_raw_key(id).c_str(), raw_ids[id - 1]);
    for (int id = new_count + 1; id <= old_count; ++id) prefs.remove(face_raw_key(id).c_str());
    prefs.putUInt("raw_count", new_count);
}

static bool run_liveness_for_face(dl::image::img_t *img, const dl::detect::result_t &face, RecognitionOutcome *outcome)
{
    if (!img || !outcome) return false;
    int box_x = face.box[0];
    int box_y = face.box[1];
    int box_w = face.box[2] - face.box[0];
    int box_h = face.box[3] - face.box[1];
    CropRegion crop = compute_scaled_crop_region(img->width, img->height, box_x, box_y, box_w, box_h, 2.7f);

    float liveness_score = antispoof_run_rgb888_crop(static_cast<const uint8_t *>(img->data),
                                                     img->width,
                                                     img->height,
                                                     crop.left,
                                                     crop.top,
                                                     crop.right,
                                                     crop.bottom);
    outcome->liveness_score = liveness_score;
    outcome->anti_ran = true;
    if (liveness_score < 0.5f) {
        outcome->overlay_status = OVERLAY_FAKE;
        snprintf(outcome->label, sizeof(outcome->label), "fake ti le (%.0f%%)", (1.0f - liveness_score) * 100.0f);
        return false;
    }
    return true;
}

static RecognitionOutcome run_face_recognition(dl::image::img_t *img, std::list<dl::detect::result_t> *results)
{
    int64_t total_start_us = esp_timer_get_time();
    int64_t crop_start_us = total_start_us;
    int64_t crop_us = 0;
    int64_t antispoof_us = 0;
    int64_t enroll_us = 0;
    int64_t recognize_us = 0;
    int64_t recognizer_prepare_us = 0;
    int64_t recognizer_mutex_wait_us = 0;
    RecognitionOutcome outcome = {};
    outcome.id = -1;
    outcome.liveness_score = -1.0f;
    outcome.anti_ran = false;
    outcome.overlay_status = OVERLAY_NONE;
    outcome.label[0] = '\0';
    if (!img || !results || results->empty()) {
        return outcome;
    }

    auto box = results->front().box;
    int box_x = box[0];
    int box_y = box[1];
    int box_w = box[2] - box[0];
    int box_h = box[3] - box[1];
    CropRegion crop = compute_scaled_crop_region(img->width, img->height, box_x, box_y, box_w, box_h, 2.7f);
    int crop_w = crop.right - crop.left + 1;
    int crop_h = crop.bottom - crop.top + 1;

    ESP_LOGI(TAG,
             "AI: Face bbox=(x=%d,y=%d,w=%d,h=%d) crop=(l=%d,t=%d,r=%d,b=%d,w=%d,h=%d) scale=%.2f->%.2f frame=%dx%d",
             box_x,
             box_y,
             box_w,
             box_h,
             crop.left,
             crop.top,
             crop.right,
             crop.bottom,
             crop_w,
             crop_h,
             crop.requested_scale,
             crop.applied_scale,
             (int)img->width,
             (int)img->height);

    static int64_t face_guide_ok_since_us = 0;
    const bool should_apply_auth_guidance = ai_ready_to_process && is_enrolling != 1 && (recognition_enabled || rfid_auth_window_active);
    if (is_enrolling == 1) {
        face_guide_ok_since_us = 0;
        updateFaceGuidanceText("Dang dang ky...", 1500);
    } else if (should_apply_auth_guidance) {
        FaceGuidance guide = evaluate_face_guidance(
            img->width,
            img->height,
            box_x,
            box_y,
            box_w,
            box_h,
            crop);
        int64_t now_us = esp_timer_get_time();

        if (guide.code != FACE_GUIDE_OK) {
            face_guide_ok_since_us = 0;
            outcome.overlay_status = OVERLAY_REAL_UNREGISTERED;
            snprintf(outcome.label, sizeof(outcome.label), "%s", guide.text);
            updateFaceGuidanceText(guide.text);
            ESP_LOGI(TAG,
                     "FACE_GUIDE: result=skip code=%d text=%s bbox=%dx%d applied_scale=%.2f center_dx=%.2f center_dy=%.2f",
                     (int)guide.code,
                     guide.text,
                     box_w,
                     box_h,
                     crop.applied_scale,
                     guide.center_dx_ratio,
                     guide.center_dy_ratio);
            return outcome;
        }

        if (face_guide_ok_since_us == 0) {
            face_guide_ok_since_us = now_us;
        }
        int64_t stable_us = now_us - face_guide_ok_since_us;
        if (stable_us < FACE_GUIDE_STABLE_US) {
            outcome.overlay_status = OVERLAY_REAL_UNREGISTERED;
            snprintf(outcome.label, sizeof(outcome.label), "Giu yen");
            updateFaceGuidanceText("Giu yen");
            ESP_LOGI(TAG,
                     "FACE_GUIDE: result=hold stable=%lldms need=%lldms bbox=%dx%d applied_scale=%.2f",
                     stable_us / 1000,
                     FACE_GUIDE_STABLE_US / 1000,
                     box_w,
                     box_h,
                     crop.applied_scale);
            return outcome;
        }

        face_guide_ok_since_us = 0;
        updateFaceGuidanceText("Dang xu ly...", 1500);
        ESP_LOGI(TAG,
                 "FACE_GUIDE: result=ok stable=%lldms bbox=%dx%d applied_scale=%.2f",
                 stable_us / 1000,
                 box_w,
                 box_h,
                 crop.applied_scale);
    }

    float liveness_score = -1.0f;
    outcome.liveness_score = liveness_score;
    auto run_antispoof_stage = [&](const char *result_name, bool include_recognition_timing) -> bool {
        crop_start_us = esp_timer_get_time();
        uint8_t *fb_data = static_cast<uint8_t *>(img->data);
        crop_us = esp_timer_get_time() - crop_start_us;

        int64_t antispoof_start_us = esp_timer_get_time();
        liveness_score = antispoof_run_rgb888_crop(fb_data,
                                                   img->width,
                                                   img->height,
                                                   crop.left,
                                                   crop.top,
                                                   crop.right,
                                                   crop.bottom);
        antispoof_us = esp_timer_get_time() - antispoof_start_us;
        outcome.liveness_score = liveness_score;
        outcome.anti_ran = true;
        // ai_analyzer_task is event-driven, so it is not registered to TWDT full-time.

        if (liveness_score < 0.5f) {
            ESP_LOGW(TAG, "AI: Spoof detected! Score: %.2f (Threshold: 0.50)", liveness_score);
            outcome.overlay_status = OVERLAY_FAKE;
            snprintf(outcome.label, sizeof(outcome.label), "fake ti le (%.0f%%)", (1.0f - liveness_score) * 100.0f);
            update_overlay_state(results, outcome);
            if (rfid_auth_window_active && rfid_expected_face_id > 0) {
                String name = prefs.getString(face_name_key(rfid_expected_face_id).c_str(), "");
                if (name.length() == 0) name = "Face " + String(rfid_expected_face_id);
                String rfid_uid = prefs.getString(face_rfid_key(rfid_expected_face_id).c_str(), "Unknown");
                append_log("Phat hien anh gia cua " + name + ", ID RFID: " + rfid_uid);
                sendTelegramPhoto("Canh bao anh gia: " + name + ", RFID: " + rfid_uid +
                                  ", real=" + String(liveness_score * 100.0f, 1) +
                                  "%, fake=" + String((1.0f - liveness_score) * 100.0f, 1) + "%");
                recordAuthFailureForFace(rfid_expected_face_id, "anh gia");
                
                rfid_auth_window_active = false;
                rfid_expected_face_id = 0;
                setCameraStatusLed(false);
            }
            if (include_recognition_timing) {
                ESP_LOGI(TAG,
                         "AI_PROF: result=%s anti_ran=%d liveness=%.2f crop=%lldms anti=%lldms ensure=%lldms mutex=%lldms recognize=%lldms total=%lldms",
                         result_name,
                         outcome.anti_ran ? 1 : 0,
                         outcome.liveness_score,
                         crop_us / 1000,
                         antispoof_us / 1000,
                         recognizer_prepare_us / 1000,
                         recognizer_mutex_wait_us / 1000,
                         recognize_us / 1000,
                         (esp_timer_get_time() - total_start_us) / 1000);
            } else {
                ESP_LOGI(TAG,
                         "AI_PROF: result=%s anti_ran=%d liveness=%.2f crop=%lldms anti=%lldms total=%lldms",
                         result_name,
                         outcome.anti_ran ? 1 : 0,
                         outcome.liveness_score,
                         crop_us / 1000,
                         antispoof_us / 1000,
                         (esp_timer_get_time() - total_start_us) / 1000);
            }
            return false;
        }
        return true;
    };

    if (is_enrolling == 1) {
        ESP_LOGI(TAG, "AI: ESP32 camera enroll mode, anti-spoof skipped.");
    }

    if (!is_enrolling && (!rfid_auth_window_active || rfid_expected_face_id <= 0)) {
        outcome.overlay_status = OVERLAY_REAL_UNREGISTERED;
        snprintf(outcome.label, sizeof(outcome.label), "can quet the RFID");
        ESP_LOGI(TAG,
                 "AI_PROF: result=waiting_rfid anti_ran=%d liveness=%.2f crop=%lldms anti=%lldms total=%lldms",
                 outcome.anti_ran ? 1 : 0,
                 outcome.liveness_score,
                 crop_us / 1000,
                 antispoof_us / 1000,
                 (esp_timer_get_time() - total_start_us) / 1000);
        return outcome;
    }

    int64_t recognizer_prepare_start_us = esp_timer_get_time();
    HumanFaceRecognizer *face_recognizer = ensure_recognizer();
    recognizer_prepare_us = esp_timer_get_time() - recognizer_prepare_start_us;
    if (!face_recognizer) {
        return outcome;
    }

    int64_t recognizer_mutex_start_us = esp_timer_get_time();
    if (xSemaphoreTake(recognizer_mutex, pdMS_TO_TICKS(5000)) != pdTRUE) {
        ESP_LOGW(TAG, "AI: Recognizer mutex timeout. Face recognition skipped.");
        outcome.overlay_status = OVERLAY_REAL_UNREGISTERED;
        snprintf(outcome.label, sizeof(outcome.label), "recognizer busy");
        return outcome;
    }
    recognizer_mutex_wait_us = esp_timer_get_time() - recognizer_mutex_start_us;

    if (is_enrolling == 1) {
        int64_t enroll_start_us = esp_timer_get_time();
        esp_err_t enroll_res = face_recognizer->enroll(*img, *results);
        enroll_us = esp_timer_get_time() - enroll_start_us;
        int enrolled_id = (enroll_res == ESP_OK) ? face_recognizer->get_num_feats() : -1;
        xSemaphoreGive(recognizer_mutex);

        if (enroll_res == ESP_OK && enrolled_id > 0) {
            sync_face_raw_map();

            String final_name = pending_enroll_name;
            if (final_name.length() == 0) {
                final_name = "Face " + String(enrolled_id);
            }

            time_t now;
            struct tm timeinfo;
            char time_str[32];
            time(&now);
            localtime_r(&now, &timeinfo);
            strftime(time_str, sizeof(time_str), "%Y-%m-%d %H:%M:%S", &timeinfo);

            prefs.putString(face_name_key(enrolled_id).c_str(), final_name);
            prefs.putString(face_time_key(enrolled_id).c_str(), String(time_str));
            prefs.putString(face_role_key(enrolled_id).c_str(), pending_enroll_role.length() ? pending_enroll_role : "Khach");
            pending_rfid_enroll_face_id = enrolled_id;
            append_log("Face enrolled, waiting RFID link: " + final_name);
            pending_enroll_name = "";
            pending_enroll_role = "";
            is_enrolling = 0;
            outcome.id = enrolled_id;
            outcome.overlay_status = OVERLAY_REAL_SUCCESS;
            snprintf(outcome.label, sizeof(outcome.label), "real ti le (%.0f%%) - quet the RFID", liveness_score * 100.0f);
            ESP_LOGI(TAG,
                     "AI_PROF: result=enrolled anti_ran=%d liveness=%.2f crop=%lldms anti=%lldms ensure=%lldms mutex=%lldms enroll=%lldms total=%lldms",
                     outcome.anti_ran ? 1 : 0,
                     outcome.liveness_score,
                     crop_us / 1000,
                     antispoof_us / 1000,
                     recognizer_prepare_us / 1000,
                     recognizer_mutex_wait_us / 1000,
                     enroll_us / 1000,
                     (esp_timer_get_time() - total_start_us) / 1000);
            return outcome;
        }

        pending_enroll_name = "";
        pending_enroll_role = "";
        is_enrolling = 0;
        outcome.overlay_status = OVERLAY_REAL_UNREGISTERED;
        snprintf(outcome.label, sizeof(outcome.label), "real ti le (%.0f%%) - chua dang ky", liveness_score * 100.0f);
        ESP_LOGI(TAG,
                 "AI_PROF: result=enroll_failed anti_ran=%d liveness=%.2f crop=%lldms anti=%lldms ensure=%lldms mutex=%lldms enroll=%lldms total=%lldms",
                 outcome.anti_ran ? 1 : 0,
                 outcome.liveness_score,
                 crop_us / 1000,
                 antispoof_us / 1000,
                 recognizer_prepare_us / 1000,
                 recognizer_mutex_wait_us / 1000,
                 enroll_us / 1000,
                 (esp_timer_get_time() - total_start_us) / 1000);
        return outcome;
    }

    int64_t recognize_start_us = esp_timer_get_time();
    auto recognized_list = face_recognizer->recognize(*img, *results);
    recognize_us = esp_timer_get_time() - recognize_start_us;
    xSemaphoreGive(recognizer_mutex);

    if (!rfid_auth_window_active || rfid_expected_face_id <= 0) {
        outcome.overlay_status = OVERLAY_REAL_UNREGISTERED;
        snprintf(outcome.label, sizeof(outcome.label), "can quet the RFID");
        ESP_LOGI(TAG,
                 "AI_PROF: result=waiting_rfid_after_recognize anti_ran=%d liveness=%.2f crop=%lldms anti=%lldms ensure=%lldms mutex=%lldms recognize=%lldms total=%lldms",
                 outcome.anti_ran ? 1 : 0,
                 outcome.liveness_score,
                 crop_us / 1000,
                 antispoof_us / 1000,
                 recognizer_prepare_us / 1000,
                 recognizer_mutex_wait_us / 1000,
                 recognize_us / 1000,
                 (esp_timer_get_time() - total_start_us) / 1000);
        return outcome;
    }

    if (!recognized_list.empty()) {
        auto best_face = recognized_list.front();
        String name = prefs.getString(face_name_key(best_face.id).c_str(), "");
        if (name.length() == 0) {
            name = "Face " + String(best_face.id);
        }

        if (best_face.id != rfid_expected_face_id) {
            String expected_name = prefs.getString(face_name_key(rfid_expected_face_id).c_str(), "");
            if (expected_name.length() == 0) {
                expected_name = "Face " + String(rfid_expected_face_id);
            }
            String rfid_uid = prefs.getString(face_rfid_key(rfid_expected_face_id).c_str(), "Unknown");
            DoorMessage fail_msg = {};
            fail_msg.cmd = CMD_PLAY_FAIL;
            fail_msg.method = DM_FACE;
            snprintf(fail_msg.actor, sizeof(fail_msg.actor), "%s", "rfid_face_mismatch");
            xQueueSend(doorQueue, &fail_msg, 0);
            append_log("khuon mat khong khop voi the");
            outcome.overlay_status = OVERLAY_REAL_UNREGISTERED;
            snprintf(outcome.label, sizeof(outcome.label), "khong khop the RFID");
            update_overlay_state(results, outcome);
            sendTelegramPhoto("Canh bao: khuon mat khong khop voi the. The cua " + expected_name +
                              ", RFID: " + rfid_uid + ", nhan dien thanh: " + name +
                              ", real=" + String(liveness_score * 100.0f, 1) +
                              "%, fake=" + String((1.0f - liveness_score) * 100.0f, 1) + "%");
            recordAuthFailureForFace(rfid_expected_face_id, "khuon mat khong khop voi the");
            rfid_auth_window_active = false;
            rfid_expected_face_id = 0;
            setCameraStatusLed(false);
            ESP_LOGI(TAG,
                     "AI_PROF: result=rfid_face_mismatch anti_ran=%d liveness=%.2f crop=%lldms anti=%lldms ensure=%lldms mutex=%lldms recognize=%lldms total=%lldms",
                     outcome.anti_ran ? 1 : 0,
                     outcome.liveness_score,
                     crop_us / 1000,
                     antispoof_us / 1000,
                     recognizer_prepare_us / 1000,
                     recognizer_mutex_wait_us / 1000,
                     recognize_us / 1000,
                     (esp_timer_get_time() - total_start_us) / 1000);
            return outcome;
        }

        if (is_occupant_inside_by_face(best_face.id)) {
            DoorMessage fail_msg = {};
            fail_msg.cmd = CMD_PLAY_FAIL;
            fail_msg.method = DM_FACE;
            snprintf(fail_msg.actor, sizeof(fail_msg.actor), "%s", "already_inside");
            xQueueSend(doorQueue, &fail_msg, 0);
            rfid_auth_window_active = false;
            rfid_expected_face_id = 0;
            setCameraStatusLed(false);
            append_log("Tu choi vao: " + name + " dang o trong phong");
            outcome.overlay_status = OVERLAY_REAL_UNREGISTERED;
            snprintf(outcome.label, sizeof(outcome.label), "dang o trong phong");
            ESP_LOGI(TAG,
                     "AI_PROF: result=already_inside anti_ran=%d liveness=%.2f crop=%lldms anti=%lldms ensure=%lldms mutex=%lldms recognize=%lldms total=%lldms",
                     outcome.anti_ran ? 1 : 0,
                     outcome.liveness_score,
                     crop_us / 1000,
                     antispoof_us / 1000,
                     recognizer_prepare_us / 1000,
                     recognizer_mutex_wait_us / 1000,
                     recognize_us / 1000,
                     (esp_timer_get_time() - total_start_us) / 1000);
            return outcome;
        }

        if (!run_antispoof_stage("fake_after_match", true)) {
            return outcome;
        }

        if (!prepare_pending_entry(best_face.id)) {
            DoorMessage fail_msg = {};
            fail_msg.cmd = CMD_PLAY_FAIL;
            fail_msg.method = DM_FACE;
            snprintf(fail_msg.actor, sizeof(fail_msg.actor), "%s", "entry_prepare_failed");
            xQueueSend(doorQueue, &fail_msg, 0);
            rfid_auth_window_active = false;
            rfid_expected_face_id = 0;
            setCameraStatusLed(false);
            append_log("Tu choi vao: " + name + " khong the tao pending entry");
            outcome.overlay_status = OVERLAY_REAL_UNREGISTERED;
            snprintf(outcome.label, sizeof(outcome.label), "real ti le (%.0f%%) - khong the vao", liveness_score * 100.0f);
            ESP_LOGI(TAG,
                     "AI_PROF: result=entry_prepare_failed anti_ran=%d liveness=%.2f crop=%lldms anti=%lldms ensure=%lldms mutex=%lldms recognize=%lldms total=%lldms",
                     outcome.anti_ran ? 1 : 0,
                     outcome.liveness_score,
                     crop_us / 1000,
                     antispoof_us / 1000,
                     recognizer_prepare_us / 1000,
                     recognizer_mutex_wait_us / 1000,
                     recognize_us / 1000,
                     (esp_timer_get_time() - total_start_us) / 1000);
            return outcome;
        }

        resetAuthFailureForFace(best_face.id);

        DoorMessage message = {};
        message.cmd = CMD_OPEN_DOOR;
        message.method = DM_FACE;
        snprintf(message.actor, sizeof(message.actor), "%s", name.c_str());
        outcome.id = best_face.id;
        outcome.overlay_status = OVERLAY_REAL_SUCCESS;
        snprintf(outcome.label, sizeof(outcome.label), "real ti le (%.0f%%) - thanh cong", liveness_score * 100.0f);
        update_overlay_state(results, outcome);
        String rfid_uid = prefs.getString(face_rfid_key(best_face.id).c_str(), "Unknown");
        telegramActor = "Xac thuc thanh cong: " + name +
                        ", ID=" + String(best_face.id) +
                        ", RFID=" + rfid_uid +
                        ", real=" + String(liveness_score * 100.0f, 1) +
                        "%, fake=" + String((1.0f - liveness_score) * 100.0f, 1) + "%";
        sendTelegramFlag = true;
        xQueueSend(doorQueue, &message, 0);
        ESP_LOGI(TAG,
                 "AI_PROF: result=success anti_ran=%d liveness=%.2f crop=%lldms anti=%lldms ensure=%lldms mutex=%lldms recognize=%lldms total=%lldms",
                 outcome.anti_ran ? 1 : 0,
                 outcome.liveness_score,
                 crop_us / 1000,
                 antispoof_us / 1000,
                 recognizer_prepare_us / 1000,
                 recognizer_mutex_wait_us / 1000,
                 recognize_us / 1000,
                 (esp_timer_get_time() - total_start_us) / 1000);
        return outcome;
    }

    outcome.overlay_status = OVERLAY_REAL_UNREGISTERED;
    snprintf(outcome.label, sizeof(outcome.label), "khong khop khuon mat");
    if (rfid_auth_window_active && rfid_expected_face_id > 0) {
        String expected_name = prefs.getString(face_name_key(rfid_expected_face_id).c_str(), "");
        if (expected_name.length() == 0) {
            expected_name = "Face " + String(rfid_expected_face_id);
        }
        String rfid_uid = prefs.getString(face_rfid_key(rfid_expected_face_id).c_str(), "Unknown");
        DoorMessage fail_msg = {};
        fail_msg.cmd = CMD_PLAY_FAIL;
        fail_msg.method = DM_FACE;
        snprintf(fail_msg.actor, sizeof(fail_msg.actor), "%s", "unknown_face");
        xQueueSend(doorQueue, &fail_msg, 0);
        append_log("khuon mat khong phai cua nguoi dung");
        update_overlay_state(results, outcome);
        sendTelegramPhoto("Canh bao: khuon mat khong phai cua nguoi dung. The dang xac thuc: " +
                          expected_name + ", RFID: " + rfid_uid +
                          ", real=" + String(liveness_score * 100.0f, 1) +
                          "%, fake=" + String((1.0f - liveness_score) * 100.0f, 1) + "%");
        recordAuthFailureForFace(rfid_expected_face_id, "khuon mat khong phai cua nguoi dung");
        rfid_auth_window_active = false;
        rfid_expected_face_id = 0;
        setCameraStatusLed(false);
    }
    ESP_LOGI(TAG,
             "AI_PROF: result=no_match anti_ran=%d liveness=%.2f crop=%lldms anti=%lldms ensure=%lldms mutex=%lldms recognize=%lldms total=%lldms",
             outcome.anti_ran ? 1 : 0,
             outcome.liveness_score,
             crop_us / 1000,
             antispoof_us / 1000,
             recognizer_prepare_us / 1000,
             recognizer_mutex_wait_us / 1000,
             recognize_us / 1000,
             (esp_timer_get_time() - total_start_us) / 1000);
    return outcome;
}
// ==================== HANDLERS ====================
static esp_err_t events_handler(httpd_req_t *req)
{
    if (require_auth(req) != ESP_OK) {
        return ESP_OK;
    }

    httpd_resp_set_type(req, "application/json");
    httpd_resp_set_hdr(req, "Access-Control-Allow-Origin", "*");

    String json = "[";
    for (int i = 0; i < events_count; ++i) {
        int idx = events_next - events_count + i;
        while (idx < 0) idx += MAX_EVENTS;
        idx %= MAX_EVENTS;
        if (i > 0) json += ",";
        json += "{\"method\":" + String(events[idx].method) + ",\"actor\":\"" + String(events[idx].actor) + "\",\"ts\":" + String(events[idx].ts) + "}";
    }
    json += "]";
    return httpd_resp_send(req, json.c_str(), HTTPD_RESP_USE_STRLEN);
}

static esp_err_t dht_handler(httpd_req_t *req)
{
    if (require_auth(req) != ESP_OK) {
        return ESP_OK;
    }

    httpd_resp_set_type(req, "application/json");
    httpd_resp_set_hdr(req, "Access-Control-Allow-Origin", "*");

    char buffer[64];
    if (isnan(dht_temp) || isnan(dht_hum)) {
        snprintf(buffer, sizeof(buffer), "{\"temp\":null,\"hum\":null}");
    } else {
        snprintf(buffer, sizeof(buffer), "{\"temp\":%.1f,\"hum\":%.1f}", dht_temp, dht_hum);
    }
    return httpd_resp_send(req, buffer, HTTPD_RESP_USE_STRLEN);
}


static esp_err_t sensors_handler(httpd_req_t *req)
{
    if (require_auth(req) != ESP_OK) return ESP_OK;
    const int64_t start_us = esp_timer_get_time();
    httpd_resp_set_type(req, "application/json");
    httpd_resp_set_hdr(req, "Access-Control-Allow-Origin", "*");
    char buffer[384];
    if (isnan(dht_temp) || isnan(dht_hum)) {
        snprintf(buffer, sizeof(buffer),
          "{\"mq2\":%d,\"smoke\":%s,\"presence\":%s,\"distance\":%d,"
          "\"fan\":%s,\"fan_auto\":%s,\"fan_threshold\":%.1f,\"occupants\":%d,"
          "\"pump\":%s,\"pump_auto\":%s,\"smoke_threshold\":%.1f,"
          "\"temp\":null,\"hum\":null}",
          (int)mq2_value,
          smoke_detected ? "true" : "false",
          presence_detected ? "true" : "false",
          (int)presence_distance,
          fan_state ? "true" : "false",
          fan_auto_mode ? "true" : "false",
          (float)fan_humidity_threshold,
          (int)occupant_count,
          pump_state ? "true" : "false",
          pump_auto_mode ? "true" : "false",
          (float)smoke_alarm_threshold);
    } else {
        snprintf(buffer, sizeof(buffer),
          "{\"mq2\":%d,\"smoke\":%s,\"presence\":%s,\"distance\":%d,"
          "\"fan\":%s,\"fan_auto\":%s,\"fan_threshold\":%.1f,\"occupants\":%d,"
          "\"pump\":%s,\"pump_auto\":%s,\"smoke_threshold\":%.1f,"
          "\"temp\":%.1f,\"hum\":%.1f}",
          (int)mq2_value,
          smoke_detected ? "true" : "false",
          presence_detected ? "true" : "false",
          (int)presence_distance,
          fan_state ? "true" : "false",
          fan_auto_mode ? "true" : "false",
          (float)fan_humidity_threshold,
          (int)occupant_count,
          pump_state ? "true" : "false",
          pump_auto_mode ? "true" : "false",
          (float)smoke_alarm_threshold,
          (float)dht_temp,
          (float)dht_hum);
    }
    esp_err_t ret = httpd_resp_send(req, buffer, HTTPD_RESP_USE_STRLEN);
    const int64_t elapsed_ms = (esp_timer_get_time() - start_us) / 1000;
    if (elapsed_ms > 100) {
        ESP_LOGW(TAG, "HTTP_PROF: /sensors slow=%lldms", elapsed_ms);
    }
    return ret;
}

static esp_err_t ld2410_debug_handler(httpd_req_t *req)
{
    if (require_auth(req) != ESP_OK) return ESP_OK;
    const int64_t start_us = esp_timer_get_time();
    String json = getLd2410DebugJson();
    httpd_resp_set_type(req, "application/json");
    httpd_resp_set_hdr(req, "Access-Control-Allow-Origin", "*");
    esp_err_t ret = httpd_resp_send(req, json.c_str(), HTTPD_RESP_USE_STRLEN);
    const int64_t elapsed_ms = (esp_timer_get_time() - start_us) / 1000;
    if (elapsed_ms > 100) {
        ESP_LOGW(TAG, "HTTP_PROF: /ld2410_debug slow=%lldms json_len=%u", elapsed_ms, (unsigned)json.length());
    }
    return ret;
}

static esp_err_t ld2410_refresh_handler(httpd_req_t *req)
{
    if (require_auth(req) != ESP_OK) return ESP_OK;
    bool eng_ok = enableLd2410EngineeringMode();
    bool cfg_ok = refreshLd2410Configuration();
    char response[64];
    snprintf(response, sizeof(response), "{\"ok\":%s,\"engineering\":%s}",
             cfg_ok ? "true" : "false",
             eng_ok ? "true" : "false");
    httpd_resp_set_type(req, "application/json");
    httpd_resp_set_hdr(req, "Access-Control-Allow-Origin", "*");
    return httpd_resp_send(req, response, HTTPD_RESP_USE_STRLEN);
}

static esp_err_t ld2410_set_sensitivity_handler(httpd_req_t *req)
{
    if (require_auth(req) != ESP_OK) return ESP_OK;
    char *buf = nullptr;
    if (parse_get(req, &buf) != ESP_OK) return httpd_resp_send_500(req);

    char gate_buf[8] = "";
    char moving_buf[8] = "";
    char stationary_buf[8] = "";
    int gate = -1;
    int moving = -1;
    int stationary = -1;

    if (httpd_query_key_value(buf, "gate", gate_buf, sizeof(gate_buf)) == ESP_OK) gate = atoi(gate_buf);
    if (httpd_query_key_value(buf, "moving", moving_buf, sizeof(moving_buf)) == ESP_OK) moving = atoi(moving_buf);
    if (httpd_query_key_value(buf, "stationary", stationary_buf, sizeof(stationary_buf)) == ESP_OK) stationary = atoi(stationary_buf);
    free(buf);

    bool ok = false;
    if (gate >= 0 && gate <= 8 && moving >= 0 && moving <= 100 && stationary >= 0 && stationary <= 100) {
        ok = setLd2410GateSensitivity((uint8_t)gate, (uint8_t)moving, (uint8_t)stationary);
    }

    char response[96];
    snprintf(response, sizeof(response), "{\"ok\":%s,\"gate\":%d}", ok ? "true" : "false", gate);
    httpd_resp_set_type(req, "application/json");
    httpd_resp_set_hdr(req, "Access-Control-Allow-Origin", "*");
    return httpd_resp_send(req, response, HTTPD_RESP_USE_STRLEN);
}

static esp_err_t occupants_handler(httpd_req_t *req)
{
    if (require_auth(req) != ESP_OK) return ESP_OK;
    String json = get_occupants_json();
    httpd_resp_set_type(req, "application/json");
    httpd_resp_set_hdr(req, "Access-Control-Allow-Origin", "*");
    return httpd_resp_send(req, json.c_str(), HTTPD_RESP_USE_STRLEN);
}

static esp_err_t sync_time_handler(httpd_req_t *req)
{
    if (require_auth(req) != ESP_OK) return ESP_OK;
    char *buf = nullptr;
    if (parse_get(req, &buf) != ESP_OK) return httpd_resp_send_500(req);
    char ts_buf[24] = "";
    uint32_t ts = 0;
    if (httpd_query_key_value(buf, "ts", ts_buf, sizeof(ts_buf)) == ESP_OK) {
        ts = static_cast<uint32_t>(strtoul(ts_buf, nullptr, 10));
    }
    free(buf);

    if (!is_valid_unix_time(ts)) {
        httpd_resp_set_type(req, "application/json");
        return httpd_resp_send(req, "{\"ok\":false,\"error\":\"invalid_ts\"}", HTTPD_RESP_USE_STRLEN);
    }

    set_system_unix_time(ts);
    char response[80];
    snprintf(response, sizeof(response), "{\"ok\":true,\"now\":%lu}", static_cast<unsigned long>(ts));
    httpd_resp_set_type(req, "application/json");
    httpd_resp_set_hdr(req, "Access-Control-Allow-Origin", "*");
    return httpd_resp_send(req, response, HTTPD_RESP_USE_STRLEN);
}

static esp_err_t time_status_handler(httpd_req_t *req)
{
    if (require_auth(req) != ESP_OK) return ESP_OK;
    time_t now;
    time(&now);
    bool synced = is_valid_unix_time(now);
    char response[96];
    snprintf(response, sizeof(response),
             "{\"synced\":%s,\"now\":%lu,\"last\":%lu}",
             synced ? "true" : "false",
             synced ? static_cast<unsigned long>(now) : 0UL,
             static_cast<unsigned long>(prefs.getUInt("last_unix", 0)));
    httpd_resp_set_type(req, "application/json");
    httpd_resp_set_hdr(req, "Access-Control-Allow-Origin", "*");
    return httpd_resp_send(req, response, HTTPD_RESP_USE_STRLEN);
}

static esp_err_t fan_control_handler(httpd_req_t *req)
{
    if (require_auth(req) != ESP_OK) return ESP_OK;
    char *buf = nullptr;
    if (parse_get(req, &buf) != ESP_OK) return httpd_resp_send_500(req);
    char action[16] = "";
    char value[16] = "";
    if (httpd_query_key_value(buf, "action", action, sizeof(action)) == ESP_OK) {
        if (strcmp(action, "toggle") == 0) {
            fan_state = !fan_state;
            fan_auto_mode = false;
            if (mcp_found) {
                if (xSemaphoreTake(i2cMutex, pdMS_TO_TICKS(100)) == pdTRUE) {
                    mcp.digitalWrite(MCP_FAN_PIN, fan_state ? HIGH : LOW);
                    xSemaphoreGive(i2cMutex);
                }
            }
        } else if (strcmp(action, "auto") == 0) {
            fan_auto_mode = !fan_auto_mode;
        } else if (strcmp(action, "threshold") == 0) {
            if (httpd_query_key_value(buf, "val", value, sizeof(value)) == ESP_OK) {
                fan_humidity_threshold = atof(value);
            }
        }
    }
    free(buf);
    char response[128];
    snprintf(response, sizeof(response),
      "{\"fan\":%s,\"auto\":%s,\"threshold\":%.1f}",
      fan_state ? "true" : "false",
      fan_auto_mode ? "true" : "false",
      (float)fan_humidity_threshold);
    httpd_resp_set_type(req, "application/json");
    httpd_resp_set_hdr(req, "Access-Control-Allow-Origin", "*");
    return httpd_resp_send(req, response, HTTPD_RESP_USE_STRLEN);
}

static esp_err_t pump_control_handler(httpd_req_t *req)
{
    if (require_auth(req) != ESP_OK) return ESP_OK;
    char *buf = nullptr;
    if (parse_get(req, &buf) != ESP_OK) return httpd_resp_send_500(req);
    char action[16] = "";
    char value[16] = "";
    if (httpd_query_key_value(buf, "action", action, sizeof(action)) == ESP_OK) {
        if (strcmp(action, "toggle") == 0) {
            pump_state = !pump_state;
            pump_auto_mode = false;
            if (mcp_found && xSemaphoreTake(i2cMutex, pdMS_TO_TICKS(100)) == pdTRUE) {
                mcp.digitalWrite(MCP_PUMP_PIN, pump_state ? HIGH : LOW);
                xSemaphoreGive(i2cMutex);
            }
        } else if (strcmp(action, "auto") == 0) {
            pump_auto_mode = !pump_auto_mode;
            if (!pump_auto_mode) {
                pump_state = false;
                if (mcp_found && xSemaphoreTake(i2cMutex, pdMS_TO_TICKS(100)) == pdTRUE) {
                    mcp.digitalWrite(MCP_PUMP_PIN, LOW);
                    xSemaphoreGive(i2cMutex);
                }
            }
        } else if (strcmp(action, "threshold") == 0) {
            if (httpd_query_key_value(buf, "val", value, sizeof(value)) == ESP_OK) {
                float next = atof(value);
                if (next < 10.0f) next = 10.0f;
                if (next > 100.0f) next = 100.0f;
                smoke_alarm_threshold = next;
            }
        }
    }
    free(buf);
    char response[128];
    snprintf(response, sizeof(response),
      "{\"pump\":%s,\"auto\":%s,\"threshold\":%.1f}",
      pump_state ? "true" : "false",
      pump_auto_mode ? "true" : "false",
      (float)smoke_alarm_threshold);
    httpd_resp_set_type(req, "application/json");
    httpd_resp_set_hdr(req, "Access-Control-Allow-Origin", "*");
    return httpd_resp_send(req, response, HTTPD_RESP_USE_STRLEN);
}

static esp_err_t capture_handler(httpd_req_t *req)
{
    if (require_auth(req) != ESP_OK) {
        return ESP_OK;
    }

    ESP_LOGI(TAG, "CAPTURE: request received, free_heap=%u, free_psram=%u",
             (unsigned)ESP.getFreeHeap(),
             (unsigned)ESP.getFreePsram());
    camera_fb_t *fb = esp_camera_fb_get();
    if (!fb) {
        ESP_LOGW(TAG, "CAPTURE: esp_camera_fb_get() failed, free_heap=%u, free_psram=%u",
                 (unsigned)ESP.getFreeHeap(),
                 (unsigned)ESP.getFreePsram());
        return httpd_resp_send_500(req);
    }

    uint8_t *jpg_buf = nullptr;
    size_t jpg_len = 0;
    bool needs_free = false;
    if (fb->format == PIXFORMAT_JPEG) {
        jpg_buf = fb->buf;
        jpg_len = fb->len;
    } else if (frame2jpg(fb, 80, &jpg_buf, &jpg_len)) {
        needs_free = true;
    } else {
        esp_camera_fb_return(fb);
        return httpd_resp_send_500(req);
    }

    httpd_resp_set_type(req, "image/jpeg");
    httpd_resp_set_hdr(req, "Content-Disposition", "inline; filename=capture.jpg");
    httpd_resp_set_hdr(req, "Access-Control-Allow-Origin", "*");
    esp_err_t res = httpd_resp_send(req, (const char *)jpg_buf, jpg_len);
    if (needs_free && jpg_buf) free(jpg_buf);
    esp_camera_fb_return(fb);
    return res;
}

static esp_err_t stream_handler(httpd_req_t *req)
{
    ensure_ai_background();
    bool clean_stream = false;
    char *query = nullptr;
    if (parse_get(req, &query) == ESP_OK) {
        char clean[8] = "";
        clean_stream = httpd_query_key_value(query, "clean", clean, sizeof(clean)) == ESP_OK && atoi(clean) == 1;
        free(query);
    }
    ESP_LOGI(TAG, "STREAM: client connected, free_heap=%u, free_psram=%u",
             (unsigned)ESP.getFreeHeap(),
             (unsigned)ESP.getFreePsram());

    esp_err_t res = httpd_resp_set_type(req, _STREAM_CONTENT_TYPE);
    if (res != ESP_OK) {
        return res;
    }
    httpd_resp_set_hdr(req, "Access-Control-Allow-Origin", "*");
    setCameraStatusLed(true);

    while (true) {
        camera_fb_t *fb = esp_camera_fb_get();
        if (!fb) {
            ESP_LOGW(TAG, "STREAM: esp_camera_fb_get() failed, free_heap=%u, free_psram=%u",
                     (unsigned)ESP.getFreeHeap(),
                     (unsigned)ESP.getFreePsram());
            res = ESP_FAIL;
            break;
        }

        OverlayState overlay = {};
        bool has_overlay = !clean_stream && (is_enrolling != 1) && snapshot_overlay_state(&overlay);
        uint8_t *jpg_buf = nullptr;
        size_t jpg_len = 0;
        bool needs_free = false;
        if (has_overlay) {
            size_t rgb_len = fb->width * fb->height * 3;
            uint8_t *rgb_buf = static_cast<uint8_t *>(heap_caps_malloc(rgb_len, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT));
            if (!rgb_buf) {
                ESP_LOGW(TAG, "STREAM: Failed to allocate overlay RGB buffer, sending raw JPEG.");
                has_overlay = false;
            } else {
                bool converted = fmt2rgb888(fb->buf, fb->len, fb->format, rgb_buf);
                if (!converted) {
                    ESP_LOGW(TAG, "STREAM: Failed to convert frame for overlay (format=%d).", (int)fb->format);
                    free(rgb_buf);
                    has_overlay = false;
                } else {
                    fb_data_t rfb;
                    rfb.width = fb->width;
                    rfb.height = fb->height;
                    rfb.bytes_per_pixel = 3;
                    rfb.format = FB_RGB888;
                    rfb.data = rgb_buf;
                    draw_overlay(&rfb, overlay);
                    if (fmt2jpg(rgb_buf, rgb_len, fb->width, fb->height, PIXFORMAT_RGB888, 90, &jpg_buf, &jpg_len)) {
                        needs_free = true;
                    } else {
                        ESP_LOGW(TAG, "STREAM: Failed to encode overlay JPEG.");
                    }
                    free(rgb_buf);
                }
            }
        }

        if (!jpg_buf && fb->format == PIXFORMAT_JPEG) {
            jpg_buf = fb->buf;
            jpg_len = fb->len;
        } else if (!jpg_buf && frame2jpg(fb, 80, &jpg_buf, &jpg_len)) {
            needs_free = true;
        } else if (!jpg_buf) {
            esp_camera_fb_return(fb);
            res = ESP_FAIL;
            break;
        }

        int64_t now = esp_timer_get_time();
        static int64_t last_ai_push_ms = 0;
        int64_t now_ms = now / 1000;
        int cadence_ms = current_ai_cadence_ms();
        if (!clean_stream &&
            (detection_enabled || recognition_enabled) &&
            ai_frame_buffer && ai_frame_mutex && !ai_ready_to_process &&
            (now_ms - last_ai_push_ms >= cadence_ms)) {
                if (xSemaphoreTake(ai_frame_mutex, 0) != pdTRUE) {
                    goto skip_ai_copy;
                }
                int64_t convert_start_us = esp_timer_get_time();
                const char *decode_path = "fmt2rgb888";
                bool converted = decode_to_rgb888(fb->buf, fb->len, fb->format, ai_frame_buffer);
                if (!converted && fb->format == PIXFORMAT_JPEG) {
                    decode_path = "jpg2rgb565_fallback";
                    // Giải mã JPEG thành RGB565 (2 byte/pixel)
                    if (jpg2rgb565(fb->buf, fb->len, ai_frame_buffer, JPG_SCALE_NONE)) {
                        // Chuyển đổi từ RGB565 lên RGB888 (Expansion Loop - Chạy ngược để tránh ghi đè)
                        int pixel_count = fb->width * fb->height;
                        uint16_t *src = (uint16_t *)ai_frame_buffer;
                        uint8_t *dst = ai_frame_buffer;
                        
                        for (int i = pixel_count - 1; i >= 0; i--) {
                            uint16_t p = src[i];
                            // Tách các thành phần màu 5-6-5
                            uint8_t r = (p >> 11) & 0x1F;
                            uint8_t g = (p >> 5) & 0x3F;
                            uint8_t b = p & 0x1F;
                            
                            // Gán lại theo hệ RGB888 (8-8-8)
                            dst[i * 3 + 0] = (r * 255) / 31; // R
                            dst[i * 3 + 1] = (g * 255) / 63; // G
                            dst[i * 3 + 2] = (b * 255) / 31; // B
                        }
                        converted = true;
                    }
                }

                if (converted) {
                    ai_frame_w = fb->width;
                    ai_frame_h = fb->height;
                    ai_ready_to_process = true;
                    last_ai_push_ms = now_ms;
                    notify_ai_task();
                    ESP_LOGI(TAG,
                             "AI_PROF: stream_ai_copy frame=%dx%d convert=%lldms decode=%s cadence=%dms",
                             fb->width,
                             fb->height,
                             (esp_timer_get_time() - convert_start_us) / 1000,
                             decode_path,
                             cadence_ms);
                } else {
                    ESP_LOGW(TAG, "AI: Frame conversion failed (format=%d, len=%u)", (int)fb->format, (unsigned)fb->len);
                }
                xSemaphoreGive(ai_frame_mutex);
        }
skip_ai_copy:

        char part_buf[128];
        size_t part_len = snprintf(part_buf,
                                   sizeof(part_buf),
                                   _STREAM_PART,
                                   (unsigned)jpg_len,
                                   (long)(now / 1000000),
                                   (long)(now % 1000000));

        res = httpd_resp_send_chunk(req, _STREAM_BOUNDARY, strlen(_STREAM_BOUNDARY));
        if (res == ESP_OK) res = httpd_resp_send_chunk(req, part_buf, part_len);
        if (res == ESP_OK) res = httpd_resp_send_chunk(req, (const char *)jpg_buf, jpg_len);

        if (needs_free && jpg_buf) free(jpg_buf);
        esp_camera_fb_return(fb);
        if (res != ESP_OK) break;
    }

    setCameraStatusLed(false);
    httpd_resp_send_chunk(req, nullptr, 0);
    return res == ESP_OK ? ESP_OK : res;
}
// ==================== AI BACKGROUND TASK ====================
static void ai_analyzer_task(void *pvParameters)
{
    ESP_LOGI(TAG, "AI analyzer task started");
    int64_t last_self_capture_ms = 0;

    while (true) {
        if (!ai_ready_to_process) {
            uint32_t wait_ms = (rfid_auth_window_active || is_enrolling) ? 50 : 1000;
            ulTaskNotifyTake(pdTRUE, pdMS_TO_TICKS(wait_ms));
        }

        // Self-capture: khi RFID auth window hoặc enrolling mà stream không đang chạy
        if (!ai_ready_to_process && ai_frame_buffer && detector &&
            (rfid_auth_window_active || is_enrolling)) {
            int64_t now_ms = esp_timer_get_time() / 1000;
            int cadence_ms = current_ai_cadence_ms();
            bool auth_warmup_active = false;
            if (rfid_auth_window_active && ai_auth_warmup_until_ms != 0) {
                uint32_t now32 = millis();
                if ((int32_t)(ai_auth_warmup_until_ms - now32) > 0) {
                    auth_warmup_active = true;
                } else {
                    ai_auth_warmup_until_ms = 0;
                    ESP_LOGI(TAG,
                             "AI_PROF: auth_warmup_done flush_remaining=%u",
                             (unsigned)ai_auth_flush_frames_remaining);
                }
            }

            if (auth_warmup_active) {
                last_self_capture_ms = now_ms;
                vTaskDelay(pdMS_TO_TICKS(20));
            } else if (now_ms - last_self_capture_ms >= cadence_ms) {
                camera_fb_t *fb = esp_camera_fb_get();
                if (fb) {
                    if (rfid_auth_window_active && ai_auth_flush_frames_remaining > 0) {
                        uint8_t flush_remaining = ai_auth_flush_frames_remaining - 1;
                        ai_auth_flush_frames_remaining = flush_remaining;
                        last_self_capture_ms = now_ms;
                        ESP_LOGI(TAG,
                                 "AI_PROF: auth_flush_frame remaining=%u frame=%dx%d len=%u",
                                 (unsigned)flush_remaining,
                                 fb->width,
                                 fb->height,
                                 (unsigned)fb->len);
                        esp_camera_fb_return(fb);
                        continue;
                    }

                    if (xSemaphoreTake(ai_frame_mutex, pdMS_TO_TICKS(100)) == pdTRUE) {
                        int64_t convert_start_us = esp_timer_get_time();
                        const char *decode_path = "fmt2rgb888";
                        bool converted = decode_to_rgb888(fb->buf, fb->len, fb->format, ai_frame_buffer);
                        if (!converted && fb->format == PIXFORMAT_JPEG) {
                            decode_path = "jpg2rgb565_fallback";
                            if (jpg2rgb565(fb->buf, fb->len, ai_frame_buffer, JPG_SCALE_NONE)) {
                                int pixel_count = fb->width * fb->height;
                                uint16_t *src = (uint16_t *)ai_frame_buffer;
                                uint8_t *dst = ai_frame_buffer;
                                for (int i = pixel_count - 1; i >= 0; i--) {
                                    uint16_t p = src[i];
                                    uint8_t r = (p >> 11) & 0x1F;
                                    uint8_t g = (p >> 5) & 0x3F;
                                    uint8_t b = p & 0x1F;
                                    dst[i * 3 + 0] = (r * 255) / 31;
                                    dst[i * 3 + 1] = (g * 255) / 63;
                                    dst[i * 3 + 2] = (b * 255) / 31;
                                }
                                converted = true;
                            }
                        }
                        if (converted) {
                            ai_frame_w = fb->width;
                            ai_frame_h = fb->height;
                            ai_ready_to_process = true;
                            last_self_capture_ms = now_ms;
                            ESP_LOGI(TAG,
                                     "AI_PROF: self_capture frame=%dx%d convert=%lldms decode=%s cadence=%dms",
                                     fb->width,
                                     fb->height,
                                     (esp_timer_get_time() - convert_start_us) / 1000,
                                     decode_path,
                                     cadence_ms);
                        }
                        xSemaphoreGive(ai_frame_mutex);
                    }
                    esp_camera_fb_return(fb);
                }
            }
        }

        if (ai_ready_to_process && ai_frame_buffer && detector) {
            if (xSemaphoreTake(ai_frame_mutex, portMAX_DELAY) == pdTRUE) {
                int64_t frame_start_us = esp_timer_get_time();
                int64_t detect_start_us = frame_start_us;
                dl::image::img_t img = {ai_frame_buffer, (uint16_t)ai_frame_w, (uint16_t)ai_frame_h, dl::image::DL_IMAGE_PIX_TYPE_RGB888};
                auto results = detector->run(img); 
                int64_t detect_us = esp_timer_get_time() - detect_start_us;
                if (results.empty()) {
                    clear_overlay_state();
                    ESP_LOGI(TAG,
                             "AI_PROF: result=no_face frame=%dx%d detect=%lldms total=%lldms",
                             ai_frame_w,
                             ai_frame_h,
                             detect_us / 1000,
                             (esp_timer_get_time() - frame_start_us) / 1000);
                } else {
                    notifyRfidFaceSeen();
                }
                if (!results.empty() && (recognition_enabled || is_enrolling)) {
                    RecognitionOutcome outcome = run_face_recognition(&img, &results);
                    update_overlay_state(&results, outcome);
                    if (outcome.overlay_status == OVERLAY_FAKE) {
                        DoorMessage fail_msg = {};
                        fail_msg.cmd = CMD_PLAY_FAIL;
                        fail_msg.method = DM_FACE;
                        snprintf(fail_msg.actor, sizeof(fail_msg.actor), "%s", "spoof");
                        xQueueSend(doorQueue, &fail_msg, 0);
                    }
                } else if (!results.empty()) {
                    clear_overlay_state();
                }
                if (!results.empty()) {
                    ESP_LOGI(TAG,
                             "AI_PROF: frame_done faces=%d frame=%dx%d detect=%lldms total=%lldms",
                             (int)results.size(),
                             ai_frame_w,
                             ai_frame_h,
                             detect_us / 1000,
                             (esp_timer_get_time() - frame_start_us) / 1000);
                }
                ai_ready_to_process = false;
                xSemaphoreGive(ai_frame_mutex);
            }
        }
    }
}

static esp_err_t enroll_face_handler(httpd_req_t *req)
{
    if (require_auth(req) != ESP_OK) {
        return ESP_OK;
    }

    char *buf = nullptr;
    pending_enroll_name = "";
    pending_enroll_role = "";
    if (parse_get(req, &buf) == ESP_OK) {
        char name[64] = "";
        char role[32] = "";
        if (httpd_query_key_value(buf, "name", name, sizeof(name)) == ESP_OK) pending_enroll_name = urlDecode(String(name));
        if (httpd_query_key_value(buf, "role", role, sizeof(role)) == ESP_OK) pending_enroll_role = urlDecode(String(role));
        free(buf);
    }
    clear_overlay_state();
    is_enrolling = 1;
    ensure_ai_background();
    notify_ai_task();
    return httpd_resp_sendstr(req, "OK");
}

// ==================== PHONE CAMERA ENROLL (POST) ====================
static esp_err_t enroll_phone_handler(httpd_req_t *req)
{
    if (require_auth(req) != ESP_OK) {
        return ESP_OK;
    }

    // 1. Parse query parameter "name"
    char *qbuf = nullptr;
    char name_str[64] = "";
    if (parse_get(req, &qbuf) == ESP_OK) {
        httpd_query_key_value(qbuf, "name", name_str, sizeof(name_str));
        free(qbuf);
    }

    // 2. Read POST body (JPEG data)
    size_t content_len = req->content_len;
    ESP_LOGI(TAG, "ENROLL_PHONE: Receiving JPEG, content_len=%u", (unsigned)content_len);

    if (content_len == 0 || content_len > 100 * 1024) {
        httpd_resp_set_type(req, "application/json");
        httpd_resp_set_hdr(req, "Access-Control-Allow-Origin", "*");
        return httpd_resp_send(req, "{\"error\":\"Invalid image size (max 100KB)\"}", HTTPD_RESP_USE_STRLEN);
    }

    uint8_t *jpg_buf = (uint8_t *)heap_caps_malloc(content_len, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    if (!jpg_buf) {
        ESP_LOGE(TAG, "ENROLL_PHONE: Failed to allocate JPEG buffer (%u bytes)", (unsigned)content_len);
        httpd_resp_set_type(req, "application/json");
        httpd_resp_set_hdr(req, "Access-Control-Allow-Origin", "*");
        return httpd_resp_send(req, "{\"error\":\"Out of memory\"}", HTTPD_RESP_USE_STRLEN);
    }

    // Read body in chunks
    size_t received = 0;
    while (received < content_len) {
        int ret = httpd_req_recv(req, (char *)(jpg_buf + received), content_len - received);
        if (ret <= 0) {
            if (ret == HTTPD_SOCK_ERR_TIMEOUT) continue;
            ESP_LOGE(TAG, "ENROLL_PHONE: httpd_req_recv failed (ret=%d)", ret);
            free(jpg_buf);
            httpd_resp_set_type(req, "application/json");
            httpd_resp_set_hdr(req, "Access-Control-Allow-Origin", "*");
            return httpd_resp_send(req, "{\"error\":\"Receive error\"}", HTTPD_RESP_USE_STRLEN);
        }
        received += ret;
    }
    ESP_LOGI(TAG, "ENROLL_PHONE: Received %u bytes JPEG", (unsigned)received);

    // 3. Decode JPEG to RGB888
    // Allocate RGB buffer for 320x240 image (worst case)
    size_t rgb_buf_size = 320 * 240 * 3;
    uint8_t *rgb_buf = (uint8_t *)heap_caps_malloc(rgb_buf_size, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    if (!rgb_buf) {
        ESP_LOGE(TAG, "ENROLL_PHONE: Failed to allocate RGB buffer");
        free(jpg_buf);
        httpd_resp_set_type(req, "application/json");
        httpd_resp_set_hdr(req, "Access-Control-Allow-Origin", "*");
        return httpd_resp_send(req, "{\"error\":\"Out of memory for RGB\"}", HTTPD_RESP_USE_STRLEN);
    }

    // Decode JPEG directly to RGB888 to avoid the old RGB565 expansion pass.
    bool decoded = decode_to_rgb888(jpg_buf, received, PIXFORMAT_JPEG, rgb_buf);
    if (!decoded && jpg2rgb565(jpg_buf, received, rgb_buf, JPG_SCALE_NONE)) {
        // We need to know the actual image dimensions from the JPEG header
        // For 320x240 source, this should work
        int pixel_count = 320 * 240;
        uint16_t *src = (uint16_t *)rgb_buf;
        uint8_t *dst = rgb_buf;
        for (int i = pixel_count - 1; i >= 0; i--) {
            uint16_t p = src[i];
            uint8_t r = (p >> 11) & 0x1F;
            uint8_t g = (p >> 5) & 0x3F;
            uint8_t b = p & 0x1F;
            dst[i * 3 + 0] = (r * 255) / 31;
            dst[i * 3 + 1] = (g * 255) / 63;
            dst[i * 3 + 2] = (b * 255) / 31;
        }
        decoded = true;
    }
    free(jpg_buf);

    if (!decoded) {
        ESP_LOGE(TAG, "ENROLL_PHONE: JPEG decode failed");
        free(rgb_buf);
        httpd_resp_set_type(req, "application/json");
        httpd_resp_set_hdr(req, "Access-Control-Allow-Origin", "*");
        return httpd_resp_send(req, "{\"error\":\"JPEG decode failed\"}", HTTPD_RESP_USE_STRLEN);
    }

    // 4. Run AI pipeline: Face Detect → Anti-spoof → Enroll
    ESP_LOGI(TAG, "ENROLL_PHONE: Running face detection on 320x240 image...");
    dl::image::img_t img = {rgb_buf, 320, 240, dl::image::DL_IMAGE_PIX_TYPE_RGB888};

    if (!detector) {
        ESP_LOGE(TAG, "ENROLL_PHONE: Face detector not initialized");
        free(rgb_buf);
        httpd_resp_set_type(req, "application/json");
        httpd_resp_set_hdr(req, "Access-Control-Allow-Origin", "*");
        return httpd_resp_send(req, "{\"error\":\"AI detector not ready\"}", HTTPD_RESP_USE_STRLEN);
    }

    auto results = detector->run(img);
    ESP_LOGI(TAG, "ENROLL_PHONE: Detected %d faces", (int)results.size());

    if (results.empty()) {
        free(rgb_buf);
        httpd_resp_set_type(req, "application/json");
        httpd_resp_set_hdr(req, "Access-Control-Allow-Origin", "*");
        return httpd_resp_send(req, "{\"error\":\"No face detected in photo\"}", HTTPD_RESP_USE_STRLEN);
    }

    // Set enroll mode
    pending_enroll_name = urlDecode(String(name_str));
    if (pending_enroll_name.length() == 0) pending_enroll_name = "Chua co ten";
    is_enrolling = 1;

    RecognitionOutcome outcome = run_face_recognition(&img, &results);
    free(rgb_buf);

    // 5. Return JSON result
    char resp[256];
    if (outcome.overlay_status == OVERLAY_REAL_SUCCESS && outcome.id > 0) {
        snprintf(resp, sizeof(resp),
                 "{\"status\":\"ok\",\"id\":%d,\"name\":\"%s\",\"score\":%.2f}",
                 outcome.id, pending_enroll_name.length() > 0 ? name_str : "Face", outcome.liveness_score);
    } else if (outcome.overlay_status == OVERLAY_FAKE) {
        snprintf(resp, sizeof(resp),
                 "{\"status\":\"fake\",\"error\":\"Anti-spoof failed (%.0f%%)\",\"score\":%.2f}",
                 (1.0f - outcome.liveness_score) * 100.0f, outcome.liveness_score);
    } else {
        snprintf(resp, sizeof(resp),
                 "{\"status\":\"fail\",\"error\":\"%s\",\"score\":%.2f}",
                 outcome.label, outcome.liveness_score);
    }

    // Reset enroll state in case it wasn't consumed
    is_enrolling = 0;
    pending_enroll_name = "";

    ESP_LOGI(TAG, "ENROLL_PHONE: Result: %s", resp);
    httpd_resp_set_type(req, "application/json");
    httpd_resp_set_hdr(req, "Access-Control-Allow-Origin", "*");
    return httpd_resp_send(req, resp, HTTPD_RESP_USE_STRLEN);
}

static esp_err_t enroll_phone_handler_v2(httpd_req_t *req)
{
    if (require_auth(req) != ESP_OK) return ESP_OK;

    char *qbuf = nullptr;
    char name_str[64] = "";
    char role_str[32] = "";
    if (parse_get(req, &qbuf) == ESP_OK) {
        httpd_query_key_value(qbuf, "name", name_str, sizeof(name_str));
        httpd_query_key_value(qbuf, "role", role_str, sizeof(role_str));
        free(qbuf);
    }
    String final_name = urlDecode(String(name_str));
    if (final_name.length() == 0) final_name = "Chua co ten";
    String final_role = urlDecode(String(role_str));
    if (final_role.length() == 0) final_role = "Khach";

    size_t content_len = req->content_len;
    if (content_len == 0 || content_len > 100 * 1024) {
        httpd_resp_set_type(req, "application/json");
        httpd_resp_set_hdr(req, "Access-Control-Allow-Origin", "*");
        return httpd_resp_send(req, "{\"status\":\"fail\",\"error\":\"Invalid image size (max 100KB)\"}", HTTPD_RESP_USE_STRLEN);
    }

    uint8_t *jpg_buf = (uint8_t *)heap_caps_malloc(content_len, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    if (!jpg_buf) {
        httpd_resp_set_type(req, "application/json");
        httpd_resp_set_hdr(req, "Access-Control-Allow-Origin", "*");
        return httpd_resp_send(req, "{\"status\":\"fail\",\"error\":\"Out of memory\"}", HTTPD_RESP_USE_STRLEN);
    }

    size_t received = 0;
    while (received < content_len) {
        int ret = httpd_req_recv(req, (char *)(jpg_buf + received), content_len - received);
        if (ret <= 0) {
            if (ret == HTTPD_SOCK_ERR_TIMEOUT) continue;
            free(jpg_buf);
            httpd_resp_set_type(req, "application/json");
            httpd_resp_set_hdr(req, "Access-Control-Allow-Origin", "*");
            return httpd_resp_send(req, "{\"status\":\"fail\",\"error\":\"Receive error\"}", HTTPD_RESP_USE_STRLEN);
        }
        received += ret;
    }

    int jpeg_w = 0;
    int jpeg_h = 0;
    if (!readJpegSize(jpg_buf, received, &jpeg_w, &jpeg_h) || jpeg_w != 320 || jpeg_h != 240) {
        free(jpg_buf);
        httpd_resp_set_type(req, "application/json");
        httpd_resp_set_hdr(req, "Access-Control-Allow-Origin", "*");
        return httpd_resp_send(req, "{\"status\":\"fail\",\"error\":\"Image must be 320x240 JPEG\"}", HTTPD_RESP_USE_STRLEN);
    }
    ESP_LOGI(TAG, "ENROLL_PHONE: received JPEG %u bytes, %dx%d", (unsigned)received, jpeg_w, jpeg_h);

    size_t rgb_buf_size = (size_t)jpeg_w * jpeg_h * 3;
    uint8_t *rgb_buf = (uint8_t *)heap_caps_malloc(rgb_buf_size, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    if (!rgb_buf) {
        free(jpg_buf);
        httpd_resp_set_type(req, "application/json");
        httpd_resp_set_hdr(req, "Access-Control-Allow-Origin", "*");
        return httpd_resp_send(req, "{\"status\":\"fail\",\"error\":\"Out of memory for RGB\"}", HTTPD_RESP_USE_STRLEN);
    }

    bool decoded = decode_to_rgb888(jpg_buf, received, PIXFORMAT_JPEG, rgb_buf);
    if (!decoded && jpg2rgb565(jpg_buf, received, rgb_buf, JPG_SCALE_NONE)) {
        int pixel_count = jpeg_w * jpeg_h;
        uint16_t *src = (uint16_t *)rgb_buf;
        uint8_t *dst = rgb_buf;
        for (int i = pixel_count - 1; i >= 0; --i) {
            uint16_t p = src[i];
            uint8_t r = (p >> 11) & 0x1F;
            uint8_t g = (p >> 5) & 0x3F;
            uint8_t b = p & 0x1F;
            dst[i * 3 + 0] = (r * 255) / 31;
            dst[i * 3 + 1] = (g * 255) / 63;
            dst[i * 3 + 2] = (b * 255) / 31;
        }
        decoded = true;
    }
    free(jpg_buf);

    if (!decoded) {
        free(rgb_buf);
        httpd_resp_set_type(req, "application/json");
        httpd_resp_set_hdr(req, "Access-Control-Allow-Origin", "*");
        return httpd_resp_send(req, "{\"status\":\"fail\",\"error\":\"JPEG decode failed\"}", HTTPD_RESP_USE_STRLEN);
    }

    dl::image::img_t img = {rgb_buf, (uint16_t)jpeg_w, (uint16_t)jpeg_h, dl::image::DL_IMAGE_PIX_TYPE_RGB888};
    if (!detector) {
        free(rgb_buf);
        httpd_resp_set_type(req, "application/json");
        httpd_resp_set_hdr(req, "Access-Control-Allow-Origin", "*");
        return httpd_resp_send(req, "{\"status\":\"fail\",\"error\":\"AI detector not ready\"}", HTTPD_RESP_USE_STRLEN);
    }

    auto results = detector->run(img);
    if (results.empty() || results.size() != 1) {
        free(rgb_buf);
        httpd_resp_set_type(req, "application/json");
        httpd_resp_set_hdr(req, "Access-Control-Allow-Origin", "*");
        if (results.empty()) {
            return httpd_resp_send(req, "{\"status\":\"fail\",\"error\":\"No face detected in photo\"}", HTTPD_RESP_USE_STRLEN);
        }
        return httpd_resp_send(req, "{\"status\":\"fail\",\"error\":\"Only one face is allowed\"}", HTTPD_RESP_USE_STRLEN);
    }

    const dl::detect::result_t &phone_face = results.front();
    int phone_box_w = phone_face.box[2] - phone_face.box[0];
    int phone_box_h = phone_face.box[3] - phone_face.box[1];
    ESP_LOGI(TAG, "ENROLL_PHONE: face box x=%d y=%d w=%d h=%d",
             phone_face.box[0], phone_face.box[1], phone_box_w, phone_box_h);
    if (phone_box_w < 70 || phone_box_h < 70) {
        free(rgb_buf);
        httpd_resp_set_type(req, "application/json");
        httpd_resp_set_hdr(req, "Access-Control-Allow-Origin", "*");
        return httpd_resp_send(req, "{\"status\":\"fail\",\"error\":\"Face too small. Move closer and keep face centered\"}", HTTPD_RESP_USE_STRLEN);
    }

    RecognitionOutcome outcome = {};
    outcome.id = -1;
    outcome.liveness_score = -1.0f;
    outcome.anti_ran = false;
    outcome.overlay_status = OVERLAY_NONE;
    outcome.label[0] = '\0';
    if (!run_liveness_for_face(&img, results.front(), &outcome)) {
        free(rgb_buf);
        ESP_LOGW(TAG, "ENROLL_PHONE: anti-spoof rejected, score=%.2f", outcome.liveness_score);
        String resp = "{\"status\":\"fake\",\"error\":\"Anti-spoof failed\",\"score\":" + String(outcome.liveness_score, 2) + "}";
        httpd_resp_set_type(req, "application/json");
        httpd_resp_set_hdr(req, "Access-Control-Allow-Origin", "*");
        return httpd_resp_send(req, resp.c_str(), HTTPD_RESP_USE_STRLEN);
    }
    ESP_LOGI(TAG, "ENROLL_PHONE: anti-spoof accepted, score=%.2f", outcome.liveness_score);

    HumanFaceRecognizer *face_recognizer = ensure_recognizer();
    if (!face_recognizer || xSemaphoreTake(recognizer_mutex, pdMS_TO_TICKS(5000)) != pdTRUE) {
        free(rgb_buf);
        httpd_resp_set_type(req, "application/json");
        httpd_resp_set_hdr(req, "Access-Control-Allow-Origin", "*");
        return httpd_resp_send(req, "{\"status\":\"fail\",\"error\":\"Recognizer busy\"}", HTTPD_RESP_USE_STRLEN);
    }

    esp_err_t enroll_res = face_recognizer->enroll(img, results);
    int enrolled_id = (enroll_res == ESP_OK) ? face_recognizer->get_num_feats() : -1;
    xSemaphoreGive(recognizer_mutex);
    free(rgb_buf);

    String resp;
    if (enroll_res == ESP_OK && enrolled_id > 0) {
        sync_face_raw_map();
        time_t now;
        struct tm timeinfo;
        char time_str[32];
        time(&now);
        localtime_r(&now, &timeinfo);
        strftime(time_str, sizeof(time_str), "%Y-%m-%d %H:%M:%S", &timeinfo);

        prefs.putString(face_name_key(enrolled_id).c_str(), final_name);
        prefs.putString(face_time_key(enrolled_id).c_str(), String(time_str));
        prefs.putString(face_role_key(enrolled_id).c_str(), final_role);
        pending_rfid_enroll_face_id = enrolled_id;
        append_log("Phone face enrolled, waiting RFID link: " + final_name);
        resp = "{\"status\":\"ok\",\"id\":" + String(enrolled_id) +
               ",\"name\":\"" + jsonEscape(final_name) +
               "\",\"score\":" + String(outcome.liveness_score, 2) + "}";
    } else {
        resp = "{\"status\":\"fail\",\"error\":\"Enroll failed\",\"score\":" + String(outcome.liveness_score, 2) + "}";
    }

    httpd_resp_set_type(req, "application/json");
    httpd_resp_set_hdr(req, "Access-Control-Allow-Origin", "*");
    return httpd_resp_send(req, resp.c_str(), HTTPD_RESP_USE_STRLEN);
}

static esp_err_t setname_handler(httpd_req_t *req)
{
    if (require_auth(req) != ESP_OK) {
        return ESP_OK;
    }

    char *buf = nullptr;
    if (parse_get(req, &buf) != ESP_OK) return httpd_resp_send_500(req);

    char id_str[16] = "";
    char name_str[64] = "";
    char role_str[32] = "";
    char rfid_str[32] = "";
    esp_err_t res = ESP_FAIL;
    if (httpd_query_key_value(buf, "id", id_str, sizeof(id_str)) == ESP_OK &&
        httpd_query_key_value(buf, "name", name_str, sizeof(name_str)) == ESP_OK) {
        int face_id = atoi(id_str);
        prefs.putString(face_name_key(face_id).c_str(), urlDecode(String(name_str)));
        if (httpd_query_key_value(buf, "role", role_str, sizeof(role_str)) == ESP_OK) {
            prefs.putString(face_role_key(face_id).c_str(), urlDecode(String(role_str)));
        }
        if (httpd_query_key_value(buf, "rfid", rfid_str, sizeof(rfid_str)) == ESP_OK) {
            String uid = urlDecode(String(rfid_str));
            uid.trim();
            uid.toUpperCase();
            if (uid.length() > 0) {
                bool duplicate = false;
                int count = 0;
                if (ensure_recognizer() && xSemaphoreTake(recognizer_mutex, pdMS_TO_TICKS(1000)) == pdTRUE) {
                    count = recognizer->get_num_feats();
                    xSemaphoreGive(recognizer_mutex);
                }
                for (int id = 1; id <= count; ++id) {
                    if (id == face_id) continue;
                    String existing_uid = prefs.getString(face_rfid_key(id).c_str(), "");
                    if (existing_uid.equalsIgnoreCase(uid)) {
                        duplicate = true;
                        break;
                    }
                }
                if (duplicate) {
                    free(buf);
                    return httpd_resp_sendstr(req, "RFID already linked to another face");
                }
                prefs.putString(face_rfid_key(face_id).c_str(), uid);
            }
        }
        res = httpd_resp_sendstr(req, "OK");
    } else {
        res = httpd_resp_send_500(req);
    }
    free(buf);
    return res;
}

static esp_err_t arm_rfid_link_handler(httpd_req_t *req)
{
    if (require_auth(req) != ESP_OK) {
        return ESP_OK;
    }

    char *buf = nullptr;
    if (parse_get(req, &buf) != ESP_OK) return httpd_resp_send_500(req);

    char id_str[16] = "";
    esp_err_t res = ESP_FAIL;
    if (httpd_query_key_value(buf, "id", id_str, sizeof(id_str)) == ESP_OK) {
        int face_id = atoi(id_str);
        int count = 0;
        if (ensure_recognizer() && xSemaphoreTake(recognizer_mutex, pdMS_TO_TICKS(1000)) == pdTRUE) {
            count = recognizer->get_num_feats();
            xSemaphoreGive(recognizer_mutex);
        }

        if (face_id > 0 && face_id <= count) {
            pending_rfid_enroll_face_id = face_id;
            rfid_auth_window_active = false;
            rfid_expected_face_id = 0;
            String name = prefs.getString(face_name_key(face_id).c_str(), "Face " + String(face_id));
            append_log("Ready to link RFID for " + name + " (ID " + String(face_id) + ")");
            res = httpd_resp_sendstr(req, "READY");
        } else {
            res = httpd_resp_send(req, "Face not found", HTTPD_RESP_USE_STRLEN);
        }
    } else {
        res = httpd_resp_send_500(req);
    }
    free(buf);
    return res;
}

static esp_err_t cancel_rfid_link_handler(httpd_req_t *req)
{
    if (require_auth(req) != ESP_OK) {
        return ESP_OK;
    }

    if (pending_rfid_enroll_face_id > 0) {
        append_log("RFID link cancelled for Face " + String(pending_rfid_enroll_face_id));
    }
    pending_rfid_enroll_face_id = 0;
    httpd_resp_set_hdr(req, "Access-Control-Allow-Origin", "*");
    return httpd_resp_sendstr(req, "OK");
}

// ==================== SET RFID MANUAL (GET) ====================
static esp_err_t set_rfid_manual_handler(httpd_req_t *req)
{
    if (require_auth(req) != ESP_OK) {
        return ESP_OK;
    }

    char *buf = nullptr;
    if (parse_get(req, &buf) != ESP_OK) return httpd_resp_send_500(req);

    char id_str[16] = "";
    char uid_str[64] = "";
    httpd_query_key_value(buf, "id", id_str, sizeof(id_str));
    httpd_query_key_value(buf, "uid", uid_str, sizeof(uid_str));
    free(buf);

    int face_id = atoi(id_str);
    String uid = urlDecode(String(uid_str));

    if (face_id <= 0 || uid.length() < 4) {
        return httpd_resp_sendstr(req, "Invalid id or uid");
    }

    // Kiểm tra face_id hợp lệ
    int count = 0;
    if (ensure_recognizer() && xSemaphoreTake(recognizer_mutex, pdMS_TO_TICKS(1000)) == pdTRUE) {
        count = recognizer->get_num_feats();
        xSemaphoreGive(recognizer_mutex);
    }
    if (face_id > count) {
        return httpd_resp_sendstr(req, "Face ID not found");
    }

    // Lưu RFID UID
    for (int id = 1; id <= count; ++id) {
        if (id == face_id) {
            continue;
        }
        String existing_uid = prefs.getString(face_rfid_key(id).c_str(), "");
        if (existing_uid.length() > 0 && existing_uid == uid) {
            return httpd_resp_sendstr(req, "RFID already linked to another face");
        }
    }

    prefs.putString(face_rfid_key(face_id).c_str(), uid);
    pending_rfid_enroll_face_id = 0; // Clear pending state
    String name = prefs.getString(face_name_key(face_id).c_str(), "Face " + String(face_id));
    append_log("RFID manual link: " + name + " (ID " + String(face_id) + ") -> " + uid);
    ESP_LOGI(TAG, "RFID manual link: Face %d -> %s", face_id, uid.c_str());

    return httpd_resp_sendstr(req, "OK");
}

static esp_err_t rfid_state_handler(httpd_req_t *req)
{
    if (require_auth(req) != ESP_OK) {
        return ESP_OK;
    }

    String pending_name = "";
    if (pending_rfid_enroll_face_id > 0) {
        pending_name = prefs.getString(face_name_key(pending_rfid_enroll_face_id).c_str(), "");
        if (pending_name.length() == 0) {
            pending_name = "Face " + String(pending_rfid_enroll_face_id);
        }
    }

    String expected_name = "";
    if (rfid_expected_face_id > 0) {
        expected_name = prefs.getString(face_name_key(rfid_expected_face_id).c_str(), "");
        if (expected_name.length() == 0) {
            expected_name = "Face " + String(rfid_expected_face_id);
        }
    }

    String json = "{";
    json += "\"pending_enroll_face_id\":" + String(pending_rfid_enroll_face_id) + ",";
    json += "\"pending_enroll_name\":\"" + jsonEscape(pending_name) + "\",";
    json += "\"rfid_auth_window_active\":" + String(rfid_auth_window_active ? "true" : "false") + ",";
    json += "\"rfid_expected_face_id\":" + String(rfid_expected_face_id) + ",";
    json += "\"rfid_expected_name\":\"" + jsonEscape(expected_name) + "\"";
    json += "}";

    httpd_resp_set_type(req, "application/json");
    httpd_resp_set_hdr(req, "Access-Control-Allow-Origin", "*");
    return httpd_resp_send(req, json.c_str(), HTTPD_RESP_USE_STRLEN);
}

static esp_err_t get_faces_handler(httpd_req_t *req)
{
    if (require_auth(req) != ESP_OK) {
        return ESP_OK;
    }

    int count = 0;
    if (ensure_recognizer() && xSemaphoreTake(recognizer_mutex, pdMS_TO_TICKS(1000)) == pdTRUE) {
        count = recognizer->get_num_feats();
        xSemaphoreGive(recognizer_mutex);
    }

    String json = "[";
    for (int id = 1; id <= count; ++id) {
        if (id > 1) json += ",";
        String name = prefs.getString(face_name_key(id).c_str(), "Unknown");
        String rfid = prefs.getString(face_rfid_key(id).c_str(), "");
        String time = prefs.getString(face_time_key(id).c_str(), "");
        String role = prefs.getString(face_role_key(id).c_str(), "");
        bool locked = prefs.getBool(face_lock_key(id).c_str(), false);
        int fail_count = prefs.getInt(face_fail_key(id).c_str(), 0);
        json += "{\"id\":" + String(id) +
                ",\"name\":\"" + jsonEscape(name) +
                "\",\"role\":\"" + jsonEscape(role) +
                "\",\"rfid\":\"" + jsonEscape(rfid) +
                "\",\"time\":\"" + jsonEscape(time) +
                "\",\"locked\":" + String(locked ? "true" : "false") +
                ",\"fail_count\":" + String(fail_count) + "}";
    }
    json += "]";
    httpd_resp_set_type(req, "application/json");
    httpd_resp_set_hdr(req, "Access-Control-Allow-Origin", "*");
    return httpd_resp_send(req, json.c_str(), HTTPD_RESP_USE_STRLEN);
}

static esp_err_t unlock_face_handler(httpd_req_t *req)
{
    if (require_auth(req) != ESP_OK) return ESP_OK;

    char *buf = nullptr;
    if (parse_get(req, &buf) != ESP_OK) return httpd_resp_send_500(req);

    char id_str[16] = "";
    esp_err_t res = ESP_FAIL;
    if (httpd_query_key_value(buf, "id", id_str, sizeof(id_str)) == ESP_OK) {
        int face_id = atoi(id_str);
        if (face_id > 0) {
            prefs.putBool(face_lock_key(face_id).c_str(), false);
            prefs.putInt(face_fail_key(face_id).c_str(), 0);
            String name = prefs.getString(face_name_key(face_id).c_str(), "Face " + String(face_id));
            append_log("Mo khoa nguoi dung: " + name + " (ID " + String(face_id) + ")");
            res = httpd_resp_sendstr(req, "OK");
        } else {
            res = httpd_resp_send_500(req);
        }
    } else {
        res = httpd_resp_send_500(req);
    }
    free(buf);
    return res;
}

static esp_err_t delete_face_handler(httpd_req_t *req)
{
    if (require_auth(req) != ESP_OK) {
        return ESP_OK;
    }

    char *buf = nullptr;
    if (parse_get(req, &buf) != ESP_OK) return httpd_resp_send_500(req);

    char id_str[16] = "";
    if (httpd_query_key_value(buf, "id", id_str, sizeof(id_str)) != ESP_OK) {
        free(buf);
        return httpd_resp_send_500(req);
    }
    free(buf);

    int visible_id = atoi(id_str);
    sync_face_raw_map();
    uint32_t raw_id = prefs.getUInt(face_raw_key(visible_id).c_str(), 0);
    if (raw_id == 0) return httpd_resp_send(req, "Face not found", HTTPD_RESP_USE_STRLEN);

    int old_count = 0;
    esp_err_t del_res = ESP_FAIL;
    if (ensure_recognizer() && xSemaphoreTake(recognizer_mutex, pdMS_TO_TICKS(5000)) == pdTRUE) {
        old_count = recognizer->get_num_feats();
        del_res = recognizer->delete_feat((uint16_t)raw_id);
        xSemaphoreGive(recognizer_mutex);
    }
    if (del_res != ESP_OK) return httpd_resp_send(req, "Delete failed", HTTPD_RESP_USE_STRLEN);

    if (rfid_expected_face_id == visible_id) {
        rfid_auth_window_active = false;
        rfid_expected_face_id = 0;
    } else if (rfid_expected_face_id > visible_id) {
        rfid_expected_face_id = rfid_expected_face_id - 1;
    }

    if (pending_rfid_enroll_face_id == visible_id) {
        pending_rfid_enroll_face_id = 0;
    } else if (pending_rfid_enroll_face_id > visible_id) {
        pending_rfid_enroll_face_id = pending_rfid_enroll_face_id - 1;
    }

    shift_occupants_after_face_delete(visible_id);
    shift_face_metadata_left(visible_id, old_count);
    sync_face_raw_map();
    return httpd_resp_sendstr(req, "OK");
}

static esp_err_t delete_all_handler(httpd_req_t *req)
{
    if (require_auth(req) != ESP_OK) {
        return ESP_OK;
    }

    if (ensure_recognizer() && xSemaphoreTake(recognizer_mutex, pdMS_TO_TICKS(5000)) == pdTRUE) {
        int old_count = recognizer->get_num_feats();
        recognizer->clear_all_feats();
        xSemaphoreGive(recognizer_mutex);

        // Selectively remove face metadata keys instead of prefs.clear()
        for (int id = 1; id <= old_count; ++id) {
            prefs.remove(face_name_key(id).c_str());
            prefs.remove(face_time_key(id).c_str());
            prefs.remove(face_rfid_key(id).c_str());
            prefs.remove(face_role_key(id).c_str());
            prefs.remove(face_fail_key(id).c_str());
            prefs.remove(face_lock_key(id).c_str());
            prefs.remove(face_raw_key(id).c_str());
        }
        prefs.remove("raw_count");
    }
    clear_occupants();
    rfid_auth_window_active = false;
    rfid_expected_face_id = 0;
    pending_rfid_enroll_face_id = 0;
    LittleFS.remove("/logs.json");
    sync_face_raw_map();
    return httpd_resp_sendstr(req, "OK");
}

static esp_err_t clear_logs_handler(httpd_req_t *req)
{
    if (require_auth(req) != ESP_OK) {
        return ESP_OK;
    }

    LittleFS.remove("/logs.json");
    return httpd_resp_sendstr(req, "OK");
}

static esp_err_t logs_handler(httpd_req_t *req)
{
    if (require_auth(req) != ESP_OK) {
        return ESP_OK;
    }

    httpd_resp_set_type(req, "application/json");
    httpd_resp_set_hdr(req, "Access-Control-Allow-Origin", "*");

    if (!LittleFS.begin(true)) {
        return httpd_resp_send(req, "[]", HTTPD_RESP_USE_STRLEN);
    }

    File file = LittleFS.open("/logs.json", "r");
    if (!file) {
        return httpd_resp_send(req, "[]", HTTPD_RESP_USE_STRLEN);
    }

    String body;
    while (file.available()) {
        body += static_cast<char>(file.read());
    }
    file.close();

    body.trim();
    while (body.endsWith(",") || body.endsWith("\n") || body.endsWith("\r")) {
        if (body.endsWith(",")) {
            body.remove(body.length() - 1);
        } else {
            body.trim();
        }
    }

    String json = body.length() > 0 ? "[" + body + "]" : "[]";
    return httpd_resp_send(req, json.c_str(), HTTPD_RESP_USE_STRLEN);
}

static esp_err_t login_handler(httpd_req_t *req)
{
    bool reset_session = false;
    char *buf = nullptr;
    if (parse_get(req, &buf) == ESP_OK) {
        char reset[8] = "";
        reset_session = httpd_query_key_value(buf, "reset", reset, sizeof(reset)) == ESP_OK && strcmp(reset, "1") == 0;
        free(buf);
    }

    if (reset_session) {
        clear_otp();
        httpd_resp_set_hdr(req, "Set-Cookie", "auth=0; Path=/; Max-Age=0");
    }

    if (!reset_session && is_authenticated(req)) {
        return redirect_to(req, "/");
    }
    if (!reset_session && is_step1_authenticated(req)) {
        return redirect_to(req, "/verify_2fa");
    }

    const char *page =
        "<!doctype html><html><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width,initial-scale=1'>"
        "<title>Door Login</title>"
        "<style>body{font-family:Arial,sans-serif;background:#f8fafc;display:flex;align-items:center;justify-content:center;height:100vh;margin:0}"
        ".card{background:#fff;padding:28px;border-radius:14px;box-shadow:0 10px 30px rgba(15,23,42,.12);width:320px}"
        "input,button{width:100%;padding:12px;margin-top:12px;border-radius:10px;border:1px solid #cbd5e1;box-sizing:border-box}"
        "button{background:#0f766e;color:#fff;border:none;font-weight:600;cursor:pointer}"
        "a{display:block;text-align:center;margin-top:14px;color:#0f766e;font-weight:600;text-decoration:none}</style></head><body>"
        "<div class='card'><h2>Door Login</h2><p>Enter admin passcode</p>"
        "<form action='/do_login' method='get'><input type='password' name='code' maxlength='16' autofocus required>"
        "<button type='submit'>Continue</button></form><a href='/forgot_password'>Quên mật khẩu?</a></div></body></html>";

    httpd_resp_set_type(req, "text/html");
    return httpd_resp_sendstr(req, page);
}

static esp_err_t do_login_handler(httpd_req_t *req)
{
    char *buf = nullptr;
    if (parse_get(req, &buf) != ESP_OK) {
        return redirect_to(req, "/login");
    }

    char code[32] = "";
    bool ok = false;
    if (httpd_query_key_value(buf, "code", code, sizeof(code)) == ESP_OK) {
        ok = strcmp(code, passkey.c_str()) == 0;
    }
    free(buf);

    if (!ok) {
        const char *page =
            "<!doctype html><html><head><meta charset='utf-8'><title>Login Failed</title></head>"
            "<body style='font-family:Arial,sans-serif;padding:32px'><h2>Wrong passcode</h2>"
            "<p><a href='/login'>Back to login</a></p></body></html>";
        httpd_resp_set_type(req, "text/html");
        return httpd_resp_sendstr(req, page);
    }

    issue_otp();
    httpd_resp_set_hdr(req, "Set-Cookie", "auth=1; Path=/; Max-Age=180; HttpOnly");
    return redirect_to(req, "/verify_2fa");
}

static esp_err_t verify_2fa_handler(httpd_req_t *req)
{
    if (is_authenticated(req)) {
        return redirect_to(req, "/");
    }
    if (!is_step1_authenticated(req)) {
        return redirect_to(req, "/login");
    }

    int remain = otp_remaining_seconds();
    if (remain <= 0) {
        clear_otp();
        httpd_resp_set_hdr(req, "Set-Cookie", "auth=0; Path=/; Max-Age=0");
        const char *expired_page =
            "<!doctype html><html><head><meta charset='utf-8'><title>OTP Expired</title></head>"
            "<body style='font-family:Arial,sans-serif;padding:32px'><h2>OTP đã hết hạn</h2>"
            "<p><a href='/login?reset=1'>Quay lại đăng nhập</a></p></body></html>";
        httpd_resp_set_type(req, "text/html");
        return httpd_resp_sendstr(req, expired_page);
    }

    String page = String(
        "<!doctype html><html><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width,initial-scale=1'>"
        "<title>OTP Verify</title>"
        "<style>body{font-family:Arial,sans-serif;background:#f8fafc;display:flex;align-items:center;justify-content:center;height:100vh;margin:0}"
        ".card{background:#fff;padding:28px;border-radius:14px;box-shadow:0 10px 30px rgba(15,23,42,.12);width:320px}"
        "input,button{width:100%;padding:12px;margin-top:12px;border-radius:10px;border:1px solid #cbd5e1;box-sizing:border-box}"
        "button{background:#2563eb;color:#fff;border:none;font-weight:600;cursor:pointer}"
        ".secondary{display:block;width:100%;box-sizing:border-box;text-align:center;padding:12px;margin-top:12px;border-radius:10px;background:#f1f5f9;color:#0f172a;font-weight:600;text-decoration:none}"
        ".timer{margin:8px 0 4px;color:#64748b;font-size:14px}.timer b{color:#ef4444}</style></head><body>"
        "<div class='card'><h2>Two-factor verification</h2><p>Enter the OTP sent to email</p>") +
        "<p class='timer'>Mã OTP hết hạn sau <b id='otp-timer'>" + String(remain) + "</b> giây</p>"
        "<form action='/do_verify_2fa' method='get'><input type='text' name='otp' maxlength='6' autofocus required>"
        "<button type='submit'>Verify</button></form><a class='secondary' href='/login?reset=1'>Quay lại nhập mật khẩu</a></div>"
        "<script>let t=" + String(remain) + ";const el=document.getElementById('otp-timer');setInterval(()=>{t--;if(t<=0){el.textContent='0';location.href='/login?reset=1';}else{el.textContent=t;}},1000);</script>"
        "</body></html>";

    httpd_resp_set_type(req, "text/html");
    return httpd_resp_send(req, page.c_str(), HTTPD_RESP_USE_STRLEN);
}

static esp_err_t do_verify_2fa_handler(httpd_req_t *req)
{
    if (!is_step1_authenticated(req)) {
        return redirect_to(req, "/login");
    }

    char *buf = nullptr;
    if (parse_get(req, &buf) != ESP_OK) {
        return redirect_to(req, "/verify_2fa");
    }

    char otp[10] = "";
    bool ok = false;
    if (httpd_query_key_value(buf, "otp", otp, sizeof(otp)) == ESP_OK) {
        ok = otp_is_valid(otp);
    }
    free(buf);

    if (!ok) {
        if (otp_remaining_seconds() <= 0) {
            clear_otp();
            httpd_resp_set_hdr(req, "Set-Cookie", "auth=0; Path=/; Max-Age=0");
        }
        const char *page =
            "<!doctype html><html><head><meta charset='utf-8'><title>OTP Failed</title></head>"
            "<body style='font-family:Arial,sans-serif;padding:32px'><h2>Invalid OTP</h2>"
            "<p><a href='/verify_2fa'>Try again</a></p><p><a href='/login?reset=1'>Back to login</a></p></body></html>";
        httpd_resp_set_type(req, "text/html");
        return httpd_resp_sendstr(req, page);
    }

    clear_otp();
    httpd_resp_set_hdr(req, "Set-Cookie", "auth=2; Path=/; HttpOnly");
    return redirect_to(req, "/");
}

static esp_err_t forgot_password_handler(httpd_req_t *req)
{
    if (is_authenticated(req)) {
        return redirect_to(req, "/");
    }

    issue_otp();

    int remain = otp_remaining_seconds();
    String page = String(
        "<!doctype html><html><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width,initial-scale=1'>"
        "<title>Reset Password</title>"
        "<style>body{font-family:Arial,sans-serif;background:#f8fafc;display:flex;align-items:center;justify-content:center;height:100vh;margin:0}"
        ".card{background:#fff;padding:28px;border-radius:14px;box-shadow:0 10px 30px rgba(15,23,42,.12);width:340px}"
        "input,button{width:100%;padding:12px;margin-top:12px;border-radius:10px;border:1px solid #cbd5e1;box-sizing:border-box}"
        "button{background:#0f766e;color:#fff;border:none;font-weight:600;cursor:pointer}"
        "a{display:block;text-align:center;margin-top:14px;color:#64748b;text-decoration:none}"
        ".timer{margin:8px 0 4px;color:#64748b;font-size:14px}.timer b{color:#ef4444}</style></head><body>"
        "<div class='card'><h2>Đặt lại mật khẩu</h2><p>Mã xác minh đã được gửi đến email quản trị.</p>") +
        "<p class='timer'>Mã xác minh hết hạn sau <b id='otp-timer'>" + String(remain) + "</b> giây</p>"
        "<form action='/reset_password' method='get'>"
        "<input type='text' name='otp' maxlength='6' placeholder='Mã xác minh' autofocus required>"
        "<input type='password' name='code' maxlength='16' placeholder='Mật khẩu mới' required>"
        "<button type='submit'>Đổi mật khẩu</button></form><a href='/login?reset=1'>Quay lại đăng nhập</a></div>"
        "<script>let t=" + String(remain) + ";const el=document.getElementById('otp-timer');setInterval(()=>{t--;if(t<=0){el.textContent='0';location.href='/login?reset=1';}else{el.textContent=t;}},1000);</script>"
        "</body></html>";

    httpd_resp_set_type(req, "text/html");
    return httpd_resp_send(req, page.c_str(), HTTPD_RESP_USE_STRLEN);
}

static esp_err_t reset_password_handler(httpd_req_t *req)
{
    char *buf = nullptr;
    if (parse_get(req, &buf) != ESP_OK) {
        return redirect_to(req, "/forgot_password");
    }

    char otp[10] = "";
    char code[16] = "";
    bool otp_ok = false;
    bool pass_ok = false;
    if (httpd_query_key_value(buf, "otp", otp, sizeof(otp)) == ESP_OK) {
        otp_ok = otp_is_valid(otp);
    }
    if (httpd_query_key_value(buf, "code", code, sizeof(code)) == ESP_OK) {
        size_t len = strlen(code);
        pass_ok = len >= 4 && len < sizeof(code);
    }
    free(buf);

    if (!otp_ok || !pass_ok) {
        if (otp_remaining_seconds() <= 0) {
            clear_otp();
        }
        const char *page =
            "<!doctype html><html><head><meta charset='utf-8'><title>Reset Failed</title></head>"
            "<body style='font-family:Arial,sans-serif;padding:32px'><h2>Không thể đổi mật khẩu</h2>"
            "<p>Mã xác minh không đúng hoặc mật khẩu mới dưới 4 ký tự.</p>"
            "<p><a href='/forgot_password'>Thử lại</a></p></body></html>";
        httpd_resp_set_type(req, "text/html");
        return httpd_resp_sendstr(req, page);
    }

    passkey = String(code);
    prefs.putString("login_pass", passkey);
    clear_otp();
    httpd_resp_set_hdr(req, "Set-Cookie", "auth=0; Path=/; Max-Age=0");
    const char *page =
        "<!doctype html><html><head><meta charset='utf-8'><title>Reset OK</title></head>"
        "<body style='font-family:Arial,sans-serif;padding:32px'><h2>Đã đổi mật khẩu</h2>"
        "<p><a href='/login'>Đăng nhập lại</a></p></body></html>";
    httpd_resp_set_type(req, "text/html");
    return httpd_resp_sendstr(req, page);
}

static esp_err_t logout_handler(httpd_req_t *req)
{
    httpd_resp_set_hdr(req, "Set-Cookie", "auth=0; Path=/; Max-Age=0");
    return redirect_to(req, "/login");
}

static esp_err_t setpass_handler(httpd_req_t *req)
{
    if (require_auth(req) != ESP_OK) {
        return ESP_OK;
    }

    char *buf = nullptr;
    if (parse_get(req, &buf) != ESP_OK) {
        return httpd_resp_send(req, "Missing code", HTTPD_RESP_USE_STRLEN);
    }

    char code[16] = "";
    esp_err_t res = ESP_OK;
    if (httpd_query_key_value(buf, "code", code, sizeof(code)) == ESP_OK) {
        if (strlen(code) >= 4 && strlen(code) < sizeof(code)) {
            passkey = String(code);
            prefs.putString("login_pass", passkey);
            res = httpd_resp_send(req, "Pass updated", HTTPD_RESP_USE_STRLEN);
        } else {
            res = httpd_resp_send(req, "Pass must be 4-15 characters", HTTPD_RESP_USE_STRLEN);
        }
    } else {
        res = httpd_resp_send(req, "Missing code", HTTPD_RESP_USE_STRLEN);
    }
    free(buf);
    return res;
}

static esp_err_t cmd_handler(httpd_req_t *req)
{
    if (require_auth(req) != ESP_OK) {
        return ESP_OK;
    }

    char *buf = nullptr;
    if (parse_get(req, &buf) != ESP_OK) {
        return httpd_resp_send_500(req);
    }

    char variable[32] = "";
    char value[32] = "";
    char cmd[32] = "";
    sensor_t *sensor = esp_camera_sensor_get();
    int res = 0;

    if (httpd_query_key_value(buf, "cmd", cmd, sizeof(cmd)) == ESP_OK) {
        if (strcmp(cmd, "open") == 0) {
            free(buf);
            return httpd_resp_send(req, "Remote open disabled", HTTPD_RESP_USE_STRLEN);
        }
    }

    if (!sensor ||
        httpd_query_key_value(buf, "var", variable, sizeof(variable)) != ESP_OK ||
        httpd_query_key_value(buf, "val", value, sizeof(value)) != ESP_OK) {
        free(buf);
        return httpd_resp_send_500(req);
    }

    int val = atoi(value);
    if (!strcmp(variable, "framesize")) {
        if (sensor->pixformat == PIXFORMAT_JPEG) res = sensor->set_framesize(sensor, (framesize_t)val);
    } else if (!strcmp(variable, "quality")) {
        res = sensor->set_quality(sensor, val);
    } else if (!strcmp(variable, "contrast")) {
        res = sensor->set_contrast(sensor, val);
    } else if (!strcmp(variable, "brightness")) {
        res = sensor->set_brightness(sensor, val);
    } else if (!strcmp(variable, "saturation")) {
        res = sensor->set_saturation(sensor, val);
    } else if (!strcmp(variable, "gainceiling")) {
        res = sensor->set_gainceiling(sensor, (gainceiling_t)val);
    } else if (!strcmp(variable, "colorbar")) {
        res = sensor->set_colorbar(sensor, val);
    } else if (!strcmp(variable, "awb")) {
        res = sensor->set_whitebal(sensor, val);
    } else if (!strcmp(variable, "agc")) {
        res = sensor->set_gain_ctrl(sensor, val);
    } else if (!strcmp(variable, "aec")) {
        res = sensor->set_exposure_ctrl(sensor, val);
    } else if (!strcmp(variable, "hmirror")) {
        res = sensor->set_hmirror(sensor, val);
    } else if (!strcmp(variable, "vflip")) {
        res = sensor->set_vflip(sensor, val);
    } else if (!strcmp(variable, "awb_gain")) {
        res = sensor->set_awb_gain(sensor, val);
    } else if (!strcmp(variable, "agc_gain")) {
        res = sensor->set_agc_gain(sensor, val);
    } else if (!strcmp(variable, "aec_value")) {
        res = sensor->set_aec_value(sensor, val);
    } else if (!strcmp(variable, "aec2")) {
        res = sensor->set_aec2(sensor, val);
    } else if (!strcmp(variable, "dcw")) {
        res = sensor->set_dcw(sensor, val);
    } else if (!strcmp(variable, "bpc")) {
        res = sensor->set_bpc(sensor, val);
    } else if (!strcmp(variable, "wpc")) {
        res = sensor->set_wpc(sensor, val);
    } else if (!strcmp(variable, "raw_gma")) {
        res = sensor->set_raw_gma(sensor, val);
    } else if (!strcmp(variable, "lenc")) {
        res = sensor->set_lenc(sensor, val);
    } else if (!strcmp(variable, "special_effect")) {
        res = sensor->set_special_effect(sensor, val);
    } else if (!strcmp(variable, "wb_mode")) {
        res = sensor->set_wb_mode(sensor, val);
    } else if (!strcmp(variable, "ae_level")) {
        res = sensor->set_ae_level(sensor, val);
    } else if (!strcmp(variable, "face_detect")) {
        detection_enabled = val; if (!detection_enabled) recognition_enabled = 0;
    } else if (!strcmp(variable, "face_recognize")) {
        recognition_enabled = val; if (recognition_enabled) detection_enabled = 1;
    } else if (!strcmp(variable, "face_enroll")) {
        is_enrolling = val ? 1 : 0;
        if (is_enrolling) {
            ensure_ai_background();
            notify_ai_task();
        }
    } else {
        res = -1;
    }

    free(buf);
    if (res < 0) return httpd_resp_send_500(req);
    httpd_resp_set_hdr(req, "Access-Control-Allow-Origin", "*");
    return httpd_resp_send(req, nullptr, 0);
}

static esp_err_t index_handler(httpd_req_t *req)
{
    if (is_step1_authenticated(req) && !is_authenticated(req)) {
        return redirect_to(req, "/verify_2fa");
    }
    if (!is_authenticated(req)) {
        return redirect_to(req, "/login");
    }

    if (!esp_camera_sensor_get()) return httpd_resp_send_500(req);
    httpd_resp_set_type(req, "text/html");
    httpd_resp_set_hdr(req, "Content-Encoding", "gzip");
    return httpd_resp_send(req, (const char *)index_ov2640_html_gz, index_ov2640_html_gz_len);
}

static esp_err_t bmp_handler(httpd_req_t *req)
{
    if (require_auth(req) != ESP_OK) {
        return ESP_OK;
    }

    camera_fb_t *fb = esp_camera_fb_get();
    if (!fb) return httpd_resp_send_500(req);
    uint8_t *bmp = nullptr; size_t bmp_len = 0;
    if (!frame2bmp(fb, &bmp, &bmp_len)) {
        esp_camera_fb_return(fb);
        return httpd_resp_send_500(req);
    }
    httpd_resp_set_type(req, "image/x-windows-bmp");
    httpd_resp_set_hdr(req, "Access-Control-Allow-Origin", "*");
    esp_err_t res = httpd_resp_send(req, (const char *)bmp, bmp_len);
    free(bmp);
    esp_camera_fb_return(fb);
    return res;
}

static int print_reg(char *p, sensor_t *s, uint16_t reg, uint32_t mask)
{
    return sprintf(p, "\"0x%x\":%u,", reg, s->get_reg(s, reg, mask));
}

static esp_err_t status_handler(httpd_req_t *req)
{
    if (require_auth(req) != ESP_OK) {
        return ESP_OK;
    }

    static char json_response[1800];
    sensor_t *s = esp_camera_sensor_get();
    if (!s) return httpd_resp_send_500(req);

    int face_count = 0;
    if (ensure_recognizer() && xSemaphoreTake(recognizer_mutex, pdMS_TO_TICKS(1000)) == pdTRUE) {
        face_count = recognizer->get_num_feats();
        xSemaphoreGive(recognizer_mutex);
    }

    char *p = json_response;
    *p++ = '{';
    if (s->id.PID == OV2640_PID) {
        p += print_reg(p, s, 0xd3, 0xFF);
        p += print_reg(p, s, 0x111, 0xFF);
        p += print_reg(p, s, 0x132, 0xFF);
    }
    p += sprintf(p, "\"xclk\":%u,", s->xclk_freq_hz / 1000000);
    p += sprintf(p, "\"pixformat\":%u,", s->pixformat);
    p += sprintf(p, "\"framesize\":%u,", s->status.framesize);
    p += sprintf(p, "\"quality\":%u,", s->status.quality);
    p += sprintf(p, "\"brightness\":%d,", s->status.brightness);
    p += sprintf(p, "\"contrast\":%d,", s->status.contrast);
    p += sprintf(p, "\"saturation\":%d,", s->status.saturation);
    p += sprintf(p, "\"sharpness\":%d,", s->status.sharpness);
    p += sprintf(p, "\"special_effect\":%u,", s->status.special_effect);
    p += sprintf(p, "\"wb_mode\":%u,", s->status.wb_mode);
    p += sprintf(p, "\"awb\":%u,", s->status.awb);
    p += sprintf(p, "\"awb_gain\":%u,", s->status.awb_gain);
    p += sprintf(p, "\"aec\":%u,", s->status.aec);
    p += sprintf(p, "\"aec2\":%u,", s->status.aec2);
    p += sprintf(p, "\"ae_level\":%d,", s->status.ae_level);
    p += sprintf(p, "\"aec_value\":%u,", s->status.aec_value);
    p += sprintf(p, "\"agc\":%u,", s->status.agc);
    p += sprintf(p, "\"agc_gain\":%u,", s->status.agc_gain);
    p += sprintf(p, "\"gainceiling\":%u,", s->status.gainceiling);
    p += sprintf(p, "\"bpc\":%u,", s->status.bpc);
    p += sprintf(p, "\"wpc\":%u,", s->status.wpc);
    p += sprintf(p, "\"raw_gma\":%u,", s->status.raw_gma);
    p += sprintf(p, "\"lenc\":%u,", s->status.lenc);
    p += sprintf(p, "\"hmirror\":%u,", s->status.hmirror);
    p += sprintf(p, "\"vflip\":%u,", s->status.vflip);
    p += sprintf(p, "\"dcw\":%u,", s->status.dcw);
    p += sprintf(p, "\"colorbar\":%u,", s->status.colorbar);
    p += sprintf(p, "\"face_detect\":%u,", detection_enabled);
    p += sprintf(p, "\"face_enroll\":%u,", is_enrolling);
    p += sprintf(p, "\"face_recognize\":%u,", recognition_enabled);
    p += sprintf(p, "\"faces\":%d,", face_count);
    p += sprintf(p, "\"door_open\":%u,", LockState ? 1 : 0);
    p += sprintf(p, "\"mq2\":%d,", (int)mq2_value);
    p += sprintf(p, "\"smoke\":%s,", smoke_detected ? "true" : "false");
    p += sprintf(p, "\"presence\":%s,", presence_detected ? "true" : "false");
    p += sprintf(p, "\"distance\":%d,", (int)presence_distance);
    p += sprintf(p, "\"fan\":%s,", fan_state ? "true" : "false");
    p += sprintf(p, "\"fan_auto\":%s,", fan_auto_mode ? "true" : "false");
    p += sprintf(p, "\"fan_threshold\":%.1f,", (float)fan_humidity_threshold);
    p += sprintf(p, "\"pump\":%s,", pump_state ? "true" : "false");
    p += sprintf(p, "\"pump_auto\":%s,", pump_auto_mode ? "true" : "false");
    p += sprintf(p, "\"smoke_threshold\":%.1f,", (float)smoke_alarm_threshold);
    if (isnan(dht_temp) || isnan(dht_hum)) p += sprintf(p, "\"temp\":null,\"hum\":null");
    else p += sprintf(p, "\"temp\":%.1f,\"hum\":%.1f", dht_temp, dht_hum);
    *p++ = '}';
    *p = 0;

    httpd_resp_set_type(req, "application/json");
    httpd_resp_set_hdr(req, "Access-Control-Allow-Origin", "*");
    return httpd_resp_send(req, json_response, HTTPD_RESP_USE_STRLEN);
}

static esp_err_t door_status_handler(httpd_req_t *req)
{
    if (require_auth(req) != ESP_OK) {
        return ESP_OK;
    }

    char buffer[32];
    snprintf(buffer, sizeof(buffer), "{\"open\":%s}", LockState ? "true" : "false");
    httpd_resp_set_type(req, "application/json");
    httpd_resp_set_hdr(req, "Access-Control-Allow-Origin", "*");
    return httpd_resp_send(req, buffer, HTTPD_RESP_USE_STRLEN);
}

static esp_err_t reboot_handler(httpd_req_t *req)
{
    if (require_auth(req) != ESP_OK) {
        return ESP_OK;
    }

    httpd_resp_send(req, "{\"status\":\"rebooting\"}", HTTPD_RESP_USE_STRLEN);
    vTaskDelay(pdMS_TO_TICKS(1000));
    esp_restart();
    return ESP_OK;
}

static esp_err_t xclk_handler(httpd_req_t *req)
{
    if (require_auth(req) != ESP_OK) {
        return ESP_OK;
    }

    char *buf = nullptr; char xclk_str[16] = "";
    if (parse_get(req, &buf) != ESP_OK) return ESP_FAIL;
    if (httpd_query_key_value(buf, "xclk", xclk_str, sizeof(xclk_str)) != ESP_OK) {
        free(buf); return httpd_resp_send_404(req);
    }
    sensor_t *s = esp_camera_sensor_get(); int xclk = atoi(xclk_str); free(buf);
    if (!s || s->set_xclk(s, LEDC_TIMER_0, xclk) != 0) return httpd_resp_send_500(req);
    httpd_resp_set_hdr(req, "Access-Control-Allow-Origin", "*");
    return httpd_resp_send(req, nullptr, 0);
}

static esp_err_t reg_handler(httpd_req_t *req)
{
    if (require_auth(req) != ESP_OK) return ESP_OK;
    char *buf = nullptr; char reg_str[16] = ""; char mask_str[16] = ""; char val_str[16] = "";
    if (parse_get(req, &buf) != ESP_OK) return ESP_FAIL;
    if (httpd_query_key_value(buf, "reg", reg_str, sizeof(reg_str)) != ESP_OK ||
        httpd_query_key_value(buf, "mask", mask_str, sizeof(mask_str)) != ESP_OK ||
        httpd_query_key_value(buf, "val", val_str, sizeof(val_str)) != ESP_OK) { free(buf); return httpd_resp_send_404(req); }
    sensor_t *s = esp_camera_sensor_get(); int reg = atoi(reg_str); int mask = atoi(mask_str); int val = atoi(val_str); free(buf);
    if (!s || s->set_reg(s, reg, mask, val) != 0) return httpd_resp_send_500(req);
    httpd_resp_set_hdr(req, "Access-Control-Allow-Origin", "*");
    return httpd_resp_send(req, nullptr, 0);
}

static esp_err_t greg_handler(httpd_req_t *req)
{
    if (require_auth(req) != ESP_OK) return ESP_OK;
    char *buf = nullptr; char reg_str[16] = ""; char mask_str[16] = "";
    if (parse_get(req, &buf) != ESP_OK) return ESP_FAIL;
    if (httpd_query_key_value(buf, "reg", reg_str, sizeof(reg_str)) != ESP_OK ||
        httpd_query_key_value(buf, "mask", mask_str, sizeof(mask_str)) != ESP_OK) { free(buf); return httpd_resp_send_404(req); }
    sensor_t *s = esp_camera_sensor_get(); int reg = atoi(reg_str); int mask = atoi(mask_str); free(buf);
    if (!s) return httpd_resp_send_500(req);
    int value = s->get_reg(s, reg, mask); if (value < 0) return httpd_resp_send_500(req);
    char response[16]; snprintf(response, sizeof(response), "%d", value);
    httpd_resp_set_hdr(req, "Access-Control-Allow-Origin", "*");
    return httpd_resp_send(req, response, HTTPD_RESP_USE_STRLEN);
}

static int parse_get_var(char *buf, const char *key, int def)
{
    char value[16] = "";
    if (httpd_query_key_value(buf, key, value, sizeof(value)) != ESP_OK) return def;
    return atoi(value);
}

static esp_err_t pll_handler(httpd_req_t *req)
{
    if (require_auth(req) != ESP_OK) return ESP_OK;
    char *buf = nullptr; if (parse_get(req, &buf) != ESP_OK) return ESP_FAIL;
    int bypass = parse_get_var(buf, "bypass", 0); int mul = parse_get_var(buf, "mul", 0); int sys = parse_get_var(buf, "sys", 0); int root = parse_get_var(buf, "root", 0); int pre = parse_get_var(buf, "pre", 0); int seld5 = parse_get_var(buf, "seld5", 0); int pclken = parse_get_var(buf, "pclken", 0); int pclk = parse_get_var(buf, "pclk", 0); free(buf);
    sensor_t *s = esp_camera_sensor_get(); if (!s || s->set_pll(s, bypass, mul, sys, root, pre, seld5, pclken, pclk) != 0) return httpd_resp_send_500(req);
    httpd_resp_set_hdr(req, "Access-Control-Allow-Origin", "*");
    return httpd_resp_send(req, nullptr, 0);
}

static esp_err_t win_handler(httpd_req_t *req)
{
    if (require_auth(req) != ESP_OK) return ESP_OK;
    char *buf = nullptr; if (parse_get(req, &buf) != ESP_OK) return ESP_FAIL;
    int startX = parse_get_var(buf, "sx", 0); int startY = parse_get_var(buf, "sy", 0); int endX = parse_get_var(buf, "ex", 0); int endY = parse_get_var(buf, "ey", 0); int offsetX = parse_get_var(buf, "offx", 0); int offsetY = parse_get_var(buf, "offy", 0); int totalX = parse_get_var(buf, "tx", 0); int totalY = parse_get_var(buf, "ty", 0); int outputX = parse_get_var(buf, "ox", 0); int outputY = parse_get_var(buf, "oy", 0); bool scale = parse_get_var(buf, "scale", 0) == 1; bool binning = parse_get_var(buf, "binning", 0) == 1; free(buf);
    sensor_t *s = esp_camera_sensor_get(); if (!s || s->set_res_raw(s, startX, startY, endX, endY, offsetX, offsetY, totalX, totalY, outputX, outputY, scale, binning) != 0) return httpd_resp_send_500(req);
    httpd_resp_set_hdr(req, "Access-Control-Allow-Origin", "*");
    return httpd_resp_send(req, nullptr, 0);
}

// ==================== SERVER START ====================
extern "C" void startCameraServer()
{
    // prefs already opened in setup(), do not call prefs.begin() again
    ensure_recognizer();
    ensure_ai_background();
    sync_face_raw_map();

    httpd_config_t config = HTTPD_DEFAULT_CONFIG();
    config.max_uri_handlers = 50;
    config.stack_size = 16384;
    config.max_open_sockets = 4;
    config.lru_purge_enable = true;
    config.recv_wait_timeout = 10;
    config.send_wait_timeout = 10;

    httpd_handle_t main_server = nullptr;
    httpd_handle_t stream_server = nullptr;

    httpd_uri_t index_uri = { .uri = "/", .method = HTTP_GET, .handler = index_handler, .user_ctx = nullptr };
    httpd_uri_t login_uri = { .uri = "/login", .method = HTTP_GET, .handler = login_handler, .user_ctx = nullptr };
    httpd_uri_t do_login_uri = { .uri = "/do_login", .method = HTTP_GET, .handler = do_login_handler, .user_ctx = nullptr };
    httpd_uri_t verify_uri = { .uri = "/verify_2fa", .method = HTTP_GET, .handler = verify_2fa_handler, .user_ctx = nullptr };
    httpd_uri_t do_verify_uri = { .uri = "/do_verify_2fa", .method = HTTP_GET, .handler = do_verify_2fa_handler, .user_ctx = nullptr };
    httpd_uri_t forgot_uri = { .uri = "/forgot_password", .method = HTTP_GET, .handler = forgot_password_handler, .user_ctx = nullptr };
    httpd_uri_t reset_password_uri = { .uri = "/reset_password", .method = HTTP_GET, .handler = reset_password_handler, .user_ctx = nullptr };
    httpd_uri_t logout_uri = { .uri = "/logout", .method = HTTP_GET, .handler = logout_handler, .user_ctx = nullptr };
    httpd_uri_t stream_uri = { .uri = "/stream", .method = HTTP_GET, .handler = stream_handler, .user_ctx = nullptr };
    httpd_uri_t enroll_uri = { .uri = "/enroll", .method = HTTP_GET, .handler = enroll_face_handler, .user_ctx = nullptr };
    httpd_uri_t enroll_phone_uri = { .uri = "/enroll_phone", .method = HTTP_POST, .handler = enroll_phone_handler_v2, .user_ctx = nullptr };
    httpd_uri_t setname_uri = { .uri = "/setname", .method = HTTP_GET, .handler = setname_handler, .user_ctx = nullptr };
    httpd_uri_t arm_rfid_link_uri = { .uri = "/arm_rfid_link", .method = HTTP_GET, .handler = arm_rfid_link_handler, .user_ctx = nullptr };
    httpd_uri_t cancel_rfid_link_uri = { .uri = "/cancel_rfid_link", .method = HTTP_GET, .handler = cancel_rfid_link_handler, .user_ctx = nullptr };
    httpd_uri_t set_rfid_manual_uri = { .uri = "/set_rfid_manual", .method = HTTP_GET, .handler = set_rfid_manual_handler, .user_ctx = nullptr };
    httpd_uri_t rfid_state_uri = { .uri = "/rfid_state", .method = HTTP_GET, .handler = rfid_state_handler, .user_ctx = nullptr };
    httpd_uri_t delete_face_uri = { .uri = "/delete_face", .method = HTTP_GET, .handler = delete_face_handler, .user_ctx = nullptr };
    httpd_uri_t unlock_face_uri = { .uri = "/unlock_face", .method = HTTP_GET, .handler = unlock_face_handler, .user_ctx = nullptr };
    httpd_uri_t delete_all_uri = { .uri = "/delete_all", .method = HTTP_GET, .handler = delete_all_handler, .user_ctx = nullptr };
    httpd_uri_t get_faces_uri = { .uri = "/get_faces", .method = HTTP_GET, .handler = get_faces_handler, .user_ctx = nullptr };
    httpd_uri_t control_uri = { .uri = "/control", .method = HTTP_GET, .handler = cmd_handler, .user_ctx = nullptr };
    httpd_uri_t dht_uri = { .uri = "/dht", .method = HTTP_GET, .handler = dht_handler, .user_ctx = nullptr };
    httpd_uri_t events_uri = { .uri = "/events", .method = HTTP_GET, .handler = events_handler, .user_ctx = nullptr };
    httpd_uri_t capture_uri = { .uri = "/capture", .method = HTTP_GET, .handler = capture_handler, .user_ctx = nullptr };
    httpd_uri_t bmp_uri = { .uri = "/bmp", .method = HTTP_GET, .handler = bmp_handler, .user_ctx = nullptr };
    httpd_uri_t status_uri = { .uri = "/status", .method = HTTP_GET, .handler = status_handler, .user_ctx = nullptr };
    httpd_uri_t sync_time_uri = { .uri = "/sync_time", .method = HTTP_GET, .handler = sync_time_handler, .user_ctx = nullptr };
    httpd_uri_t time_status_uri = { .uri = "/time_status", .method = HTTP_GET, .handler = time_status_handler, .user_ctx = nullptr };
    httpd_uri_t logs_uri = { .uri = "/logs", .method = HTTP_GET, .handler = logs_handler, .user_ctx = nullptr };
    httpd_uri_t clear_logs_uri = { .uri = "/clear_logs", .method = HTTP_GET, .handler = clear_logs_handler, .user_ctx = nullptr };
    httpd_uri_t setpass_uri = { .uri = "/setpass", .method = HTTP_GET, .handler = setpass_handler, .user_ctx = nullptr };
    httpd_uri_t reboot_uri = { .uri = "/reboot", .method = HTTP_GET, .handler = reboot_handler, .user_ctx = nullptr };
    httpd_uri_t xclk_uri = { .uri = "/xclk", .method = HTTP_GET, .handler = xclk_handler, .user_ctx = nullptr };
    httpd_uri_t reg_uri = { .uri = "/reg", .method = HTTP_GET, .handler = reg_handler, .user_ctx = nullptr };
    httpd_uri_t greg_uri = { .uri = "/greg", .method = HTTP_GET, .handler = greg_handler, .user_ctx = nullptr };
    httpd_uri_t pll_uri = { .uri = "/pll", .method = HTTP_GET, .handler = pll_handler, .user_ctx = nullptr };
    httpd_uri_t win_uri = { .uri = "/resolution", .method = HTTP_GET, .handler = win_handler, .user_ctx = nullptr };
    httpd_uri_t door_status_uri = { .uri = "/door_status", .method = HTTP_GET, .handler = door_status_handler, .user_ctx = nullptr };
    httpd_uri_t sensors_uri = { .uri = "/sensors", .method = HTTP_GET, .handler = sensors_handler, .user_ctx = nullptr };
    httpd_uri_t ld2410_debug_uri = { .uri = "/ld2410_debug", .method = HTTP_GET, .handler = ld2410_debug_handler, .user_ctx = nullptr };
    httpd_uri_t ld2410_refresh_uri = { .uri = "/ld2410_refresh", .method = HTTP_GET, .handler = ld2410_refresh_handler, .user_ctx = nullptr };
    httpd_uri_t ld2410_set_sens_uri = { .uri = "/ld2410_set_sensitivity", .method = HTTP_GET, .handler = ld2410_set_sensitivity_handler, .user_ctx = nullptr };
    httpd_uri_t occupants_uri = { .uri = "/occupants", .method = HTTP_GET, .handler = occupants_handler, .user_ctx = nullptr };
    httpd_uri_t fan_uri = { .uri = "/fan_control", .method = HTTP_GET, .handler = fan_control_handler, .user_ctx = nullptr };
    httpd_uri_t pump_uri = { .uri = "/pump_control", .method = HTTP_GET, .handler = pump_control_handler, .user_ctx = nullptr };

    if (httpd_start(&main_server, &config) == ESP_OK) {
        httpd_register_uri_handler(main_server, &index_uri);
        httpd_register_uri_handler(main_server, &login_uri);
        httpd_register_uri_handler(main_server, &do_login_uri);
        httpd_register_uri_handler(main_server, &verify_uri);
        httpd_register_uri_handler(main_server, &do_verify_uri);
        httpd_register_uri_handler(main_server, &forgot_uri);
        httpd_register_uri_handler(main_server, &reset_password_uri);
        httpd_register_uri_handler(main_server, &logout_uri);
        httpd_register_uri_handler(main_server, &enroll_uri);
        httpd_register_uri_handler(main_server, &enroll_phone_uri);
        httpd_register_uri_handler(main_server, &setname_uri);
        httpd_register_uri_handler(main_server, &arm_rfid_link_uri);
        httpd_register_uri_handler(main_server, &cancel_rfid_link_uri);
        httpd_register_uri_handler(main_server, &set_rfid_manual_uri);
        httpd_register_uri_handler(main_server, &rfid_state_uri);
        httpd_register_uri_handler(main_server, &delete_face_uri);
        httpd_register_uri_handler(main_server, &unlock_face_uri);
        httpd_register_uri_handler(main_server, &delete_all_uri);
        httpd_register_uri_handler(main_server, &get_faces_uri);
        httpd_register_uri_handler(main_server, &control_uri);
        httpd_register_uri_handler(main_server, &dht_uri);
        httpd_register_uri_handler(main_server, &events_uri);
        httpd_register_uri_handler(main_server, &capture_uri);
        httpd_register_uri_handler(main_server, &bmp_uri);
        httpd_register_uri_handler(main_server, &status_uri);
        httpd_register_uri_handler(main_server, &sync_time_uri);
        httpd_register_uri_handler(main_server, &time_status_uri);
        httpd_register_uri_handler(main_server, &logs_uri);
        httpd_register_uri_handler(main_server, &clear_logs_uri);
        httpd_register_uri_handler(main_server, &setpass_uri);
        httpd_register_uri_handler(main_server, &reboot_uri);
        httpd_register_uri_handler(main_server, &xclk_uri);
        httpd_register_uri_handler(main_server, &reg_uri);
        httpd_register_uri_handler(main_server, &greg_uri);
        httpd_register_uri_handler(main_server, &pll_uri);
        httpd_register_uri_handler(main_server, &win_uri);
        httpd_register_uri_handler(main_server, &door_status_uri);
        httpd_register_uri_handler(main_server, &sensors_uri);
        httpd_register_uri_handler(main_server, &ld2410_debug_uri);
        httpd_register_uri_handler(main_server, &ld2410_refresh_uri);
        httpd_register_uri_handler(main_server, &ld2410_set_sens_uri);
        httpd_register_uri_handler(main_server, &occupants_uri);
        httpd_register_uri_handler(main_server, &fan_uri);
        httpd_register_uri_handler(main_server, &pump_uri);
        ESP_LOGI(TAG, "Main server started on port 80");
    }

    config.server_port = 81;
    config.ctrl_port += 1;
    config.max_open_sockets = 3;
    if (httpd_start(&stream_server, &config) == ESP_OK) {
        httpd_register_uri_handler(stream_server, &stream_uri);
        ESP_LOGI(TAG, "Stream server started on port 81");
    }
}






