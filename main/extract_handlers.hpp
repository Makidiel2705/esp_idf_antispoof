static esp_err_t events_handler(httpd_req_t *req) {
    // trả về JSON array, older -> newer
    httpd_resp_set_type(req, "application/json");
    httpd_resp_set_hdr(req, "Access-Control-Allow-Origin", "*");

    // gửi chunk để tránh buffer overflow
    httpd_resp_send_chunk(req, "[", 1);
    for (int i = 0; i < events_count; ++i) {
        int idx = (events_next - events_count + i);
        while (idx < 0) idx += MAX_EVENTS;
        idx %= MAX_EVENTS;

        char small[256];
        const char* method_str = "UNKNOWN";
        switch(events[idx].method) {
            case DM_FACE: method_str = "face"; break;
            case DM_KEYPAD: method_str = "keypad"; break;
            case DM_RFID: method_str = "rfid"; break;
            case DM_BUTTON: method_str = "button"; break;
        }
        // actor JSON escaped minimally (assume simple ascii ids). If you expect quotes, escape them.
        int n = snprintf(small, sizeof(small),
            "%s{\"method\":\"%s\",\"time\":%u,\"actor\":\"%s\"}",
            (i==0)?"":",",
            method_str,
            events[idx].ts,
            events[idx].actor);
        if (n > 0) httpd_resp_send_chunk(req, small, n);
    }
    httpd_resp_send_chunk(req, "]", 1);
    return httpd_resp_send_chunk(req, NULL, 0); // finish
}
// --- END: Event log system ---

// --- DHT11 JSON endpoint ---
static esp_err_t dht_handler(httpd_req_t *req) {
    httpd_resp_set_type(req, "application/json");
    httpd_resp_set_hdr(req, "Access-Control-Allow-Origin", "*");
    char buf[64];
    int n;
    if (isnan(dht_temp) || isnan(dht_hum)) {
        n = snprintf(buf, sizeof(buf), "{\"temp\":null,\"hum\":null}");
    } else {
        n = snprintf(buf, sizeof(buf), "{\"temp\":%.1f,\"hum\":%.1f}", dht_temp, dht_hum);
    }
    return httpd_resp_send(req, buf, n);
}
// --- END: DHT11 endpoint ---


typedef struct
{
    httpd_req_t *req;
    size_t len;
} jpg_chunking_t;

#define PART_BOUNDARY "123456789000000000000987654321"
static const char *_STREAM_CONTENT_TYPE = "multipart/x-mixed-replace;boundary=" PART_BOUNDARY;
static const char *_STREAM_BOUNDARY = "\r\n--" PART_BOUNDARY "\r\n";
static const char *_STREAM_PART = "Content-Type: image/jpeg\r\nContent-Length: %u\r\nX-Timestamp: %d.%06d\r\n\r\n";

httpd_handle_t stream_httpd = NULL;
httpd_handle_t camera_httpd = NULL;

#if CONFIG_ESP_FACE_DETECT_ENABLED

static int8_t detection_enabled = 1;

// #if TWO_STAGE
// static HumanFaceDetectMSR01 s1(0.1F, 0.5F, 10, 0.2F);
// static HumanFaceDetectMNP01 s2(0.5F, 0.3F, 5);
// #else
// static HumanFaceDetectMSR01 s1(0.3F, 0.5F, 10, 0.2F);
// #endif

#if CONFIG_ESP_FACE_RECOGNITION_ENABLED
static int8_t recognition_enabled = 1;
static int8_t is_enrolling = 0;
static String pending_enroll_name = "";

#if QUANT_TYPE
    // S16 model
    FaceRecognition112V1S16 recognizer;
#else
    // S8 model
    FaceRecognition112V1S8 recognizer;
#endif
#endif

#endif

typedef struct
{
    size_t size;  //number of values used for filtering
    size_t index; //current value index
    size_t count; //value count
    int sum;
    int *values; //array to be filled with values
} ra_filter_t;

static ra_filter_t ra_filter;

static ra_filter_t *ra_filter_init(ra_filter_t *filter, size_t sample_size)
{
    memset(filter, 0, sizeof(ra_filter_t));

    filter->values = (int *)malloc(sample_size * sizeof(int));
    if (!filter->values)
    {
        return NULL;
    }
    memset(filter->values, 0, sample_size * sizeof(int));

    filter->size = sample_size;
    return filter;
}

#if ARDUHAL_LOG_LEVEL >= ARDUHAL_LOG_LEVEL_INFO
static int ra_filter_run(ra_filter_t *filter, int value)
{
    if (!filter->values)
    {
        return value;
    }
    filter->sum -= filter->values[filter->index];
    filter->values[filter->index] = value;
    filter->sum += filter->values[filter->index];
    filter->index++;
    filter->index = filter->index % filter->size;
    if (filter->count < filter->size)
    {
        filter->count++;
    }
    return filter->sum / filter->count;
}
#endif

#if CONFIG_ESP_FACE_DETECT_ENABLED
#if CONFIG_ESP_FACE_RECOGNITION_ENABLED
static void rgb_print(fb_data_t *fb, uint32_t color, const char *str)
{
    fb_gfx_print(fb, (fb->width - (strlen(str) * 14)) / 2, 10, color, str);
}

static int rgb_printf(fb_data_t *fb, uint32_t color, const char *format, ...)
{
    char loc_buf[64];
    char *temp = loc_buf;
    int len;
    va_list arg;
    va_list copy;
    va_start(arg, format);
    va_copy(copy, arg);
    len = vsnprintf(loc_buf, sizeof(loc_buf), format, arg);
    va_end(copy);
    if (len >= sizeof(loc_buf))
    {
        temp = (char *)malloc(len + 1);
        if (temp == NULL)
        {
            return 0;
        }
    }
    vsnprintf(temp, len + 1, format, arg);
    va_end(arg);
    rgb_print(fb, color, temp);
    if (len > 64)
    {
        free(temp);
    }
    return len;
}
#endif
static void draw_face_boxes(fb_data_t *fb, std::list<dl::detect::result_t> *results, int face_id)
{
    int x, y, w, h;
    uint32_t color = FACE_COLOR_YELLOW;
    if (face_id < 0)
    {
        color = FACE_COLOR_RED;
    }
    else if (face_id > 0)
    {
        color = FACE_COLOR_GREEN;
    }
    if(fb->bytes_per_pixel == 2){
        //color = ((color >> 8) & 0xF800) | ((color >> 3) & 0x07E0) | (color & 0x001F);
        color = ((color >> 16) & 0x001F) | ((color >> 3) & 0x07E0) | ((color << 8) & 0xF800);
    }
    int i = 0;
    for (std::list<dl::detect::result_t>::iterator prediction = results->begin(); prediction != results->end(); prediction++, i++)
    {
        // rectangle box
        x = (int)prediction->box[0];
        y = (int)prediction->box[1];
        w = (int)prediction->box[2] - x + 1;
        h = (int)prediction->box[3] - y + 1;
        if((x + w) > fb->width){
            w = fb->width - x;
        }
        if((y + h) > fb->height){
            h = fb->height - y;
        }
        fb_gfx_drawFastHLine(fb, x, y, w, color);
        fb_gfx_drawFastHLine(fb, x, y + h - 1, w, color);
        fb_gfx_drawFastVLine(fb, x, y, h, color);
        fb_gfx_drawFastVLine(fb, x + w - 1, y, h, color);
#if TWO_STAGE
        // landmarks (left eye, mouth left, nose, right eye, mouth right)
        int x0, y0, j;
        for (j = 0; j < 10; j+=2) {
            x0 = (int)prediction->keypoint[j];
            y0 = (int)prediction->keypoint[j+1];
            fb_gfx_fillRect(fb, x0, y0, 3, 3, color);
        }
#endif
    }
}

#if CONFIG_ESP_FACE_RECOGNITION_ENABLED
// =====================================================================
// LƯU TRỮ KHUÔN MẶT QUA LITTLEFS (giải pháp thay thế partition fr)
// Tại sao: write_ids_to_flash() trong thư viện không hoạt động vì
// embedding được lưu trong PSRAM, còn esp_partition_write() không thể
// đọc từ PSRAM làm nguồn dữ liệu.
// =====================================================================

// Lưu embedding của một khuôn mặt xuống LittleFS
bool save_face_to_littlefs(int face_id) {
    if (!LittleFS.begin(false)) {
        Serial.println("[FACE-FS] LittleFS mount failed!");
        return false;
    }
    
    // Tạo thư mục nếu chưa có
    if (!LittleFS.exists("/faces")) {
        LittleFS.mkdir("/faces");
    }
    
    // Lấy embedding từ recognizer
    Tensor<float>& emb = recognizer.get_face_emb(face_id);
    int emb_size = emb.get_size(); // Số phần tử float
    
    // Copy từ PSRAM vào DRAM stack buffer để ghi file
    // (LittleFS có thể ghi từ PSRAM, khác với esp_partition_write)
    String filename = "/faces/" + String(face_id) + ".bin";
    File f = LittleFS.open(filename, "w");
    if (!f) {
        Serial.printf("[FACE-FS] Cannot open %s for write\n", filename.c_str());
        return false;
    }
    
    // Ghi emb_size (4 bytes) để biết khi đọc lại
    f.write((uint8_t*)&emb_size, sizeof(emb_size));
    // Ghi dữ liệu embedding (emb_size * 4 bytes)
    f.write((uint8_t*)emb.element, emb_size * sizeof(float));
    f.close();
    
    Serial.printf("[FACE-FS] Saved face ID %d: %d floats to %s\n", face_id, emb_size, filename.c_str());
    return true;
}

// Xóa file embedding của một khuôn mặt
void delete_face_from_littlefs(int face_id) {
    if (!LittleFS.begin(false)) return;
    String filename = "/faces/" + String(face_id) + ".bin";
    if (LittleFS.exists(filename)) {
        LittleFS.remove(filename);
        Serial.printf("[FACE-FS] Deleted %s\n", filename.c_str());
    }
}

// Nạp tất cả khuôn mặt từ LittleFS vào RAM khi khởi động
int load_faces_from_littlefs() {
    if (!LittleFS.begin(false)) {
        Serial.println("[FACE-FS] LittleFS mount failed on load!");
        return 0;
    }
    
    if (!LittleFS.exists("/faces")) {
        Serial.println("[FACE-FS] /faces directory not found, starting fresh.");
        return 0;
    }
    
    int loaded_count = 0;
    // Lặp qua các ID có tên trong Preferences
    for (int i = 0; i < FACE_ID_SAVE_NUMBER; i++) {
        String key = "id_" + String(i);
        if (!prefs.isKey(key.c_str())) continue; // Không có tên ⇒ bỏ qua
        
        String filename = "/faces/" + String(i) + ".bin";
        if (!LittleFS.exists(filename)) {
            Serial.printf("[FACE-FS] No embedding file for ID %d (%s)\n", i, key.c_str());
            continue;
        }
        
        File f = LittleFS.open(filename, "r");
        if (!f) continue;
        
        // Đọc emb_size
        int emb_size = 0;
        f.read((uint8_t*)&emb_size, sizeof(emb_size));
        if (emb_size <= 0 || emb_size > 1024) { // Sanity check
            Serial.printf("[FACE-FS] Invalid emb_size %d for ID %d\n", emb_size, i);
            f.close();
            continue;
        }
        
        // Đọc dữ liệu embedding vào DRAM buffer
        float* emb_buf = (float*)malloc(emb_size * sizeof(float));
        if (!emb_buf) {
            Serial.println("[FACE-FS] malloc failed for emb_buf!");
            f.close();
            continue;
        }
        f.read((uint8_t*)emb_buf, emb_size * sizeof(float));
        f.close();
        
        // Tạo Tensor<float> từ DRAM buffer và enroll vào recognizer
        Tensor<float> emb_tensor;
        emb_tensor.set_shape({emb_size})
                  .set_element(emb_buf, false) // false = không auto_free
                  .set_auto_free(false);
        
        String name = prefs.getString(key.c_str(), "Unknown");
        int enrolled_id = recognizer.enroll_id(emb_tensor, name.c_str(), false);
        free(emb_buf);
        
        Serial.printf("[FACE-FS] Loaded ID %d ('%s') -> enrolled as %d\n", i, name.c_str(), enrolled_id);
        loaded_count++;
    }
    
    Serial.printf("[FACE-FS] Total loaded: %d faces\n", loaded_count);
    return loaded_count;
}

// Xóa tất cả file embedding
void clear_all_face_files() {
    if (!LittleFS.begin(false)) return;
    if (!LittleFS.exists("/faces")) return;
    File dir = LittleFS.open("/faces");
    File entry = dir.openNextFile();
    while (entry) {
        String path = "/faces/" + String(entry.name());
        entry.close();
        LittleFS.remove(path);
        entry = dir.openNextFile();
    }
    dir.close();
    Serial.println("[FACE-FS] All face files deleted.");
}

// Hàm trợ giúp để lấy ID tiếp theo
static int get_next_id() {
    int max_id = prefs.getInt("max_id", -1);
    return max_id + 1;
}

static int run_face_recognition(fb_data_t *fb, std::list<dl::detect::result_t> *results)
{
    std::vector<int> landmarks = results->front().keypoint;
    int id = -1;

    Tensor<uint8_t> tensor;
    tensor.set_element((uint8_t *)fb->data).set_shape({fb->height, fb->width, 3}).set_auto_free(false);

    int enrolled_count = recognizer.get_enrolled_id_num();

    if (enrolled_count < FACE_ID_SAVE_NUMBER && is_enrolling){
        // Enroll vào RAM
        id = recognizer.enroll_id(tensor, landmarks, "", false);
        
        // LƯU XUỐNG LITTLEFS (thay thế partition fr bị hỏng)
        bool saved = save_face_to_littlefs(id);
        Serial.printf("[FACE] Enrolled ID: %d | RAM count: %d | LittleFS save: %s\n",
                      id, recognizer.get_enrolled_id_num(), saved ? "OK" : "FAIL");
        
        // Cập nhật max_id để tránh ID trùng sau reboot
        if (id > prefs.getInt("max_id", -1)) {
            prefs.putInt("max_id", id);
        }
        
        // Auto-save Name & Time to NVS
        time_t now; struct tm timeinfo;
        time(&now); localtime_r(&now, &timeinfo);
        char timeStr[64];
        strftime(timeStr, sizeof(timeStr), "%d/%m/%Y %H:%M", &timeinfo);

        String nameToSave = (pending_enroll_name != "") ? pending_enroll_name : ("Khuon mat " + String(id));
        String key = "id_" + String(id);
        prefs.putString(key.c_str(), nameToSave);
        String timeKey = "time_" + String(id);
        prefs.putString(timeKey.c_str(), timeStr);

        rgb_printf(fb, FACE_COLOR_CYAN, "ID[%u]", id);
        is_enrolling = 0;
        pending_enroll_name = "";
    }

    face_info_t recognize = recognizer.recognize(tensor, landmarks);
    if(recognize.id >= 0){
        // 1. Lấy tên từ bộ nhớ
        String key = "id_" + String(recognize.id);
        String savedName = prefs.getString(key.c_str(), ""); // Lấy tên, nếu ko có trả về rỗng
        
        if (savedName != "") {
             // Nếu có tên, hiển thị tên lên màn hình camera
             rgb_printf(fb, FACE_COLOR_GREEN, "%s: %.2f", savedName.c_str(), recognize.similarity);
        } else {
             // Nếu chưa đặt tên, hiển thị ID như cũ
             rgb_printf(fb, FACE_COLOR_GREEN, "ID[%u]: %.2f", recognize.id, recognize.similarity);
        }
        // ------------------------------------------------

        // log event 
        static int last_face_id = -1;
        static int64_t last_face_log_time = 0;
        int64_t now_us = esp_timer_get_time();
        
        // Debounce 3s cho cùng 1 người, nhưng nếu có người lạ/khác thì log luôn
        if (recognize.id != last_face_id || (now_us - last_face_log_time) > 3000000LL) {
            char actor[32]; // Tăng kích thước buffer để chứa tên dài
            
            if (savedName != "") {
                // Nếu có tên, ghi tên vào log
                snprintf(actor, sizeof(actor), "%s", savedName.c_str());
            } else {
                // Nếu không, ghi ID
                snprintf(actor, sizeof(actor), "ID %u", recognize.id);
            }
            // ------------------------
            
            // Note: add_event logic is moved to doorControlTask in main file 
            // We just trigger telegram here and send queue command
            // sendTelegramFlag = true;
            telegramActor = String(actor);

            // Gửi lệnh mở cửa qua Queue tới doorControlTask
            DoorMessage msg;
            msg.cmd = CMD_OPEN_DOOR;
            msg.method = DM_FACE;
            strncpy(msg.actor, actor, sizeof(msg.actor) - 1);
            msg.actor[sizeof(msg.actor) - 1] = '\0';
            xQueueSend(doorQueue, &msg, 0); // Không block, nếu queue full thì bỏ qua (vì mặt liên tục dc nhận)

            last_face_id = recognize.id;
            last_face_log_time = now_us;
        }

    } else {
        rgb_print(fb, FACE_COLOR_RED, "Nguoi la!");
    }
    return recognize.id;
}
#endif
#endif

#if CONFIG_LED_ILLUMINATOR_ENABLED
void enable_led(bool en)
{ // Turn LED On or Off
    int duty = en ? led_duty : 0;
    if (en && isStreaming && (led_duty > CONFIG_LED_MAX_INTENSITY))
    {
        duty = CONFIG_LED_MAX_INTENSITY;
    }
    ledcWrite(LED_LEDC_CHANNEL, duty);
    //ledc_set_duty(CONFIG_LED_LEDC_SPEED_MODE, CONFIG_LED_LEDC_CHANNEL, duty);
    //ledc_update_duty(CONFIG_LED_LEDC_SPEED_MODE, CONFIG_LED_LEDC_CHANNEL);
    log_i("Set LED intensity to %d", duty);
}
#endif

static esp_err_t bmp_handler(httpd_req_t *req)
{
    camera_fb_t *fb = NULL;
    esp_err_t res = ESP_OK;
#if ARDUHAL_LOG_LEVEL >= ARDUHAL_LOG_LEVEL_INFO
    uint64_t fr_start = esp_timer_get_time();
#endif
    fb = esp_camera_fb_get();
    if (!fb)
    {
        log_e("Camera capture failed");
        httpd_resp_send_500(req);
        return ESP_FAIL;
    }

    httpd_resp_set_type(req, "image/x-windows-bmp");
    httpd_resp_set_hdr(req, "Content-Disposition", "inline; filename=capture.bmp");
    httpd_resp_set_hdr(req, "Access-Control-Allow-Origin", "*");

    char ts[32];
    snprintf(ts, 32, "%ld.%06ld", fb->timestamp.tv_sec, fb->timestamp.tv_usec);
    httpd_resp_set_hdr(req, "X-Timestamp", (const char *)ts);


    uint8_t * buf = NULL;
    size_t buf_len = 0;
    bool converted = frame2bmp(fb, &buf, &buf_len);
    esp_camera_fb_return(fb);
    if(!converted){
        log_e("BMP Conversion failed");
        httpd_resp_send_500(req);
        return ESP_FAIL;
    }
    res = httpd_resp_send(req, (const char *)buf, buf_len);
    free(buf);
#if ARDUHAL_LOG_LEVEL >= ARDUHAL_LOG_LEVEL_INFO
    uint64_t fr_end = esp_timer_get_time();
#endif
    log_i("BMP: %llums, %uB", (uint64_t)((fr_end - fr_start) / 1000), buf_len);
    return res;
}

static size_t jpg_encode_stream(void *arg, size_t index, const void *data, size_t len)
{
    jpg_chunking_t *j = (jpg_chunking_t *)arg;
    if (!index)
    {
        j->len = 0;
    }
    if (httpd_resp_send_chunk(j->req, (const char *)data, len) != ESP_OK)
    {
        return 0;
    }
    j->len += len;
    return len;
}





static esp_err_t parse_get(httpd_req_t *req, char **obuf)
{
    char *buf = NULL;
    size_t buf_len = 0;

    buf_len = httpd_req_get_url_query_len(req) + 1;
    if (buf_len > 1) {
        buf = (char *)malloc(buf_len);
        if (!buf) {
            httpd_resp_send_500(req);
            return ESP_FAIL;
        }
        if (httpd_req_get_url_query_str(req, buf, buf_len) == ESP_OK) {
            *obuf = buf;
            return ESP_OK;
        }
        free(buf);
    }
    httpd_resp_send_404(req);
    return ESP_FAIL;
}

// Hàm hỗ trợ URL Decode thủ công
String urlDecode(String str) {
    String decoded = "";
    char temp[] = "0x00";
    for (int i = 0; i < str.length(); i++) {
        if (str[i] == '%') {
            if (i + 2 < str.length()) {
                temp[2] = str[i + 1];
                temp[3] = str[i + 2];
                decoded += (char)strtol(temp, NULL, 16);
                i += 2;
            }
        } else if (str[i] == '+') {
            decoded += ' ';
        } else {
            decoded += str[i];
        }
    }
    return decoded;
}

static esp_err_t enroll_face_handler(httpd_req_t *req)
{
    char *buf = NULL;
    char name_str[64] = "";
    if (parse_get(req, &buf) == ESP_OK) {
        if (httpd_query_key_value(buf, "name", name_str, sizeof(name_str)) == ESP_OK) {
            pending_enroll_name = urlDecode(String(name_str));
        } else {
            pending_enroll_name = "";
        }
        free(buf);
    }
    is_enrolling = 1;
    return httpd_resp_sendstr(req, "OK");
}

static esp_err_t setname_handler(httpd_req_t *req)
{
    char *buf = NULL;
    if (parse_get(req, &buf) != ESP_OK) {
        return ESP_FAIL;
    }
    
    char id_str[10];
    char name_str[64];
    
    // Lấy ID và Name từ URL (ví dụ: /setname?id=0&name=Kien)
    if (httpd_query_key_value(buf, "id", id_str, sizeof(id_str)) == ESP_OK &&
        httpd_query_key_value(buf, "name", name_str, sizeof(name_str)) == ESP_OK) {
        
        String key = "id_" + String(id_str); // Tạo khóa, ví dụ: "id_0"
        String decodedName = urlDecode(String(name_str));
        prefs.putString(key.c_str(), decodedName); // Lưu tên vào bộ nhớ
        
        log_i("Map ID %s to Name %s", id_str, decodedName.c_str());
        httpd_resp_sendstr(req, "OK: Da luu ten");
    } else {
        httpd_resp_send_500(req);
    }
    free(buf);
    return ESP_OK;
}

// Trong app_httpd.cpp

static esp_err_t delete_face_handler(httpd_req_t *req)
{
    char *buf = NULL;
    if (parse_get(req, &buf) != ESP_OK) {
        return ESP_FAIL;
    }

    char id_str[10];
    if (httpd_query_key_value(buf, "id", id_str, sizeof(id_str)) != ESP_OK) {
        free(buf);
        httpd_resp_send_500(req);
        return ESP_FAIL;
    }
    free(buf);

    int id_to_delete = atoi(id_str);

    // 1. Xóa khỏi bộ nhận diện AI
    int remaining = recognizer.delete_id(id_to_delete, false);
    
    // 2. Xóa file embedding trong LittleFS
    delete_face_from_littlefs(id_to_delete);
    
    Serial.printf("[FACE] delete_id(%d): %d remaining\n", id_to_delete, remaining);

    if (remaining >= 0) {
        // 2. Xóa tên khỏi bộ nhớ Flash (Preferences)
        String key = "id_" + String(id_to_delete);
        prefs.remove(key.c_str());

        log_i("Deleted Face ID: %d. Remaining: %d", id_to_delete, remaining);
        
        char msg[64];
        snprintf(msg, sizeof(msg), "Da xoa ID %d. Con lai: %d khuon mat.", id_to_delete, remaining);
        httpd_resp_sendstr(req, msg);
    } else {
        httpd_resp_sendstr(req, "Loi: Khong tim thay ID nay!");
    }

    return ESP_OK;
}

static esp_err_t cmd_handler(httpd_req_t *req)
{
    char *buf = NULL;
    char variable[32];
    char value[32];

    if (parse_get(req, &buf) != ESP_OK) {
        return ESP_FAIL;
    }
    if (httpd_query_key_value(buf, "var", variable, sizeof(variable)) != ESP_OK ||
        httpd_query_key_value(buf, "val", value, sizeof(value)) != ESP_OK) {
        free(buf);
        httpd_resp_send_404(req);
        return ESP_FAIL;
    }
    free(buf);

    int val = atoi(value);
    log_i("%s = %d", variable, val);
    sensor_t *s = esp_camera_sensor_get();
    int res = 0;

    if (!strcmp(variable, "framesize")) {
        if (s->pixformat == PIXFORMAT_JPEG) {
            res = s->set_framesize(s, (framesize_t)val);
        }
    }
    else if (!strcmp(variable, "quality"))
        res = s->set_quality(s, val);
    else if (!strcmp(variable, "contrast"))
        res = s->set_contrast(s, val);
    else if (!strcmp(variable, "brightness"))
        res = s->set_brightness(s, val);
    else if (!strcmp(variable, "saturation"))
        res = s->set_saturation(s, val);
    else if (!strcmp(variable, "gainceiling"))
        res = s->set_gainceiling(s, (gainceiling_t)val);
    else if (!strcmp(variable, "colorbar"))
        res = s->set_colorbar(s, val);
    else if (!strcmp(variable, "awb"))
        res = s->set_whitebal(s, val);
    else if (!strcmp(variable, "agc"))
        res = s->set_gain_ctrl(s, val);
    else if (!strcmp(variable, "aec"))
        res = s->set_exposure_ctrl(s, val);
    else if (!strcmp(variable, "hmirror"))
        res = s->set_hmirror(s, val);
    else if (!strcmp(variable, "vflip"))
        res = s->set_vflip(s, val);
    else if (!strcmp(variable, "awb_gain"))
        res = s->set_awb_gain(s, val);
    else if (!strcmp(variable, "agc_gain"))
        res = s->set_agc_gain(s, val);
    else if (!strcmp(variable, "aec_value"))
        res = s->set_aec_value(s, val);
    else if (!strcmp(variable, "aec2"))
        res = s->set_aec2(s, val);
    else if (!strcmp(variable, "dcw"))
        res = s->set_dcw(s, val);
    else if (!strcmp(variable, "bpc"))
        res = s->set_bpc(s, val);
    else if (!strcmp(variable, "wpc"))
        res = s->set_wpc(s, val);
    else if (!strcmp(variable, "raw_gma"))
        res = s->set_raw_gma(s, val);
    else if (!strcmp(variable, "lenc"))
        res = s->set_lenc(s, val);
    else if (!strcmp(variable, "special_effect"))
        res = s->set_special_effect(s, val);
    else if (!strcmp(variable, "wb_mode"))
        res = s->set_wb_mode(s, val);
    else if (!strcmp(variable, "ae_level"))
        res = s->set_ae_level(s, val);
#if CONFIG_LED_ILLUMINATOR_ENABLED
    else if (!strcmp(variable, "led_intensity")) {
        led_duty = val;
        if (isStreaming)
            enable_led(true);
    }
#endif

#if CONFIG_ESP_FACE_DETECT_ENABLED
    else if (!strcmp(variable, "face_detect")) {
        detection_enabled = val;
#if CONFIG_ESP_FACE_RECOGNITION_ENABLED
        if (!detection_enabled) {
            recognition_enabled = 0;
        }
#endif
    }
#if CONFIG_ESP_FACE_RECOGNITION_ENABLED
    else if (!strcmp(variable, "face_enroll")){
        is_enrolling = !is_enrolling;
        log_i("Enrolling: %s", is_enrolling?"true":"false");
    }
    else if (!strcmp(variable, "face_recognize")) {
        recognition_enabled = val;
        if (recognition_enabled) {
            detection_enabled = val;
        }
    }
#endif
#endif
    else {
        log_i("Unknown command: %s", variable);
        res = -1;
    }

    if (res < 0) {
        return httpd_resp_send_500(req);
    }

    httpd_resp_set_hdr(req, "Access-Control-Allow-Origin", "*");
    return httpd_resp_send(req, NULL, 0);
}

static int print_reg(char * p, sensor_t * s, uint16_t reg, uint32_t mask){
    return sprintf(p, "\"0x%x\":%u,", reg, s->get_reg(s, reg, mask));
}

static esp_err_t status_handler(httpd_req_t *req)
{
    static char json_response[1024];

    sensor_t *s = esp_camera_sensor_get();
    char *p = json_response;
    *p++ = '{';

    if(s->id.PID == OV5640_PID || s->id.PID == OV3660_PID){
        for(int reg = 0x3400; reg < 0x3406; reg+=2){
            p+=print_reg(p, s, reg, 0xFFF);//12 bit
        }
        p+=print_reg(p, s, 0x3406, 0xFF);

        p+=print_reg(p, s, 0x3500, 0xFFFF0);//16 bit
        p+=print_reg(p, s, 0x3503, 0xFF);
        p+=print_reg(p, s, 0x350a, 0x3FF);//10 bit
        p+=print_reg(p, s, 0x350c, 0xFFFF);//16 bit

        for(int reg = 0x5480; reg <= 0x5490; reg++){
            p+=print_reg(p, s, reg, 0xFF);
        }

        for(int reg = 0x5380; reg <= 0x538b; reg++){
            p+=print_reg(p, s, reg, 0xFF);
        }

        for(int reg = 0x5580; reg < 0x558a; reg++){
            p+=print_reg(p, s, reg, 0xFF);
        }
        p+=print_reg(p, s, 0x558a, 0x1FF);//9 bit
    } else if(s->id.PID == OV2640_PID){
        p+=print_reg(p, s, 0xd3, 0xFF);
        p+=print_reg(p, s, 0x111, 0xFF);
        p+=print_reg(p, s, 0x132, 0xFF);
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
    p += sprintf(p, "\"dcw\":%u,", s->status.dcw);
    p += sprintf(p, "\"colorbar\":%u", s->status.colorbar);
#if CONFIG_LED_ILLUMINATOR_ENABLED
    p += sprintf(p, ",\"led_intensity\":%u", led_duty);
#else
    p += sprintf(p, ",\"led_intensity\":%d", -1);
#endif
#if CONFIG_ESP_FACE_DETECT_ENABLED
    p += sprintf(p, ",\"face_detect\":%u", detection_enabled);
#if CONFIG_ESP_FACE_RECOGNITION_ENABLED
    p += sprintf(p, ",\"face_enroll\":%u,", is_enrolling);
    p += sprintf(p, "\"face_recognize\":%u", recognition_enabled);
#endif
#endif
    *p++ = '}';
    *p++ = 0;
    httpd_resp_set_type(req, "application/json");
    httpd_resp_set_hdr(req, "Access-Control-Allow-Origin", "*");
    return httpd_resp_send(req, json_response, strlen(json_response));
}

static esp_err_t xclk_handler(httpd_req_t *req)
{
    char *buf = NULL;
    char _xclk[32];

    if (parse_get(req, &buf) != ESP_OK) {
        return ESP_FAIL;
    }
    if (httpd_query_key_value(buf, "xclk", _xclk, sizeof(_xclk)) != ESP_OK) {
        free(buf);
        httpd_resp_send_404(req);
        return ESP_FAIL;
    }
    free(buf);

    int xclk = atoi(_xclk);
    log_i("Set XCLK: %d MHz", xclk);

    sensor_t *s = esp_camera_sensor_get();
    int res = s->set_xclk(s, LEDC_TIMER_0, xclk);
    if (res) {
        return httpd_resp_send_500(req);
    }

    httpd_resp_set_hdr(req, "Access-Control-Allow-Origin", "*");
    return httpd_resp_send(req, NULL, 0);
}

static esp_err_t reg_handler(httpd_req_t *req)
{
    char *buf = NULL;
    char _reg[32];
    char _mask[32];
    char _val[32];

    if (parse_get(req, &buf) != ESP_OK) {
        return ESP_FAIL;
    }
    if (httpd_query_key_value(buf, "reg", _reg, sizeof(_reg)) != ESP_OK ||
        httpd_query_key_value(buf, "mask", _mask, sizeof(_mask)) != ESP_OK ||
        httpd_query_key_value(buf, "val", _val, sizeof(_val)) != ESP_OK) {
        free(buf);
        httpd_resp_send_404(req);
        return ESP_FAIL;
    }
    free(buf);

    int reg = atoi(_reg);
    int mask = atoi(_mask);
    int val = atoi(_val);
    log_i("Set Register: reg: 0x%02x, mask: 0x%02x, value: 0x%02x", reg, mask, val);

    sensor_t *s = esp_camera_sensor_get();
    int res = s->set_reg(s, reg, mask, val);
    if (res) {
        return httpd_resp_send_500(req);
    }

    httpd_resp_set_hdr(req, "Access-Control-Allow-Origin", "*");
    return httpd_resp_send(req, NULL, 0);
}

static esp_err_t greg_handler(httpd_req_t *req)
{
    char *buf = NULL;
    char _reg[32];
    char _mask[32];

    if (parse_get(req, &buf) != ESP_OK) {
        return ESP_FAIL;
    }
    if (httpd_query_key_value(buf, "reg", _reg, sizeof(_reg)) != ESP_OK ||
        httpd_query_key_value(buf, "mask", _mask, sizeof(_mask)) != ESP_OK) {
        free(buf);
        httpd_resp_send_404(req);
        return ESP_FAIL;
    }
    free(buf);

    int reg = atoi(_reg);
    int mask = atoi(_mask);
    sensor_t *s = esp_camera_sensor_get();
    int res = s->get_reg(s, reg, mask);
    if (res < 0) {
        return httpd_resp_send_500(req);
    }
    log_i("Get Register: reg: 0x%02x, mask: 0x%02x, value: 0x%02x", reg, mask, res);

    char buffer[20];
    const char * val = itoa(res, buffer, 10);
    httpd_resp_set_hdr(req, "Access-Control-Allow-Origin", "*");
    return httpd_resp_send(req, val, strlen(val));
}

static int parse_get_var(char *buf, const char * key, int def)
{
    char _int[16];
    if(httpd_query_key_value(buf, key, _int, sizeof(_int)) != ESP_OK){
        return def;
    }
    return atoi(_int);
}

static esp_err_t pll_handler(httpd_req_t *req)
{
    char *buf = NULL;

    if (parse_get(req, &buf) != ESP_OK) {
        return ESP_FAIL;
    }

    int bypass = parse_get_var(buf, "bypass", 0);
    int mul = parse_get_var(buf, "mul", 0);
    int sys = parse_get_var(buf, "sys", 0);
    int root = parse_get_var(buf, "root", 0);
    int pre = parse_get_var(buf, "pre", 0);
    int seld5 = parse_get_var(buf, "seld5", 0);
    int pclken = parse_get_var(buf, "pclken", 0);
    int pclk = parse_get_var(buf, "pclk", 0);
    free(buf);

    log_i("Set Pll: bypass: %d, mul: %d, sys: %d, root: %d, pre: %d, seld5: %d, pclken: %d, pclk: %d", bypass, mul, sys, root, pre, seld5, pclken, pclk);
    sensor_t *s = esp_camera_sensor_get();
    int res = s->set_pll(s, bypass, mul, sys, root, pre, seld5, pclken, pclk);
    if (res) {
        return httpd_resp_send_500(req);
    }

    httpd_resp_set_hdr(req, "Access-Control-Allow-Origin", "*");
    return httpd_resp_send(req, NULL, 0);
}

static esp_err_t win_handler(httpd_req_t *req)
{
    char *buf = NULL;

    if (parse_get(req, &buf) != ESP_OK) {
        return ESP_FAIL;
    }

    int startX = parse_get_var(buf, "sx", 0);
    int startY = parse_get_var(buf, "sy", 0);
    int endX = parse_get_var(buf, "ex", 0);
    int endY = parse_get_var(buf, "ey", 0);
    int offsetX = parse_get_var(buf, "offx", 0);
    int offsetY = parse_get_var(buf, "offy", 0);
    int totalX = parse_get_var(buf, "tx", 0);
    int totalY = parse_get_var(buf, "ty", 0);
    int outputX = parse_get_var(buf, "ox", 0);
    int outputY = parse_get_var(buf, "oy", 0);
    bool scale = parse_get_var(buf, "scale", 0) == 1;
    bool binning = parse_get_var(buf, "binning", 0) == 1;
    free(buf);

    log_i("Set Window: Start: %d %d, End: %d %d, Offset: %d %d, Total: %d %d, Output: %d %d, Scale: %u, Binning: %u", startX, startY, endX, endY, offsetX, offsetY, totalX, totalY, outputX, outputY, scale, binning);
    sensor_t *s = esp_camera_sensor_get();
    int res = s->set_res_raw(s, startX, startY, endX, endY, offsetX, offsetY, totalX, totalY, outputX, outputY, scale, binning);
    if (res) {
        return httpd_resp_send_500(req);
    }

    httpd_resp_set_hdr(req, "Access-Control-Allow-Origin", "*");
    return httpd_resp_send(req, NULL, 0);
}

//self
// ==================== AUTHENTICATION START ====================
static esp_err_t get_faces_handler(httpd_req_t *req) {
    httpd_resp_set_type(req, "application/json");
    String json = "[";
    bool first = true;
    for (int i = 0; i < FACE_ID_SAVE_NUMBER; i++) {
        String key = "id_" + String(i);
        if (prefs.isKey(key.c_str())) {
            if (!first) json += ",";
            String name = prefs.getString(key.c_str(), "Unknown");
            String timeKey = "time_" + String(i);
            String timeAdded = prefs.getString(timeKey.c_str(), "N/A");
            json += "{\"id\":" + String(i) + ",\"name\":\"" + name + "\",\"time\":\"" + timeAdded + "\"}";
            first = false;
        }
    }
    json += "]";
    return httpd_resp_sendstr(req, json.c_str());
}

static esp_err_t delete_all_handler(httpd_req_t *req) {
    // Xóa khỏi RAM
    recognizer.clear_id(false);
    
    // Xóa tất cả file embedding trong LittleFS
    clear_all_face_files();
    
    prefs.clear();
    prefs.putInt("max_id", -1);
    
    LittleFS.remove("/logs.json");
    httpd_resp_sendstr(req, "OK: Da xoa toan bo du lieu. Dang khoi dong lai...");
    vTaskDelay(pdMS_TO_TICKS(500));
    ESP.restart(); // Restart to ensure clean state
    return ESP_OK;
}

static esp_err_t get_logs_handler(httpd_req_t *req) {
    httpd_resp_set_type(req, "application/json");
    httpd_resp_send_chunk(req, "[", 1);
    
    File file = LittleFS.open("/logs.json", "r");
    if (file) {
        char buffer[256];
        while (file.available()) {
            int l = file.readBytes(buffer, sizeof(buffer));
            httpd_resp_send_chunk(req, buffer, l);
        }
        file.close();
    }
    httpd_resp_send_chunk(req, "{\"time\":\"\",\"actor\":\"\"}]", 23); 
    httpd_resp_send_chunk(req, NULL, 0);
    return ESP_OK;
}

static esp_err_t clear_logs_handler(httpd_req_t *req) {
    LittleFS.remove("/logs.json");
    httpd_resp_sendstr(req, "OK");
    return ESP_OK;
}

static bool is_authenticated(httpd_req_t *req) { return true; }
    bool ok = (strstr(cookie, "auth=2") != NULL);
    free(cookie);
    return ok;
}

static bool is_step1_authenticated(httpd_req_t *req) { return true; }
    bool ok = (strstr(cookie, "auth=1") != NULL) || (strstr(cookie, "auth=2") != NULL);
    free(cookie);
    return ok;
}


static esp_err_t login_handler(httpd_req_t *req)
{
    const char* login_page =
        "<!doctype html><html lang='vi'><head>"
        "<meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width, initial-scale=1'>"
        "<title>ESP32 Login</title>"
        "<link href='https://fonts.googleapis.com/css2?family=Inter:wght@400;600&display=swap' rel='stylesheet'>"
        "<style>"
        "body{margin:0;font-family:'Inter',sans-serif;"
        "background:url('https://blog.premagic.com/content/images/2024/08/face-recognition-2048-x-1152-wallpaper-wwabq65usam8cc3s.jpg') no-repeat center center fixed;"
        "background-size:cover;display:flex;align-items:center;justify-content:center;height:100vh;color:#0f172a;}"
        ".overlay{position:absolute;top:0;left:0;width:100%;height:100%;background:rgba(255,255,255,0.35);backdrop-filter:blur(1px);}"
        ".card{position:relative;z-index:1;background:#fff;padding:32px 28px;border-radius:14px;"
        "box-shadow:0 8px 28px rgba(15,23,42,0.15);max-width:360px;width:100%;"
        "border:1px solid rgba(15,23,42,0.06);text-align:center;}"
        "h2{margin:0 0 8px 0;font-size:20px;font-weight:600;}"
        "p.subtitle{margin:0 0 18px 0;font-size:14px;color:#6b7280;}"  /* dòng chữ nhỏ, màu xám */
        "form{display:flex;flex-direction:column;gap:14px}"
        "input{padding:10px 12px;font-size:15px;border-radius:10px;border:1px solid rgba(15,23,42,0.12);}"
        "button{padding:10px 14px;border:none;border-radius:10px;cursor:pointer;font-weight:600;"
        "background:linear-gradient(90deg,#0ea5a4,#06b6d4);color:white;"
        "box-shadow:0 6px 18px rgba(14,165,164,0.18);}"
        "button:hover{opacity:.92}"
        "</style></head>"
        "<body>"
        "<div class='overlay'></div>"
        "<div class='card'>"
        "<h2>Door's lock Setting</h2>"
        "<p class='subtitle'>Trần Chí Kiên - Trần Thành Đạt</p>"   // thêm dòng chữ này
        "<form action='/do_login' method='get'>"
        "<input type='password' name='code' placeholder='Nhập mật khẩu' autofocus required>"
        "<button type='submit'>Đăng Nhập</button>"
        "</form>"
        "</div>"
        "</body></html>";

    httpd_resp_set_type(req, "text/html");
    return httpd_resp_sendstr(req, login_page);
}



static esp_err_t do_login_handler(httpd_req_t *req)
{
    char *buf = NULL;
    if (parse_get(req, &buf) != ESP_OK) {
        httpd_resp_set_status(req, "302 Found");
        httpd_resp_set_hdr(req, "Location", "/login");
        return httpd_resp_send(req, NULL, 0);
    }

    char code[32];
    bool ok = false;
    if (httpd_query_key_value(buf, "code", code, sizeof(code)) == ESP_OK) {
        if (strcmp(code, passkey.c_str()) == 0) {
            ok = true;
        }
    }
    free(buf);

    if (ok) {
        // Generate OTP
        int randomCode = random(100000, 999999);
        currentOTP = String(randomCode);
        Serial.println("OTP Generated: " + currentOTP);
        triggerSendOTP();

        httpd_resp_set_hdr(req, "Set-Cookie", "auth=1; Path=/");
        httpd_resp_set_status(req, "302 Found");
        httpd_resp_set_hdr(req, "Location", "/verify_2fa");
        return httpd_resp_send(req, NULL, 0);
    } else {
        const char* fail_page =
        "<!doctype html><html lang='vi'><head>"
        "<meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width, initial-scale=1'>"
        "<title>Sai mật khẩu</title>"
        "<link href='https://fonts.googleapis.com/css2?family=Inter:wght@400;600&display=swap' rel='stylesheet'>"
        "<style>"
        "body{margin:0;font-family:'Inter',sans-serif;"
        "background:url('https://blog.premagic.com/content/images/2024/08/face-recognition-2048-x-1152-wallpaper-wwabq65usam8cc3s.jpg') no-repeat center center fixed;"
        "background-size:cover;display:flex;align-items:center;justify-content:center;height:100vh;color:#0f172a;}"
        ".overlay{position:absolute;top:0;left:0;width:100%;height:100%;background:rgba(255,255,255,0.35);backdrop-filter:blur(1px);}"
        ".card{position:relative;z-index:1;background:#fff;padding:32px 28px;border-radius:14px;"
        "box-shadow:0 8px 28px rgba(15,23,42,0.15);max-width:360px;width:100%;"
        "border:1px solid rgba(15,23,42,0.06);text-align:center;}"
        "h2{margin:0 0 8px 0;font-size:20px;font-weight:600;color:#dc2626;}" /* đỏ tươi */
        "p.subtitle{margin:0 0 18px 0;font-size:14px;color:#6b7280;}"
        "a.button{display:inline-block;padding:10px 14px;border-radius:10px;font-weight:600;text-decoration:none;"
        "background:linear-gradient(90deg,#0ea5a4,#06b6d4);color:white;"
        "box-shadow:0 6px 18px rgba(14,165,164,0.18);}"
        "a.button:hover{opacity:.92}"
        "</style></head>"
        "<body>"
        "<div class='overlay'></div>"
        "<div class='card'>"
        "<h2>Sai mật khẩu</h2>"
        "<p class='subtitle'>Vui lòng thử lại</p>"
        "<a href='/login' class='button'>Quay lại</a>"
        "</div>"
        "</body></html>";
        httpd_resp_set_type(req, "text/html");
        return httpd_resp_sendstr(req, fail_page);
    }
}

static esp_err_t verify_2fa_handler(httpd_req_t *req)
{
    if (!is_step1_authenticated(req)) {
        httpd_resp_set_status(req, "302 Found");
        httpd_resp_set_hdr(req, "Location", "/login");
        return httpd_resp_send(req, NULL, 0);
    }

    const char* verify_page =
        "<!doctype html><html lang='vi'><head>"
        "<meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width, initial-scale=1'>"
        "<title>Xác thực 2 lớp</title>"
        "<link href='https://fonts.googleapis.com/css2?family=Inter:wght@400;600&display=swap' rel='stylesheet'>"
        "<style>"
        "body{margin:0;font-family:'Inter',sans-serif;"
        "background:url('https://blog.premagic.com/content/images/2024/08/face-recognition-2048-x-1152-wallpaper-wwabq65usam8cc3s.jpg') no-repeat center center fixed;"
        "background-size:cover;display:flex;align-items:center;justify-content:center;height:100vh;color:#0f172a;}"
        ".overlay{position:absolute;top:0;left:0;width:100%;height:100%;background:rgba(255,255,255,0.35);backdrop-filter:blur(1px);}"
        ".card{position:relative;z-index:1;background:#fff;padding:32px 28px;border-radius:14px;"
        "box-shadow:0 8px 28px rgba(15,23,42,0.15);max-width:360px;width:100%;"
        "border:1px solid rgba(15,23,42,0.06);text-align:center;}"
        "h2{margin:0 0 8px 0;font-size:20px;font-weight:600;}"
        "p.subtitle{margin:0 0 18px 0;font-size:14px;color:#6b7280;}"
        "form{display:flex;flex-direction:column;gap:14px}"
        "input{padding:10px 12px;font-size:15px;border-radius:10px;border:1px solid rgba(15,23,42,0.12);text-align:center;letter-spacing:4px;font-weight:bold;}"
        "button{padding:10px 14px;border:none;border-radius:10px;cursor:pointer;font-weight:600;"
        "background:linear-gradient(90deg,#0ea5a4,#06b6d4);color:white;"
        "box-shadow:0 6px 18px rgba(14,165,164,0.18);}"
        "button:hover{opacity:.92}"
        "a{font-size:13px;color:#06b6d4;text-decoration:none;margin-top:10px;display:block;}"
        "</style></head>"
        "<body>"
        "<div class='overlay'></div>"
        "<div class='card'>"
        "<h2>Xác thực 2 lớp</h2>"
        "<p class='subtitle'>Mã OTP đã được gửi tới Gmail của bạn</p>"
        "<form action='/do_verify_2fa' method='get'>"
        "<input type='text' name='otp' placeholder='------' maxlength='6' autofocus required>"
        "<button type='submit'>Xác nhận</button>"
        "</form>"
        "<a href='/login'>Quay lại đăng nhập</a>"
        "</div>"
        "</body></html>";

    httpd_resp_set_type(req, "text/html");
    return httpd_resp_sendstr(req, verify_page);
}

static esp_err_t do_verify_2fa_handler(httpd_req_t *req)
{
    if (!is_step1_authenticated(req)) {
        httpd_resp_set_status(req, "302 Found");
        httpd_resp_set_hdr(req, "Location", "/login");
        return httpd_resp_send(req, NULL, 0);
    }

    char *buf = NULL;
    if (parse_get(req, &buf) != ESP_OK) {
        httpd_resp_set_status(req, "302 Found");
        httpd_resp_set_hdr(req, "Location", "/verify_2fa");
        return httpd_resp_send(req, NULL, 0);
    }

    char otp_input[10];
    bool ok = false;
    if (httpd_query_key_value(buf, "otp", otp_input, sizeof(otp_input)) == ESP_OK) {
        if (strcmp(otp_input, currentOTP.c_str()) == 0) {
            ok = true;
        }
    }
    free(buf);

    if (ok) {
        httpd_resp_set_hdr(req, "Set-Cookie", "auth=2; Path=/");
        httpd_resp_set_status(req, "302 Found");
        httpd_resp_set_hdr(req, "Location", "/");
        return httpd_resp_send(req, NULL, 0);
    } else {
        const char* fail_page =
            "<!doctype html><html lang='vi'><head>"
            "<meta charset='utf-8'><title>Sai mã OTP</title>"
            "<style>body{font-family:sans-serif;display:flex;align-items:center;justify-content:center;height:100vh;text-align:center;}"
            ".card{padding:30px;box-shadow:0 4px 12px rgba(0,0,0,0.1);border-radius:10px;}"
            "h2{color:#dc2626;}"
            "a{text-decoration:none;color:#06b6d4;font-weight:bold;}"
            "</style></head><body>"
            "<div class='card'><h2>Sai mã OTP</h2><p>Mã bạn nhập không chính xác hoặc đã hết hạn.</p>"
            "<a href='/verify_2fa'>Thử lại</a></div></body></html>";
        httpd_resp_set_type(req, "text/html");
        return httpd_resp_sendstr(req, fail_page);
    }
}

// ==================== AUTHENTICATION END ====================
// ---------------- Logout handler (ADD here, next to auth helpers) ----------------
static esp_err_t logout_handler(httpd_req_t *req)
{
    // Xóa cookie auth bằng cách đặt Max-Age=0, sau đó redirect về /login
    httpd_resp_set_hdr(req, "Set-Cookie", "auth=0; Path=/; Max-Age=0");
    httpd_resp_set_status(req, "302 Found");
    httpd_resp_set_hdr(req, "Location", "/login");
    return httpd_resp_send(req, NULL, 0);
}
// -------------------------------------------------------------------------------


static esp_err_t index_handler(httpd_req_t *req)
{
    //self
    if (!is_authenticated(req)) {
        if (is_step1_authenticated(req)) {
            httpd_resp_set_status(req, "302 Found");
            httpd_resp_set_hdr(req, "Location", "/verify_2fa");
            return httpd_resp_send(req, NULL, 0);
        } else {
            httpd_resp_set_status(req, "302 Found");
            httpd_resp_set_hdr(req, "Location", "/login");
            return httpd_resp_send(req, NULL, 0);
        }
    }

    httpd_resp_set_type(req, "text/html");
    httpd_resp_set_hdr(req, "Content-Encoding", "gzip");
    sensor_t *s = esp_camera_sensor_get();
    if (s != NULL) {
        return httpd_resp_send(req, (const char *)index_ov2640_html_gz, index_ov2640_html_gz_len);
    } else {
        log_e("Camera sensor not found");
        return httpd_resp_send_500(req);
    }
}


//self
static esp_err_t setpass_handler(httpd_req_t *req)
{
    char buf[32];
    int len = httpd_req_get_url_query_len(req) + 1;
    if (len > 1) {
        char *qry = (char *)malloc(len);
        if (httpd_req_get_url_query_str(req, qry, len) == ESP_OK) {
            if (httpd_query_key_value(qry, "code", buf, sizeof(buf)) == ESP_OK) {
                Serial.printf("New passcode: %s\n", buf);

                // Gán trực tiếp vào biến passkey
                passkey = String(buf);

                httpd_resp_sendstr(req, "Mật khẩu đã đổi!");
                free(qry);
                return ESP_OK;
            }
        }
        free(qry);
    }
    httpd_resp_sendstr(req, "Lỗi: Không nhận được code");
    return ESP_FAIL;
}


