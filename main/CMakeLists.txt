# Khai báo các file nguồn của dự án
set(srcs 
    "main.cpp" 
    "app_httpd.cpp"
    "project_logic.cpp"
    "antispoof_task.cpp"
    "bitmaps.cpp"
)

# Đăng ký Component
idf_component_register(SRCS ${srcs}
                    INCLUDE_DIRS "."
                    REQUIRES arduino-esp32 esp-dl human_face_detect human_face_recognition littlefs esp_http_server esp-tflite-micro esp32-camera
                             ESP-Mail-Client Adafruit_GFX Adafruit_SSD1306 MFRC522 Adafruit_BusIO Adafruit_Sensor Adafruit_MCP23X17 Adafruit_AHTX0 ld2410 fb_gfx VL53L0X)

# Nhúng mô hình TFLite vào thẳng bộ nhớ Flash để nạp nhanh (Binary Embedding)
# Model dùng để build hiện nằm ở root project (cùng cấp với thư mục main/).
target_add_binary_data(${COMPONENT_LIB} "../antispoof.fullint8.local.tflite" BINARY)

# Lưu ý: Lệnh Nhúng đã được bật!

# Lệnh tạo phân vùng Spiffs dùng lưu trữ Database Khuôn mặt
spiffs_create_partition_image(spiffs ../spiffs_data FLASH_IN_PROJECT)
