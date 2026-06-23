#include "antispoof_task.h"
#include "esp_log.h"
#include "esp_timer.h"
#include "esp_heap_caps.h"
#include <math.h>
#include <algorithm>

// TensorFlow Lite Micro API
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"

static const char* TAG = "ANTISPOOF";

// Đây là điểm bắt đầu của mảng nhị phân TFLite được nhúng tự động bởi CMake.
// Mảng này sẽ CHƯA TỒN TẠI cho tới khi CMake tìm thấy file data/antispoof.tflite
extern const uint8_t antispoof_fullint8_tflite_start[] asm("_binary_antispoof_fullint8_local_tflite_start");
extern const uint8_t antispoof_fullint8_tflite_end[]   asm("_binary_antispoof_fullint8_local_tflite_end");

namespace {
    const tflite::Model* model = nullptr;
    tflite::MicroInterpreter* interpreter = nullptr;
    TfLiteTensor* input = nullptr;
    TfLiteTensor* output = nullptr;
    bool logged_input_type = false;
    constexpr int kRealClassIndex = 1;
    constexpr int kFakeClassIndex = 0;

    // Arena (Tensors) pointer - will be allocated in PSRAM
    constexpr int kTensorArenaSize = 3000 * 1024; // ~3MB
    uint8_t* tensor_arena = nullptr;

    static bool is_antispoof_ready = false;
}

static int tensor_element_count(const TfLiteTensor *tensor)
{
    if (!tensor || !tensor->dims) {
        return 0;
    }
    int count = 1;
    for (int i = 0; i < tensor->dims->size; ++i) {
        count *= tensor->dims->data[i];
    }
    return count;
}

static bool fill_input_from_rgb888_crop(const uint8_t *frame,
                                        int frame_width,
                                        int frame_height,
                                        int crop_left,
                                        int crop_top,
                                        int crop_right,
                                        int crop_bottom,
                                        int64_t total_start_us)
{
    if (!frame || !input || frame_width <= 0 || frame_height <= 0) {
        ESP_LOGE(TAG, "Invalid frame/input for anti-spoof preprocessing");
        return false;
    }

    const int req_width = input->dims->data[1];
    const int req_height = input->dims->data[2];
    if (req_width <= 0 || req_height <= 0) {
        ESP_LOGE(TAG, "Invalid anti-spoof input dims: %dx%d", req_width, req_height);
        return false;
    }

    crop_left = std::max(0, std::min(crop_left, frame_width - 1));
    crop_right = std::max(0, std::min(crop_right, frame_width - 1));
    crop_top = std::max(0, std::min(crop_top, frame_height - 1));
    crop_bottom = std::max(0, std::min(crop_bottom, frame_height - 1));
    if (crop_right < crop_left || crop_bottom < crop_top) {
        ESP_LOGE(TAG, "Invalid anti-spoof crop: l=%d t=%d r=%d b=%d", crop_left, crop_top, crop_right, crop_bottom);
        return false;
    }

    const int crop_w = crop_right - crop_left + 1;
    const int crop_h = crop_bottom - crop_top + 1;

    if (input->type == kTfLiteInt8) {
        if (!logged_input_type) {
            ESP_LOGI(TAG, "AI: Input type=INT8, Scale=%.6f, ZeroPoint=%d", input->params.scale, (int)input->params.zero_point);
            logged_input_type = true;
        }
        const float in_scale = input->params.scale;
        const int in_zp = input->params.zero_point;
        if (in_scale <= 0.0f) {
            ESP_LOGE(TAG, "Invalid INT8 input scale: %.8f", in_scale);
            return false;
        }

        int32_t total_brightness = 0;
        int32_t b_sum = 0;
        int32_t g_sum = 0;
        int32_t r_sum = 0;
        int sat_min = 0;
        int sat_max = 0;
        auto quantize = [&](uint8_t val) -> int8_t {
            float x = (in_scale < 0.01f) ? (val / 255.0f) : (float)val;
            int q = (int)lroundf(x / in_scale) + in_zp;
            if (q <= -128) {
                sat_min++;
                return -128;
            }
            if (q >= 127) {
                sat_max++;
                return 127;
            }
            return (int8_t)q;
        };

        for (int y = 0; y < req_height; ++y) {
            const int src_y = crop_top + (y * crop_h) / req_height;
            for (int x = 0; x < req_width; ++x) {
                const int src_x = crop_left + (x * crop_w) / req_width;
                const int src_idx = (src_y * frame_width + src_x) * 3;
                const uint8_t r = frame[src_idx + 0];
                const uint8_t g = frame[src_idx + 1];
                const uint8_t b = frame[src_idx + 2];
                const int dst_idx = (y * req_width + x) * 3;

                total_brightness += (r + g + b) / 3;
                b_sum += b;
                g_sum += g;
                r_sum += r;

                input->data.int8[dst_idx + 0] = quantize(b);
                input->data.int8[dst_idx + 1] = quantize(g);
                input->data.int8[dst_idx + 2] = quantize(r);
            }
        }

        float avg_brightness = (float)total_brightness / (req_width * req_height);
        ESP_LOGD(TAG, "Face crop avg brightness: %.1f", avg_brightness);
        ESP_LOGD(TAG,
                 "Face crop channel mean BGR: %.1f / %.1f / %.1f, input saturation min/max: %d / %d",
                 (float)b_sum / (req_width * req_height),
                 (float)g_sum / (req_width * req_height),
                 (float)r_sum / (req_width * req_height),
                 sat_min,
                 sat_max);
        if (avg_brightness < 20.0f) {
            ESP_LOGW(TAG, "Image too dark for AI (Avg: %.1f). Skipping...", avg_brightness);
            ESP_LOGI(TAG,
                     "ANTI_PROF: result=too_dark pre=%lldms invoke=0ms post=0ms total=%lldms",
                     (esp_timer_get_time() - total_start_us) / 1000,
                     (esp_timer_get_time() - total_start_us) / 1000);
            return false;
        }
    } else if (input->type == kTfLiteFloat32) {
        if (!logged_input_type) {
            ESP_LOGW(TAG, "Input type: FLOAT32 (slow on ESP32-S3)");
            logged_input_type = true;
        }
        for (int y = 0; y < req_height; ++y) {
            const int src_y = crop_top + (y * crop_h) / req_height;
            for (int x = 0; x < req_width; ++x) {
                const int src_x = crop_left + (x * crop_w) / req_width;
                const int src_idx = (src_y * frame_width + src_x) * 3;
                const int dst_idx = (y * req_width + x) * 3;
                input->data.f[dst_idx + 0] = frame[src_idx + 2] / 255.0f;
                input->data.f[dst_idx + 1] = frame[src_idx + 1] / 255.0f;
                input->data.f[dst_idx + 2] = frame[src_idx + 0] / 255.0f;
            }
        }
    } else {
        ESP_LOGE(TAG, "Unsupported anti-spoof input tensor type: %d", input->type);
        return false;
    }

    return true;
}

static float invoke_and_read_real_score(int64_t total_start_us, int64_t preprocess_us)
{
    int64_t invoke_start_us = esp_timer_get_time();
    TfLiteStatus invoke_status = interpreter->Invoke();
    int64_t invoke_us = esp_timer_get_time() - invoke_start_us;
    if (invoke_status != kTfLiteOk) {
        ESP_LOGE(TAG, "Invoke failed");
        ESP_LOGI(TAG,
                 "ANTI_PROF: result=invoke_failed pre=%lldms invoke=%lldms post=0ms total=%lldms",
                 preprocess_us / 1000,
                 invoke_us / 1000,
                 (esp_timer_get_time() - total_start_us) / 1000);
        return 0.0f;
    }
    ESP_LOGD(TAG, "Inference time: %lld us", invoke_us);

    int64_t post_start_us = esp_timer_get_time();
    int output_count = tensor_element_count(output);
    if (output_count < 2) {
        ESP_LOGE(TAG, "Unexpected anti-spoof output size: %d", output_count);
        return 0.0f;
    }

    int n = (output_count >= 3) ? 3 : 2;
    float logits[3] = {0};

    if (output->type == kTfLiteInt8) {
        if (n == 3) {
            ESP_LOGD(TAG, "Output Raw INT8[0..2]: %d, %d, %d", output->data.int8[0], output->data.int8[1], output->data.int8[2]);
        } else {
            ESP_LOGD(TAG, "Output Raw INT8[0..1]: %d, %d", output->data.int8[0], output->data.int8[1]);
        }
        for (int i = 0; i < n; ++i) {
            logits[i] = (output->data.int8[i] - output->params.zero_point) * output->params.scale;
        }
    } else if (output->type == kTfLiteFloat32) {
        for (int i = 0; i < n; ++i) {
            logits[i] = output->data.f[i];
        }
    } else {
        ESP_LOGE(TAG, "Unsupported output tensor type: %d", output->type);
        return 0.0f;
    }

    float max_l = logits[0];
    for (int i = 1; i < n; ++i) {
        if (logits[i] > max_l) {
            max_l = logits[i];
        }
    }
    float exps[3] = {0};
    float sum = 0.0f;
    for (int i = 0; i < n; ++i) {
        exps[i] = expf(logits[i] - max_l);
        sum += exps[i];
    }
    float probs[3] = {0};
    for (int i = 0; i < n; ++i) {
        probs[i] = (sum > 0.0f) ? (exps[i] / sum) : 0.0f;
    }

    float prob_real = probs[kRealClassIndex];
    float prob_fake = 1.0f - prob_real;
    int64_t post_us = esp_timer_get_time() - post_start_us;
    ESP_LOGI(TAG,
             "ANTI_PROF: result=ok pre=%lldms invoke=%lldms post=%lldms total=%lldms",
             preprocess_us / 1000,
             invoke_us / 1000,
             post_us / 1000,
             (esp_timer_get_time() - total_start_us) / 1000);

    if (n == 3) {
        ESP_LOGI(TAG, "Antispoof Logits[0..2] -> %.6f, %.6f(REAL), %.6f", logits[0], logits[1], logits[2]);
        ESP_LOGI(TAG,
                 "Antispoof Prob[0..2] -> %.2f%%, %.2f%%(REAL), %.2f%% | FAKE(total)=%.2f%%",
                 probs[0] * 100.0f,
                 probs[1] * 100.0f,
                 probs[2] * 100.0f,
                 prob_fake * 100.0f);
    } else {
        ESP_LOGI(TAG, "Antispoof Logits -> Real(idx=%d): %.6f, Fake(idx=%d): %.6f", kRealClassIndex, logits[kRealClassIndex], kFakeClassIndex, logits[kFakeClassIndex]);
        ESP_LOGI(TAG, "Antispoof Prob -> REAL: %.2f%%, FAKE: %.2f%%", prob_real * 100.0f, prob_fake * 100.0f);
    }

    return prob_real;
}

bool antispoof_init(const char* model_path) {
    ESP_LOGI(TAG, "Initializing TFLite Micro Antispoofing model...");

    // Allocate Arena in External RAM (SPIRAM) to avoid starving internal RAM
    if (!tensor_arena) {
        tensor_arena = (uint8_t*)heap_caps_malloc(kTensorArenaSize, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
        if (!tensor_arena) {
            ESP_LOGE(TAG, "Failed to allocate tensor arena in PSRAM! Falling back to Internal RAM...");
            tensor_arena = (uint8_t*)malloc(kTensorArenaSize);
            if (!tensor_arena) {
                ESP_LOGE(TAG, "Critical: Could not allocate memory for AI model.");
                return false;
            }
        }
    }

    model = tflite::GetModel(antispoof_fullint8_tflite_start);
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        ESP_LOGE(TAG, "Model provided is schema version %lu not equal to supported version %d.",
                 model->version(), TFLITE_SCHEMA_VERSION);
        return false;
    }

    // Đăng ký các phép toán mạng nơ-ron:
    // Cần bổ sung thêm Ops nếu model của bạn quá phức tạp, có thể dùng AllOpsResolver (nhưng tốn flash hơn)
    static tflite::MicroMutableOpResolver<20> micro_op_resolver;
    micro_op_resolver.AddConv2D();
    micro_op_resolver.AddPrelu(); // Thêm phép toán PRELU (Quan trọng cho mô hình của bạn)
    micro_op_resolver.AddDepthwiseConv2D();
    micro_op_resolver.AddFullyConnected();
    micro_op_resolver.AddSoftmax();
    micro_op_resolver.AddMaxPool2D();
    micro_op_resolver.AddAveragePool2D();
    micro_op_resolver.AddRelu();
    micro_op_resolver.AddRelu6();
    micro_op_resolver.AddReshape();
    micro_op_resolver.AddPad(); // Thêm phép toán PAD
    micro_op_resolver.AddAbs(); // Thêm phép toán ABS (Absolute Value)
    micro_op_resolver.AddSub(); // Thêm phép toán SUB (Subtract)
    micro_op_resolver.AddMul(); // Thêm phép toán MUL (Multiply)
    micro_op_resolver.AddAdd(); // Thêm phép toán ADD
    micro_op_resolver.AddLogistic(); // Thêm phép toán Sigmoid (Logistic)


    // Khởi tạo Interpreter
    static tflite::MicroInterpreter static_interpreter(
        model, micro_op_resolver, tensor_arena, kTensorArenaSize);
    interpreter = &static_interpreter;

    // Cấp phát bộ nhớ
    TfLiteStatus allocate_status = interpreter->AllocateTensors();
    if (allocate_status != kTfLiteOk) {
        ESP_LOGE(TAG, "AllocateTensors() failed");
        return false;
    }

    input = interpreter->input(0);
    output = interpreter->output(0);
    
    ESP_LOGI(TAG,
             "TFLite model loaded. Input dims: %d x %d, input type=%d (scale=%.8f, zp=%d), output type=%d (scale=%.8f, zp=%d)",
             input->dims->data[1],
             input->dims->data[2],
             input->type,
             input->params.scale,
             (int)input->params.zero_point,
             output->type,
             output->params.scale,
             (int)output->params.zero_point);
    ESP_LOGI(TAG, "Output tensor elements: %d, Silent-Face real class index=%d", tensor_element_count(output), kRealClassIndex);
    if (tensor_element_count(output) != 3) {
        ESP_LOGW(TAG, "Silent-Face checkpoints are typically 3-class. Current model output=%d classes.", tensor_element_count(output));
    }
    if (input->type == kTfLiteInt8 && input->params.scale < 0.01f) {
        ESP_LOGW(TAG, "Input scale %.6f suggests 0..1 calibration. Silent-Face original preprocessing uses float 0..255.", input->params.scale);
    }
    
    is_antispoof_ready = true;
    return true;
}

float antispoof_run_rgb888_crop(const uint8_t* frame,
                                int frame_width,
                                int frame_height,
                                int crop_left,
                                int crop_top,
                                int crop_right,
                                int crop_bottom)
{
    int64_t total_start_us = esp_timer_get_time();
    if (!is_antispoof_ready || !interpreter || !input || !output) {
        static bool error_logged = false;
        if (!error_logged) {
            ESP_LOGE(TAG, "AI model not ready or tensors are NULL. Check initialization logs.");
            error_logged = true;
        }
        return 0.0f;
    }

    int64_t preprocess_start_us = total_start_us;
    if (!fill_input_from_rgb888_crop(frame,
                                     frame_width,
                                     frame_height,
                                     crop_left,
                                     crop_top,
                                     crop_right,
                                     crop_bottom,
                                     total_start_us)) {
        return 0.05f;
    }
    int64_t preprocess_us = esp_timer_get_time() - preprocess_start_us;
    return invoke_and_read_real_score(total_start_us, preprocess_us);
}

// Giả định input ảnh khuôn mặt (đã crop) là 80x80 RGB888. 
float antispoof_run(uint8_t* face_image, int width, int height) {
    int64_t total_start_us = esp_timer_get_time();
    int64_t preprocess_start_us = total_start_us;
    int64_t preprocess_us = 0;
    int64_t invoke_us = 0;
    int64_t post_us = 0;

    if (!is_antispoof_ready || !interpreter || !input || !output) {
        static bool error_logged = false;
        if (!error_logged) {
            ESP_LOGE(TAG, "AI model not ready or tensors are NULL. Check initialization logs.");
            error_logged = true;
        }
        return 0.0f;
    }

    // 1. Tiền xử lý (Preprocessing): Resize/Crop thành 80x80 nếu chưa phải 80x80
    // Thông thường MiniFASNet đầu vào là 80x80x3, tuỳ loại lượng tử hóa (Float32 hoặc INT8)
    
    // (Bỏ qua logic resize phức tạp ở đây - giả định app_httpd.cpp đã xử lý mảng này và truyền vào)
    // Nếu file TFLite là INT8, ta phải map màu sắc [0-255] vào số nguyên do lượng tử hóa (-128...127).
    
    // Ví dụ cơ bản truyền dữ liệu (chỉ dùng cho mục đích demo)
    int req_width = input->dims->data[1]; // 80
    int req_height = input->dims->data[2]; // 80
    
    if (width == req_width && height == req_height) {
        if (input->type == kTfLiteInt8) {
            if (!logged_input_type) {
                ESP_LOGI(TAG, "AI: Input type=INT8, Scale=%.6f, ZeroPoint=%d", input->params.scale, (int)input->params.zero_point);
                logged_input_type = true;
            }
            const float in_scale = input->params.scale;
            const int in_zp = input->params.zero_point;
            if (in_scale <= 0.0f) {
                ESP_LOGE(TAG, "Invalid INT8 input scale: %.8f", in_scale);
                return 0.0f;
            }

            int32_t total_brightness = 0;
            int32_t b_sum = 0;
            int32_t g_sum = 0;
            int32_t r_sum = 0;
            int sat_min = 0;
            int sat_max = 0;
            for (int i = 0; i < width * height; i++) {
                uint8_t r = face_image[i * 3 + 0];
                uint8_t g = face_image[i * 3 + 1];
                uint8_t b = face_image[i * 3 + 2];
                total_brightness += (r + g + b) / 3;
                b_sum += b;
                g_sum += g;
                r_sum += r;
                
                // Quy đổi sang BGR và Lượng tử hóa INT8.
                // Silent-Face dùng ảnh float 0..255 (xem src/data_io/functional.py: to_tensor()) nên input quant thường có scale=1.0.
                // Một số export khác dùng 0..1 (scale ~ 1/255). Suy ra domain từ input->params.scale để tránh "mù" (toàn -128..-127).
                auto quantize = [&](uint8_t val) -> int8_t {
                    float x = (in_scale < 0.01f) ? (val / 255.0f) : (float)val;
                    int q = (int)lroundf(x / in_scale) + in_zp;
                    if (q <= -128) {
                        sat_min++;
                        return -128;
                    }
                    if (q >= 127) {
                        sat_max++;
                        return 127;
                    }
                    return (int8_t)q;
                };

                input->data.int8[i * 3 + 0] = quantize(b); // Blue
                input->data.int8[i * 3 + 1] = quantize(g); // Green
                input->data.int8[i * 3 + 2] = quantize(r); // Red
            }

            float avg_brightness = (float)total_brightness / (width * height);
            ESP_LOGD(TAG, "Face crop avg brightness: %.1f", avg_brightness);
            ESP_LOGD(TAG,
                     "Face crop channel mean BGR: %.1f / %.1f / %.1f, input saturation min/max: %d / %d",
                     (float)b_sum / (width * height),
                     (float)g_sum / (width * height),
                     (float)r_sum / (width * height),
                     sat_min,
                     sat_max);
            if (avg_brightness < 20.0f) {
                ESP_LOGW(TAG, "Image too dark for AI (Avg: %.1f). Skipping...", avg_brightness);
                preprocess_us = esp_timer_get_time() - preprocess_start_us;
                ESP_LOGI(TAG,
                         "ANTI_PROF: result=too_dark pre=%lldms invoke=0ms post=0ms total=%lldms",
                         preprocess_us / 1000,
                         (esp_timer_get_time() - total_start_us) / 1000);
                return 0.05f; // Trả về xác suất thấp (Fake)
            }

            // Log 5 byte đầu tiên để kiểm tra Saturation
            ESP_LOGD(TAG, "Input Raw INT8 (first 5): %d, %d, %d, %d, %d", 
                     input->data.int8[0], input->data.int8[1], input->data.int8[2], 
                     input->data.int8[3], input->data.int8[4]);
        } else if (input->type == kTfLiteFloat32) {
            if (!logged_input_type) {
                ESP_LOGW(TAG, "Input type: FLOAT32 (slow on ESP32-S3)");
                logged_input_type = true;
            }
            for (int i = 0; i < width * height; i++) {
                input->data.f[i * 3 + 0] = face_image[i * 3 + 2] / 255.0f; // Blue
                input->data.f[i * 3 + 1] = face_image[i * 3 + 1] / 255.0f; // Green
                input->data.f[i * 3 + 2] = face_image[i * 3 + 0] / 255.0f; // Red
            }
        }
    } else {
        ESP_LOGE(TAG, "Face image mismatch. Expected %dx%d, got %dx%d", req_width, req_height, width, height);
        return 0.0f; // Trả về 0 (Fake) nếu có lỗi
    }

    // 2. Suy luận (Inference)
    preprocess_us = esp_timer_get_time() - preprocess_start_us;
    int64_t invoke_start_us = esp_timer_get_time();
    TfLiteStatus invoke_status = interpreter->Invoke();
    invoke_us = esp_timer_get_time() - invoke_start_us;
    if (invoke_status != kTfLiteOk) {
        ESP_LOGE(TAG, "Invoke failed");
        ESP_LOGI(TAG,
                 "ANTI_PROF: result=invoke_failed pre=%lldms invoke=%lldms post=0ms total=%lldms",
                 preprocess_us / 1000,
                 invoke_us / 1000,
                 (esp_timer_get_time() - total_start_us) / 1000);
        return 0.0f;
    }
    ESP_LOGD(TAG, "Inference time: %lld us", invoke_us);

    // 3. Hậu xử lý (Post-processing):
    int64_t post_start_us = esp_timer_get_time();
    int output_count = tensor_element_count(output);
    if (output_count < 2) {
        ESP_LOGE(TAG, "Unexpected anti-spoof output size: %d", output_count);
        return 0.0f;
    }

    // Silent-Face default: output 3 classes, where class index 1 is REAL (xem Silent-Face-Anti-Spoofing-master/test.py).
    // Treat anything not class=1 as spoof.
    int n = (output_count >= 3) ? 3 : 2;
    float logits[3] = {0};

    if (output->type == kTfLiteInt8) {
        if (n == 3) {
            ESP_LOGD(TAG, "Output Raw INT8[0..2]: %d, %d, %d", output->data.int8[0], output->data.int8[1], output->data.int8[2]);
        } else {
            ESP_LOGD(TAG, "Output Raw INT8[0..1]: %d, %d", output->data.int8[0], output->data.int8[1]);
        }
        for (int i = 0; i < n; ++i) {
            logits[i] = (output->data.int8[i] - output->params.zero_point) * output->params.scale;
        }
    } else if (output->type == kTfLiteFloat32) {
        for (int i = 0; i < n; ++i) {
            logits[i] = output->data.f[i];
        }
    } else {
        ESP_LOGE(TAG, "Unsupported output tensor type: %d", output->type);
        return 0.0f;
    }

    // Numerically stable softmax (n=2 or n=3)
    float max_l = logits[0];
    for (int i = 1; i < n; ++i) {
        if (logits[i] > max_l) {
            max_l = logits[i];
        }
    }
    float exps[3] = {0};
    float sum = 0.0f;
    for (int i = 0; i < n; ++i) {
        exps[i] = expf(logits[i] - max_l);
        sum += exps[i];
    }
    float probs[3] = {0};
    for (int i = 0; i < n; ++i) {
        probs[i] = (sum > 0.0f) ? (exps[i] / sum) : 0.0f;
    }

    float prob_real = probs[kRealClassIndex];
    float prob_fake = 1.0f - prob_real;
    post_us = esp_timer_get_time() - post_start_us;
    ESP_LOGI(TAG,
             "ANTI_PROF: result=ok pre=%lldms invoke=%lldms post=%lldms total=%lldms",
             preprocess_us / 1000,
             invoke_us / 1000,
             post_us / 1000,
             (esp_timer_get_time() - total_start_us) / 1000);

    if (n == 3) {
        ESP_LOGI(TAG, "Antispoof Logits[0..2] -> %.6f, %.6f(REAL), %.6f", logits[0], logits[1], logits[2]);
        ESP_LOGI(TAG,
                 "Antispoof Prob[0..2] -> %.2f%%, %.2f%%(REAL), %.2f%% | FAKE(total)=%.2f%%",
                 probs[0] * 100.0f,
                 probs[1] * 100.0f,
                 probs[2] * 100.0f,
                 prob_fake * 100.0f);
    } else {
        ESP_LOGI(TAG, "Antispoof Logits -> Real(idx=%d): %.6f, Fake(idx=%d): %.6f", kRealClassIndex, logits[kRealClassIndex], kFakeClassIndex, logits[kFakeClassIndex]);
        ESP_LOGI(TAG, "Antispoof Prob -> REAL: %.2f%%, FAKE: %.2f%%", prob_real * 100.0f, prob_fake * 100.0f);
    }

    return prob_real;
}

void antispoof_deinit(void) {
    if (tensor_arena) {
        free(tensor_arena);
        tensor_arena = nullptr;
    }
    interpreter = nullptr;
}
