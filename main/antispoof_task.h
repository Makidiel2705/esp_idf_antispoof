#ifndef ANTISPOOF_TASK_H
#define ANTISPOOF_TASK_H

#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Initialize the Anti-spoofing model from a file path.
 * 
 * @param model_path Path to the .espdl model file (e.g. "/model/antispoof.espdl")
 * @return true if initialization is successful, false otherwise.
 */
bool antispoof_init(const char* model_path);

/**
 * @brief Run inference on a detected face image.
 * 
 * @param face_image Pointer to the face image buffer (RGB888, 112x112 expected)
 * @param width Image width
 * @param height Image height
 * @return float Confidence score (0.0 to 1.0). High score = Real, Low score = Fake.
 */
float antispoof_run(uint8_t* face_image, int width, int height);

/**
 * @brief Run inference by cropping/resizing an RGB888 frame directly into the model input tensor.
 *
 * This avoids the intermediate 80x80 RGB buffer in the hot path.
 */
float antispoof_run_rgb888_crop(const uint8_t* frame,
                                int frame_width,
                                int frame_height,
                                int crop_left,
                                int crop_top,
                                int crop_right,
                                int crop_bottom);

/**
 * @brief Free the model and resources.
 */
void antispoof_deinit(void);

#ifdef __cplusplus
}
#endif

#endif
