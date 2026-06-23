#ifndef BITMAPS_H
#define BITMAPS_H

#include <Arduino.h>

#define FRAME_DELAY 42

#define FRAME_WIDTH_SUCCESS 64
#define FRAME_HEIGHT_SUCCESS 64
extern const byte frames_success[][512];
extern const int FRAME_COUNT_SUCCESS;

#define FRAME_WIDTH_UNLOCK 64
#define FRAME_HEIGHT_UNLOCK 64
extern const byte frames_unlock[][512];
extern const int FRAME_COUNT_UNLOCK;

#endif
