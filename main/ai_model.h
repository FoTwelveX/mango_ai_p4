#pragma once

#include "esp_err.h"
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

esp_err_t ai_model_run_rgb888(const uint8_t *rgb_data, size_t rgb_data_len, int *prediction_out);

#ifdef __cplusplus
}
#endif