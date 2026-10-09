/*
 * SPDX-FileCopyrightText: 2024 Espressif Systems (Shanghai) CO LTD
 *
 * SPDX-License-Identifier: CC0-1.0
 */
#include "esp_err.h"
#include "esp_log.h"
#include "esp_video_init.h"
#include "esp_lcd_mipi_dsi.h"
#include "esp_lcd_panel_ops.h"
#include "esp_cache.h"
#include "esp_heap_caps.h"
#include "esp_private/esp_cache_private.h"
#include "esp_timer.h"
#include "driver/ppa.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "app_video.h"
#include "app_lcd.h"
#include "ai_model.h"

#define ALIGN_UP(num, align)    (((num) + ((align) - 1)) & ~((align) - 1))
#define AI_INPUT_WIDTH          (64)
#define AI_INPUT_CHANNELS       (3)
#define AI_INPUT_SIZE           (AI_INPUT_WIDTH * AI_INPUT_WIDTH * AI_INPUT_CHANNELS)
#define AI_TASK_STACK_SIZE      (8 * 1024)
#define AI_INFERENCE_INTERVAL_US (5000000LL)
#define AI_OSD_WIDTH            (360)
#define AI_OSD_HEIGHT           (48)
#define AI_FONT_SCALE           (2)

typedef enum {
    AI_DISPLAY_WAITING,
    AI_DISPLAY_RUNNING,
    AI_DISPLAY_UPDATED,
    AI_DISPLAY_FAILED,
} ai_display_state_t;

typedef struct {
    char character;
    uint8_t rows[7];
} ai_font_glyph_t;

static void camera_video_frame_operation(uint8_t *camera_buf, uint8_t camera_buf_index, uint32_t camera_buf_hes, uint32_t camera_buf_ves, size_t camera_buf_len, void *user_data);
static void ai_inference_task(void *user_data);
static void camera_frame_to_ai_rgb(const uint8_t *camera_buf, uint32_t width, uint32_t height, uint8_t *rgb_output);
static void maybe_submit_camera_frame(uint8_t *camera_buf, uint32_t width, uint32_t height, size_t camera_buf_len);
static void draw_ai_overlay(uint8_t *frame_buffer, uint32_t width, uint32_t height);
static uint8_t ai_font_row(char character, int row);
static void ai_osd_set_pixel(uint8_t *frame_buffer, uint32_t width, uint32_t height,
                             int x, int y, uint8_t red, uint8_t green, uint8_t blue);

static const char *TAG = "app_main";
static const ai_font_glyph_t AI_FONT[] = {
    {'A', {14, 17, 17, 31, 17, 17, 17}}, {'B', {30, 17, 17, 30, 17, 17, 30}},
    {'C', {14, 17, 16, 16, 16, 17, 14}}, {'D', {30, 17, 17, 17, 17, 17, 30}},
    {'E', {31, 16, 16, 30, 16, 16, 31}}, {'F', {31, 16, 16, 30, 16, 16, 16}},
    {'G', {14, 17, 16, 23, 17, 17, 15}}, {'H', {17, 17, 17, 31, 17, 17, 17}},
    {'I', {14, 4, 4, 4, 4, 4, 14}},   {'L', {16, 16, 16, 16, 16, 16, 31}},
    {'M', {17, 27, 21, 21, 17, 17, 17}}, {'N', {17, 25, 25, 21, 19, 19, 17}},
    {'O', {14, 17, 17, 17, 17, 17, 14}}, {'P', {30, 17, 17, 30, 16, 16, 16}},
    {'R', {30, 17, 17, 30, 20, 18, 17}}, {'S', {15, 16, 16, 14, 1, 1, 30}},
    {'T', {31, 4, 4, 4, 4, 4, 4}},       {'U', {17, 17, 17, 17, 17, 17, 14}},
    {'V', {17, 17, 17, 17, 17, 10, 4}},  {'W', {17, 17, 17, 21, 21, 21, 10}},
    {'Y', {17, 17, 10, 4, 4, 4, 4}},     {'0', {14, 17, 19, 21, 25, 17, 14}},
    {'1', {4, 12, 4, 4, 4, 4, 14}},      {'2', {14, 17, 1, 2, 4, 8, 31}},
    {'3', {30, 1, 1, 14, 1, 1, 30}},     {'4', {2, 6, 10, 18, 31, 2, 2}},
    {'5', {31, 16, 16, 30, 1, 1, 30}},   {'6', {14, 16, 16, 30, 17, 17, 14}},
    {'7', {31, 1, 2, 4, 8, 8, 8}},       {'8', {14, 17, 17, 14, 17, 17, 14}},
    {'9', {14, 17, 17, 15, 1, 1, 14}},   {':', {0, 4, 4, 0, 4, 4, 0}},
    {'-', {0, 0, 0, 31, 0, 0, 0}},
};

static esp_lcd_panel_handle_t display_panel;
static ppa_client_handle_t ppa_srm_handle = NULL;
static size_t data_cache_line_size = 0;
static void *lcd_buffer[APP_LCD_BUF_NUM];
static uint8_t ai_rgb_input[AI_INPUT_SIZE];
static TaskHandle_t ai_inference_task_handle;
static portMUX_TYPE ai_status_mux = portMUX_INITIALIZER_UNLOCKED;
static ai_display_state_t ai_display_state = AI_DISPLAY_WAITING;
static int ai_latest_prediction = -1;
static int64_t ai_next_inference_us;
static bool ai_inference_pending;

#if CONFIG_MANGO_ENABLE_PRINT_FPS_RATE_VALUE
static int fps_count;
static int64_t start_time;
#endif

void app_main(void)
{
    // Initialize the LCD
    ESP_ERROR_CHECK(app_lcd_init(&display_panel));

    // Initialize the PPA
    ppa_client_config_t ppa_srm_config = {
        .oper_type = PPA_OPERATION_SRM,
    };
    ESP_ERROR_CHECK(ppa_register_client(&ppa_srm_config, &ppa_srm_handle));
    ESP_ERROR_CHECK(esp_cache_get_alignment(MALLOC_CAP_SPIRAM, &data_cache_line_size));

    // Initialize the video camera
    esp_err_t ret = app_video_main(NULL);
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "video main init failed with error 0x%x", ret);
        return;
    }

    // Open the video device
    int video_cam_fd0 = app_video_open(APP_CAM_DEV_PATH, APP_VIDEO_FMT);
    if (video_cam_fd0 < 0) {
        ESP_LOGE(TAG, "video cam open failed");
        return;
    }

    // Get the LCD frame buffer
#if APP_LCD_BUF_NUM == 2
    ESP_ERROR_CHECK(esp_lcd_dpi_panel_get_frame_buffer(display_panel, 2, &lcd_buffer[0], &lcd_buffer[1]));
#else
    ESP_ERROR_CHECK(esp_lcd_dpi_panel_get_frame_buffer(display_panel, 3, &lcd_buffer[0], &lcd_buffer[1], &lcd_buffer[2]));
#endif

    // Set the video buffer
#if CONFIG_MANGO_USE_MEMORY_MAPPING
    ESP_LOGI(TAG, "Using map buffer");
    ESP_ERROR_CHECK(app_video_set_bufs(video_cam_fd0, APP_CAM_BUF_NUM, NULL)); // When setting the camera video buffer, it can be written as NULL to automatically allocate the buffer using mapping
#else
    ESP_LOGI(TAG, "Using user defined buffer");
#if CONFIG_CAMERA_SC2336_MIPI_RAW8_1024x600_30FPS
    ESP_ERROR_CHECK(app_video_set_bufs(video_cam_fd0, APP_CAM_BUF_NUM, (void *)lcd_buffer));
#else
    void *camera_buf[APP_CAM_BUF_NUM];
    for (int i = 0; i < APP_CAM_BUF_NUM; i++) {
        camera_buf[i] = heap_caps_aligned_calloc(data_cache_line_size, 1, app_video_get_buf_size(), MALLOC_CAP_SPIRAM);
    }
    ESP_ERROR_CHECK(app_video_set_bufs(video_cam_fd0, APP_CAM_BUF_NUM, (void *)camera_buf));
#endif
#endif

    // Register the video frame operation callback
    ESP_ERROR_CHECK(app_video_register_frame_operation_cb(camera_video_frame_operation));

    if (xTaskCreatePinnedToCore(ai_inference_task, "ai inference task", AI_TASK_STACK_SIZE, NULL,
                                tskIDLE_PRIORITY + 2, &ai_inference_task_handle, 1) != pdPASS) {
        ai_inference_task_handle = NULL;
        portENTER_CRITICAL(&ai_status_mux);
        ai_display_state = AI_DISPLAY_FAILED;
        portEXIT_CRITICAL(&ai_status_mux);
        ESP_LOGE(TAG, "Failed to create one-shot AI inference task");
    }

    // Start the camera stream task
    ESP_ERROR_CHECK(app_video_stream_task_start(video_cam_fd0, 0, NULL));

#if CONFIG_MANGO_ENABLE_PRINT_FPS_RATE_VALUE
    start_time = esp_timer_get_time();  // Get the initial time for frame rate statistics
#endif
}

static void camera_video_frame_operation(uint8_t *camera_buf, uint8_t camera_buf_index, uint32_t camera_buf_hes, uint32_t camera_buf_ves, size_t camera_buf_len, void *user_data)
{
    maybe_submit_camera_frame(camera_buf, camera_buf_hes, camera_buf_ves, camera_buf_len);

#if CONFIG_MANGO_ENABLE_PRINT_FPS_RATE_VALUE
    fps_count++;
    if (fps_count == 50) {
        int64_t end_time = esp_timer_get_time();
        ESP_LOGI(TAG, "fps: %f", 1000000.0 / ((end_time - start_time) / 50.0));
        start_time = end_time;
        fps_count = 0;

        ESP_LOGI(TAG, "camera_buf_hes: %lu, camera_buf_ves: %lu, camera_buf_len: %d KB", camera_buf_hes, camera_buf_ves, camera_buf_len / 1024);
    }
#endif

    ppa_srm_oper_config_t srm_config = {
        .in.buffer = camera_buf,
        .in.pic_w = camera_buf_hes,
        .in.pic_h = camera_buf_ves,
        .in.block_w = camera_buf_hes,
        .in.block_h = camera_buf_ves,
        .in.block_offset_x = (camera_buf_hes > APP_LCD_H_RES) ? (camera_buf_hes - APP_LCD_H_RES) / 2 : 0,
           .in.block_offset_y = (camera_buf_ves > APP_LCD_V_RES) ? (camera_buf_ves - APP_LCD_V_RES) / 2 : 0,
           .in.srm_cm = APP_VIDEO_FMT == APP_VIDEO_FMT_RGB565 ? PPA_SRM_COLOR_MODE_RGB565 : PPA_SRM_COLOR_MODE_RGB888,
           .out.buffer = lcd_buffer[camera_buf_index],
           .out.buffer_size = ALIGN_UP(APP_LCD_H_RES * APP_LCD_V_RES * (APP_VIDEO_FMT == APP_VIDEO_FMT_RGB565 ? 2 : 3), data_cache_line_size),
           .out.pic_w = APP_LCD_H_RES,
           .out.pic_h = APP_LCD_V_RES,
           .out.block_offset_x = 0,
           .out.block_offset_y = 0,
           .out.srm_cm = APP_VIDEO_FMT == APP_VIDEO_FMT_RGB565 ? PPA_SRM_COLOR_MODE_RGB565 : PPA_SRM_COLOR_MODE_RGB888,
           .rotation_angle = PPA_SRM_ROTATION_ANGLE_0,
           .scale_x = 1,
           .scale_y = 1,
           .rgb_swap = 0,
           .byte_swap = 0,
           .mode = PPA_TRANS_MODE_BLOCKING,
    };

    if (camera_buf_hes > APP_LCD_H_RES || camera_buf_ves > APP_LCD_V_RES) {
        // The resolution of the camera does not match the LCD resolution. Image processing can be done using PPA, but there will be some frame rate loss

        srm_config.in.block_w = (camera_buf_hes > APP_LCD_H_RES) ? APP_LCD_H_RES : camera_buf_hes;
        srm_config.in.block_h = (camera_buf_ves > APP_LCD_V_RES) ? APP_LCD_V_RES : camera_buf_ves;

        ESP_ERROR_CHECK(ppa_do_scale_rotate_mirror(ppa_srm_handle, &srm_config));
        draw_ai_overlay(lcd_buffer[camera_buf_index], APP_LCD_H_RES, APP_LCD_V_RES);
        ESP_ERROR_CHECK(esp_lcd_panel_draw_bitmap(display_panel, 0, 0, srm_config.in.block_w, srm_config.in.block_h, lcd_buffer[camera_buf_index]));
    } else {
        draw_ai_overlay(camera_buf, camera_buf_hes, camera_buf_ves);
        ESP_ERROR_CHECK(esp_lcd_panel_draw_bitmap(display_panel, 0, 0, camera_buf_hes, camera_buf_ves, camera_buf));
    }
}

static void ai_inference_task(void *user_data)
{
    (void)user_data;
    while (true) {
        ulTaskNotifyTake(pdTRUE, portMAX_DELAY);

        int prediction = -1;
        esp_err_t ret = ai_model_run_rgb888(ai_rgb_input, sizeof(ai_rgb_input), &prediction);

        portENTER_CRITICAL(&ai_status_mux);
        ai_inference_pending = false;
        ai_display_state = ret == ESP_OK ? AI_DISPLAY_UPDATED : AI_DISPLAY_FAILED;
        if (ret == ESP_OK) {
            ai_latest_prediction = prediction;
        }
        portEXIT_CRITICAL(&ai_status_mux);

        if (ret != ESP_OK) {
            ESP_LOGE(TAG, "Camera-frame AI inference failed with error 0x%x", ret);
        }
    }
}

static void maybe_submit_camera_frame(uint8_t *camera_buf, uint32_t width, uint32_t height, size_t camera_buf_len)
{
    const int64_t now = esp_timer_get_time();
    bool due = false;

    portENTER_CRITICAL(&ai_status_mux);
    if (ai_inference_task_handle != NULL && !ai_inference_pending && now >= ai_next_inference_us) {
        ai_inference_pending = true;
        ai_display_state = AI_DISPLAY_RUNNING;
        ai_next_inference_us = now + AI_INFERENCE_INTERVAL_US;
        due = true;
    }
    portEXIT_CRITICAL(&ai_status_mux);

    if (!due) {
        return;
    }

#if CONFIG_LCD_PIXEL_FORMAT_RGB565
    const size_t bytes_per_pixel = 2;
#else
    const size_t bytes_per_pixel = 3;
#endif
    const size_t expected_frame_len = (size_t)width * height * bytes_per_pixel;
    if (camera_buf == NULL || width == 0 || height == 0 || camera_buf_len < expected_frame_len) {
        ESP_LOGE(TAG, "Invalid camera frame for AI: %lu x %lu, %u bytes",
                 (unsigned long)width, (unsigned long)height, (unsigned)camera_buf_len);
        portENTER_CRITICAL(&ai_status_mux);
        ai_inference_pending = false;
        ai_display_state = AI_DISPLAY_FAILED;
        portEXIT_CRITICAL(&ai_status_mux);
        return;
    }

    camera_frame_to_ai_rgb(camera_buf, width, height, ai_rgb_input);
    xTaskNotifyGive(ai_inference_task_handle);
    ESP_LOGI(TAG, "Captured fresh %lux%lu camera frame for AI", (unsigned long)width, (unsigned long)height);
}

static void draw_ai_overlay(uint8_t *frame_buffer, uint32_t width, uint32_t height)
{
    if (frame_buffer == NULL || width == 0 || height == 0) {
        return;
    }

    int prediction;
    ai_display_state_t state;
    portENTER_CRITICAL(&ai_status_mux);
    prediction = ai_latest_prediction;
    state = ai_display_state;
    portEXIT_CRITICAL(&ai_status_mux);

    const char *class_name = "--";
    switch (prediction) {
    case 0:
        class_name = "SEMI-RIPE";
        break;
    case 1:
        class_name = "RIPE";
        break;
    case 2:
        class_name = "UNRIPE";
        break;
    default:
        break;
    }

    const char *status_name = "WAITING";
    switch (state) {
    case AI_DISPLAY_RUNNING:
        status_name = "RUNNING";
        break;
    case AI_DISPLAY_UPDATED:
        status_name = "UPDATED";
        break;
    case AI_DISPLAY_FAILED:
        status_name = "FAILED";
        break;
    default:
        break;
    }

    const int overlay_width = width < AI_OSD_WIDTH ? width : AI_OSD_WIDTH;
    const int overlay_height = height < AI_OSD_HEIGHT ? height : AI_OSD_HEIGHT;
    for (int y = 0; y < overlay_height; ++y) {
        for (int x = 0; x < overlay_width; ++x) {
            ai_osd_set_pixel(frame_buffer, width, height, x, y, 16, 28, 40);
        }
    }

    char first_line[32];
    char second_line[24];
    snprintf(first_line, sizeof(first_line), "AI MATURITY: %s", class_name);
    snprintf(second_line, sizeof(second_line), "AI: %s", status_name);

    const char *lines[] = {first_line, second_line};
    const int line_y[] = {4, 25};
    for (int line = 0; line < 2; ++line) {
        int x = 10;
        for (const char *character = lines[line]; *character != '\0'; ++character) {
            for (int row = 0; row < 7; ++row) {
                const uint8_t row_bits = ai_font_row(*character, row);
                for (int column = 0; column < 5; ++column) {
                    if ((row_bits & (1U << (4 - column))) == 0) {
                        continue;
                    }
                    for (int scale_y = 0; scale_y < AI_FONT_SCALE; ++scale_y) {
                        for (int scale_x = 0; scale_x < AI_FONT_SCALE; ++scale_x) {
                            ai_osd_set_pixel(frame_buffer, width, height,
                                             x + column * AI_FONT_SCALE + scale_x,
                                             line_y[line] + row * AI_FONT_SCALE + scale_y, 255, 255, 255);
                        }
                    }
                }
            }
            x += 6 * AI_FONT_SCALE;
        }
    }
}

static void ai_osd_set_pixel(uint8_t *frame_buffer, uint32_t width, uint32_t height,
                             int x, int y, uint8_t red, uint8_t green, uint8_t blue)
{
    if (x < 0 || x >= width || y < 0 || y >= height) {
        return;
    }

#if CONFIG_LCD_PIXEL_FORMAT_RGB565
    uint16_t color = ((uint16_t)(red >> 3) << 11) | ((uint16_t)(green >> 2) << 5) | (blue >> 3);
    ((uint16_t *)frame_buffer)[(size_t)y * width + x] = color;
#else
    size_t offset = ((size_t)y * width + x) * 3;
    frame_buffer[offset] = red;
    frame_buffer[offset + 1] = green;
    frame_buffer[offset + 2] = blue;
#endif
}

static uint8_t ai_font_row(char character, int row)
{
    if (character == ' ') {
        return 0;
    }
    for (size_t i = 0; i < sizeof(AI_FONT) / sizeof(AI_FONT[0]); ++i) {
        if (AI_FONT[i].character == character) {
            return AI_FONT[i].rows[row];
        }
    }
    return 0;
}

static void camera_frame_to_ai_rgb(const uint8_t *camera_buf, uint32_t width, uint32_t height, uint8_t *rgb_output)
{
    for (uint32_t output_y = 0; output_y < AI_INPUT_WIDTH; ++output_y) {
        float source_y = ((output_y + 0.5f) * height / AI_INPUT_WIDTH) - 0.5f;
        if (source_y < 0.0f) {
            source_y = 0.0f;
        }
        uint32_t y0 = (uint32_t)source_y;
        uint32_t y1 = (y0 + 1 < height) ? y0 + 1 : y0;
        float y_weight = source_y - y0;

        for (uint32_t output_x = 0; output_x < AI_INPUT_WIDTH; ++output_x) {
            float source_x = ((output_x + 0.5f) * width / AI_INPUT_WIDTH) - 0.5f;
            if (source_x < 0.0f) {
                source_x = 0.0f;
            }
            uint32_t x0 = (uint32_t)source_x;
            uint32_t x1 = (x0 + 1 < width) ? x0 + 1 : x0;
            float x_weight = source_x - x0;
            uint8_t rgb00[3];
            uint8_t rgb01[3];
            uint8_t rgb10[3];
            uint8_t rgb11[3];

#if CONFIG_LCD_PIXEL_FORMAT_RGB565
            const uint16_t *pixels = (const uint16_t *)camera_buf;
            const uint16_t pixel00 = pixels[(size_t)y0 * width + x0];
            const uint16_t pixel01 = pixels[(size_t)y0 * width + x1];
            const uint16_t pixel10 = pixels[(size_t)y1 * width + x0];
            const uint16_t pixel11 = pixels[(size_t)y1 * width + x1];
            rgb00[0] = (((pixel00 >> 11) & 0x1f) * 255 + 15) / 31;
            rgb00[1] = (((pixel00 >> 5) & 0x3f) * 255 + 31) / 63;
            rgb00[2] = ((pixel00 & 0x1f) * 255 + 15) / 31;
            rgb01[0] = (((pixel01 >> 11) & 0x1f) * 255 + 15) / 31;
            rgb01[1] = (((pixel01 >> 5) & 0x3f) * 255 + 31) / 63;
            rgb01[2] = ((pixel01 & 0x1f) * 255 + 15) / 31;
            rgb10[0] = (((pixel10 >> 11) & 0x1f) * 255 + 15) / 31;
            rgb10[1] = (((pixel10 >> 5) & 0x3f) * 255 + 31) / 63;
            rgb10[2] = ((pixel10 & 0x1f) * 255 + 15) / 31;
            rgb11[0] = (((pixel11 >> 11) & 0x1f) * 255 + 15) / 31;
            rgb11[1] = (((pixel11 >> 5) & 0x3f) * 255 + 31) / 63;
            rgb11[2] = ((pixel11 & 0x1f) * 255 + 15) / 31;
#else
            const uint8_t *pixel00 = camera_buf + ((size_t)y0 * width + x0) * 3;
            const uint8_t *pixel01 = camera_buf + ((size_t)y0 * width + x1) * 3;
            const uint8_t *pixel10 = camera_buf + ((size_t)y1 * width + x0) * 3;
            const uint8_t *pixel11 = camera_buf + ((size_t)y1 * width + x1) * 3;
            for (int channel = 0; channel < 3; ++channel) {
                rgb00[channel] = pixel00[channel];
                rgb01[channel] = pixel01[channel];
                rgb10[channel] = pixel10[channel];
                rgb11[channel] = pixel11[channel];
            }
#endif

            const size_t output_offset = ((size_t)output_y * AI_INPUT_WIDTH + output_x) * AI_INPUT_CHANNELS;
            for (int channel = 0; channel < AI_INPUT_CHANNELS; ++channel) {
                float top = rgb00[channel] + (rgb01[channel] - rgb00[channel]) * x_weight;
                float bottom = rgb10[channel] + (rgb11[channel] - rgb10[channel]) * x_weight;
                rgb_output[output_offset + channel] = (uint8_t)(top + (bottom - top) * y_weight + 0.5f);
            }
        }
    }
}
