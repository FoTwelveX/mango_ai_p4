/*
 * SPDX-FileCopyrightText: 2024 Espressif Systems (Shanghai) CO LTD
 *
 * SPDX-License-Identifier: Apache-2.0
 */
#ifndef APP_LCD_H
#define APP_LCD_H

#define APP_LCD_H_RES                   (1024)
#define APP_LCD_V_RES                   (600)

#define APP_LCD_BUF_NUM                 (CONFIG_MANGO_CAM_BUF_COUNT)

#if CONFIG_LCD_PIXEL_FORMAT_RGB565
#define APP_LCD_BIT_PER_PIXEL           (16)
#define APP_MIPI_DPI_PX_FORMAT          (LCD_COLOR_PIXEL_FORMAT_RGB565)
#elif CONFIG_LCD_PIXEL_FORMAT_RGB888
#define APP_LCD_BIT_PER_PIXEL           (24)
#define APP_MIPI_DPI_PX_FORMAT          (LCD_COLOR_PIXEL_FORMAT_RGB888)
#endif

/**
 * @brief Initialize the LCD panel.
 *
 * This function initializes the LCD panel with the provided panel handle. It powers on the LCD,
 * installs the LCD driver, configures the bus, and sets up the panel.
 *
 * @param panel_handle Pointer to the LCD panel handle
 * @return
 *    - ESP_OK: Success
 *    - ESP_FAIL: Failure
 */
esp_err_t app_lcd_init(esp_lcd_panel_handle_t *panel_handle);

#endif
