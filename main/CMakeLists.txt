set(srcs "main.c" "app_video.c" "app_lcd.c" "ai_model.cpp")

idf_component_register(SRCS "${srcs}"
                       PRIV_INCLUDE_DIRS .)

idf_component_get_property(esp_dl_component_dir espressif__esp-dl COMPONENT_DIR)
set(cmake_dir "${esp_dl_component_dir}/fbs_loader/cmake")
include("${cmake_dir}/utilities.cmake")
target_add_aligned_binary_data(${COMPONENT_LIB}
                               "${PROJECT_DIR}/models/mango_cnn.espdl"
                               BINARY
                               RENAME_TO mango_cnn_espdl)
