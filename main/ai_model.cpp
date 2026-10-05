#include "ai_model.h"

#include "dl_model_base.hpp"
#include "esp_log.h"

#include <cmath>
#include <cstring>
#include <new>
#include <vector>

extern "C" const uint8_t mango_cnn_espdl[];

namespace {

static const char *TAG = "ai_model";
static const char *CLASS_NAMES[] = {"半成熟", "成熟", "未成熟"};
static dl::Model *model = nullptr;

} // namespace

extern "C" esp_err_t ai_model_run_fixed_input(void)
{
    if (model == nullptr) {
        model = new (std::nothrow) dl::Model(reinterpret_cast<const char *>(mango_cnn_espdl),
                                              fbs::MODEL_LOCATION_IN_FLASH_RODATA);
        if (model == nullptr || model->get_fbs_model() == nullptr) {
            ESP_LOGE(TAG, "Failed to load mango_cnn.espdl");
            return ESP_FAIL;
        }
    }

    dl::TensorBase *input = model->get_input();
    dl::TensorBase *output = model->get_output();
    const std::vector<int> expected_input_shape = {1, 64, 64, 3};
    const std::vector<int> expected_output_shape = {1, 3};
    if (input == nullptr || output == nullptr || input->get_shape() != expected_input_shape ||
        output->get_shape() != expected_output_shape || input->get_dtype() != dl::DATA_TYPE_INT8 ||
        output->get_dtype() != dl::DATA_TYPE_INT8 || !output->exponent.is_valid()) {
        ESP_LOGE(TAG, "Unexpected model input or output tensor configuration");
        return ESP_ERR_INVALID_SIZE;
    }

    void *input_data = input->get_element_ptr();
    int8_t *output_data = output->get_element_ptr<int8_t>();
    if (input_data == nullptr || output_data == nullptr) {
        ESP_LOGE(TAG, "Model tensor data is unavailable");
        return ESP_FAIL;
    }

    memset(input_data, 0, input->get_aligned_bytes());
    ESP_LOGI(TAG, "AI model loaded successfully");
    ESP_LOGI(TAG, "Input shape: 1 x 64 x 64 x 3 (NHWC), dtype: INT8, exponent: %d", input->get_exponent());
    model->print();

    model->run();

    ESP_LOGI(TAG, "Inference result:");
    int prediction = 0;
    float scores[3];
    for (int i = 0; i < 3; ++i) {
        const float scale = std::ldexp(1.0f, output->exponent.get(i));
        scores[i] = dl::dequantize<int8_t>(output_data[i], scale);
        ESP_LOGI(TAG, "%s: %.6f", CLASS_NAMES[i], static_cast<double>(scores[i]));
        if (scores[i] > scores[prediction]) {
            prediction = i;
        }
    }

    ESP_LOGI(TAG, "Prediction: %s", CLASS_NAMES[prediction]);
    return ESP_OK;
}