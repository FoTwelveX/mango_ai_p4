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

extern "C" esp_err_t ai_model_run_rgb888(const uint8_t *rgb_data, size_t rgb_data_len, int *prediction_out)
{
    constexpr size_t expected_input_bytes = 64 * 64 * 3;
    if (rgb_data == nullptr || rgb_data_len != expected_input_bytes || prediction_out == nullptr) {
        ESP_LOGE(TAG, "Invalid RGB input or prediction output buffer: %u bytes", static_cast<unsigned>(rgb_data_len));
        return ESP_ERR_INVALID_ARG;
    }

    if (model == nullptr) {
        model = new (std::nothrow) dl::Model(reinterpret_cast<const char *>(mango_cnn_espdl),
                                              fbs::MODEL_LOCATION_IN_FLASH_RODATA);
        if (model == nullptr || model->get_fbs_model() == nullptr) {
            ESP_LOGE(TAG, "Failed to load mango_cnn.espdl");
            return ESP_FAIL;
        }
    }

    if (model->get_inputs().size() != 1 || model->get_outputs().size() != 1) {
        ESP_LOGE(TAG, "Unexpected model tensor count: inputs=%u outputs=%u",
                 static_cast<unsigned>(model->get_inputs().size()),
                 static_cast<unsigned>(model->get_outputs().size()));
        return ESP_ERR_INVALID_SIZE;
    }

    dl::TensorBase *input = model->get_input();
    dl::TensorBase *output = model->get_output();
    const std::vector<int> expected_input_shape = {1, 64, 64, 3};
    const std::vector<int> expected_output_shape = {1, 3};
    if (input == nullptr || output == nullptr || input->get_shape() != expected_input_shape ||
        output->get_shape() != expected_output_shape || input->get_dtype() != dl::DATA_TYPE_INT8 ||
        output->get_dtype() != dl::DATA_TYPE_INT8 || input->get_size() != expected_input_bytes ||
        output->get_size() != 3 || !input->exponent.is_valid() || !output->exponent.is_valid()) {
        ESP_LOGE(TAG, "Unexpected model input or output tensor configuration");
        return ESP_ERR_INVALID_SIZE;
    }

    int8_t *input_data = input->get_element_ptr<int8_t>();
    int8_t *output_data = output->get_element_ptr<int8_t>();
    if (input_data == nullptr || output_data == nullptr) {
        ESP_LOGE(TAG, "Model tensor data is unavailable");
        return ESP_FAIL;
    }

    const float input_scale = std::ldexp(1.0f, -input->get_exponent());
    for (size_t i = 0; i < expected_input_bytes; ++i) {
        input_data[i] = dl::quantize<int8_t>(rgb_data[i] / 255.0f, input_scale);
    }

    ESP_LOGI(TAG, "AI model loaded and tensor configuration validated");
    ESP_LOGI(TAG, "Input shape: 1 x 64 x 64 x 3 (NHWC), dtype: INT8, exponent: %d", input->get_exponent());
    model->print();

    ESP_LOGI(TAG, "Starting inference with camera RGB input");
    model->run();

    ESP_LOGI(TAG, "Inference returned; validating output values");
    int prediction = 0;
    float scores[3];
    for (int i = 0; i < 3; ++i) {
        const float scale = std::ldexp(1.0f, output->exponent.get(i));
        scores[i] = dl::dequantize<int8_t>(output_data[i], scale);
        if (!std::isfinite(scores[i])) {
            ESP_LOGE(TAG, "Invalid inference score for class %d: %f", i, static_cast<double>(scores[i]));
            return ESP_FAIL;
        }
        ESP_LOGI(TAG, "%s: %.6f", CLASS_NAMES[i], static_cast<double>(scores[i]));
        if (scores[i] > scores[prediction]) {
            prediction = i;
        }
    }

    if (prediction < 0 || prediction >= 3) {
        ESP_LOGE(TAG, "Prediction index out of range: %d", prediction);
        return ESP_ERR_INVALID_STATE;
    }

    ESP_LOGI(TAG, "Inference succeeded with 3 finite output scores");
    ESP_LOGI(TAG, "Prediction: %s", CLASS_NAMES[prediction]);
    *prediction_out = prediction;
    return ESP_OK;
}