| Supported Targets | ESP32-P4 |
| ----------------- | -------- |

# Mango AI — ESP32-P4 芒果成熟度识别

基于乐鑫官方 `video_lcd_display` 示例改造的 AI 项目：在 ESP32-P4 上通过 MIPI-CSI 摄像头采集画面并实时显示到 MIPI-DSI LCD 上，同时集成 [esp-dl](https://github.com/espressif/esp-dl) 推理框架，运行量化后的 CNN 模型 `mango_cnn`，对芒果成熟度进行三分类识别（**未成熟 / 半成熟 / 成熟**）。

## 功能特性

- 📷 **摄像头实时预览**：SC2336 MIPI-CSI 摄像头（RAW8 1024×600 @ 30fps），经 ISP 处理后输出 RGB565/RGB888 画面
- 🖥️ **LCD 显示**：7 寸 1024×600 MIPI-DSI 屏幕（EK79007 驱动），使用 PPA 硬件加速器完成缩放 / 裁剪 / 旋转
- 🧠 **端侧 AI 推理**：INT8 量化的芒果成熟度分类 CNN，以 `.espdl` 格式嵌入固件，通过 esp-dl 在启动时加载并推理
- ⚡ **PSRAM 200MHz**：16MB Flash + 高速 PSRAM，帧缓冲与模型均驻留在片上/PSRAM 中

## 硬件要求

- ESP32-P4-Function-EV-Board（或其他 ESP32-P4 开发板，16MB Flash）
- 7 寸 1024×600 MIPI-DSI LCD（EK79007 驱动 IC），通过 `MIPI_DSI` 接口连接
- SC2336 MIPI-CSI 摄像头，通过 `MIPI_CSI` 接口连接
- USB-C 数据线（供电 + 串口烧录）

屏幕转接板与开发板的额外连线：

| 屏幕转接板 | ESP32-P4-Function-EV-Board |
| ---------- | -------------------------- |
| 5V         | 5V                         |
| GND        | GND                        |
| PWM        | GPIO26                     |
| LCD_RST    | GPIO27                     |

## 项目结构

```
.
├── main/
│   ├── main.c              # 应用入口：AI 推理 + LCD/PPA/摄像头初始化与帧回调
│   ├── ai_model.cpp/h      # esp-dl 模型加载、推理与结果反量化（三分类）
│   ├── app_lcd.c/h         # MIPI-DSI LCD 面板初始化（EK79007）
│   ├── app_video.c/h       # esp_video 摄像头设备封装（V4L2 风格接口）
│   ├── Kconfig.projbuild   # menuconfig 示例配置项
│   └── idf_component.yml   # 依赖：esp_video、esp_lcd_ek79007、esp-dl
├── models/
│   ├── mango_cnn.pth       # PyTorch 原始权重
│   ├── mango_cnn.onnx(.data) # 导出的 ONNX 模型
│   ├── mango_cnn.json      # PPQ 量化配置
│   ├── mango_cnn.info      # espdl 模型结构信息（可读）
│   └── mango_cnn.espdl     # 烧录进固件的量化模型（INT8）
├── managed_components/     # idf.py 自动拉取的托管组件
├── partitions.csv          # 自定义分区表（factory 分区 15MB，容纳模型）
├── sdkconfig.defaults      # 默认配置（esp32p4、16MB Flash、SC2336 等）
└── CMakeLists.txt
```

## AI 模型

`mango_cnn` 是一个轻量级 CNN，经 PPQ 量化为 INT8 后由 esp-dl 加载：

| 项目 | 说明 |
| ---- | ---- |
| 输入 | `1 × 64 × 64 × 3`（NHWC，INT8，exponent = -7） |
| 输出 | `1 × 3`（INT8），对应类别 **半成熟 / 成熟 / 未成熟** |
| 结构 | Conv3×3(16) → Pool → Conv3×3(32) → Pool → Conv3×3(64) → Pool → FC(4096→128) → FC(128→3) |
| 格式 | `.espdl`（Flash XIP 只读加载，`MODEL_LOCATION_IN_FLASH_RODATA`） |

模型二进制通过 CMake 的 `target_add_aligned_binary_data` 嵌入固件（符号 `mango_cnn_espdl`）。

> **当前状态**：`app_main()` 启动时调用 `ai_model_run_fixed_input()`，以全零输入执行一次推理并在串口打印各类别得分与预测结果，用于验证模型部署链路。将摄像头帧缩放为 64×64 送入模型做实时识别是下一步工作。

## 环境要求

- [ESP-IDF](https://docs.espressif.com/projects/esp-idf/zh_CN/latest/esp32p4/get-started/) **v5.4 或更高版本**
- 首次使用请先按官方指南完成开发环境搭建

## 编译与烧录

```bash
# 设置目标芯片（首次）
idf.py set-target esp32p4

# 编译
idf.py build

# 烧录并查看串口输出
idf.py -p COMx flash monitor
```

启动后串口会输出 AI 模型加载信息、各类别推理得分，随后进入摄像头预览（默认每 50 帧打印一次 FPS）。

## 项目配置

运行 `idf.py menuconfig`：

- **Example Configuration**：摄像头 SCCB I2C 引脚、帧率打印、缓冲区分配方式（内存映射 / 用户分配）、摄像头缓冲区数量、LCD 像素格式（RGB565 / RGB888）、画面翻转等
- **Component config → Espressif Camera Sensors**：选择摄像头型号及输出格式，本项目默认：

```
Component config  --->
    Espressif Camera Sensors Configurations  --->
        [*] SC2336  ---->
            (X) RAW8 1024x600 30fps, MIPI 2lane 24M input
```

`sdkconfig.defaults` 中已固化关键配置：16MB Flash、自定义分区表、PSRAM 200MHz、SCCB 引脚（SCL=GPIO8 / SDA=GPIO7）等。

## 模型转换流程（可选）

若需要重新训练或更新模型：

1. 训练得到 `mango_cnn.pth`，导出为 ONNX（`mango_cnn.onnx`）
2. 使用 [esp-ppq](https://github.com/espressif/esp-ppq) 进行 INT8 量化（参考 `mango_cnn.json` 中的量化配置）
3. 使用 [espdl-cli](https://github.com/espressif/esp-dl) 将量化后的 ONNX 转换为 `.espdl`
4. 替换 `models/mango_cnn.espdl` 后重新编译固件即可

## 致谢

- 视频显示部分基于乐鑫官方示例 [video_lcd_display](https://github.com/espressif/esp-idf/tree/master/examples/peripherals/lcd/mipi_dsi)（CC0-1.0）
- AI 推理框架：[esp-dl](https://github.com/espressif/esp-dl)
- 摄像头组件：[esp_video](https://github.com/espressif/esp-video-components)
