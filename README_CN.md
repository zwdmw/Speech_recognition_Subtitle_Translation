# 实时语音翻译工具

一个实时语音翻译工具，可以捕获系统音频并提供即时的英语到中文翻译。使用 PyQt6 构建图形界面，Faster-Whisper 进行语音识别，Opus-MT 进行翻译。

## 功能特点

- **实时音频捕获**：通过系统回环设备捕获音频
- **语音识别**：使用 Faster-Whisper 进行准确的语音转文字
- **实时翻译**：将英语语音实时翻译成中文
- **用户友好界面**：
  - 无边框、置顶窗口
  - 可拖拽界面
  - 设备选择下拉菜单
  - 无需重启即可切换音频设备
  - 模型配置选项
- **可配置选项**：
  - 源语言选择（自动检测/英语/日语）
  - Whisper 模型大小选择（base/small/medium）
  - 翻译模型选择
  - 音频设备选择及刷新功能

## 系统要求

- Windows 10/11
- Python 3.8+
- NVIDIA GPU（推荐）支持 CUDA 以获得更好性能
- 启用系统音频回环设备（如立体声混音、What U Hear 等）

## 安装步骤

1. 克隆仓库：
```bash
git clone [仓库地址]
cd [仓库名称]
```

2. 创建并激活虚拟环境（推荐）：
```bash
python -m venv venv
.\venv\Scripts\activate
```

3. 安装所需包：
```bash
pip install PyQt6 numpy soundcard faster-whisper transformers torch
```

4. 启用系统音频回环：
   - 打开 Windows 声音设置
   - 启用立体声混音或类似的回环设备
   - 如果看不到设备，右键点击录音设备区域并启用"显示禁用的设备"

## 使用方法

1. 运行应用：
```bash
python main.py
```

2. 配置应用：
   - 选择合适的 Whisper 模型大小（更大的模型更准确但更慢）
   - 选择源语言或保持自动检测
   - 从下拉菜单选择合适的音频捕获设备
   - 点击"应用设置"开始运行

3. 应用将会：
   - 捕获系统音频
   - 将语音转换为文字
   - 翻译成中文
   - 实时显示结果

4. 使用过程中的功能：
   - 点击并按住窗口任意位置可拖动
   - 使用"切换设备"按钮实时切换音频设备
   - 连接新音频设备后可刷新设备列表
   - 随时使用"应用设置"按钮更改配置

## 注意事项

- 为获得最佳性能，建议使用支持 CUDA 的 GPU
- 更大的 Whisper 模型提供更好的准确度但需要更多显存
- 应用需要正常工作的音频回环设备
- 翻译质量取决于所选模型和音频清晰度

## 故障排除

- 如果没有显示音频设备：
  - 检查 Windows 声音设置中是否启用了立体声混音
  - 点击"刷新设备"重新扫描
  - 尝试以管理员身份运行
- 如果翻译速度慢：
  - 尝试使用更小的 Whisper 模型
  - 确保 GPU 加速正常工作（如果可用）
- 如果应用崩溃：
  - 检查日志中的错误信息
  - 确保所有必需的包都已安装
  - 验证系统音频设置

## 开源许可

[选择的许可证]

## 致谢

- [Faster-Whisper](https://github.com/guillaumekln/faster-whisper) - 提供语音识别功能
- [Opus-MT](https://github.com/Helsinki-NLP/Opus-MT) - 提供翻译功能
- [SoundCard](https://github.com/bastibe/SoundCard) - 提供音频捕获功能
- [PyQt6](https://www.riverbankcomputing.com/software/pyqt/) - 提供图形界面框架 