# Real-time Speech Translation Tool (实时语音翻译工具)

A real-time speech translation tool that captures system audio and provides instant translation from English to Chinese. Built with Python using PyQt6 for the GUI, Faster-Whisper for speech recognition, and Opus-MT for translation.

## Features

- **Real-time Audio Capture**: Captures system audio through loopback devices
- **Speech Recognition**: Uses Faster-Whisper for accurate speech-to-text conversion
- **Translation**: Translates English speech to Chinese in real-time
- **User-friendly Interface**: 
  - Frameless, always-on-top window
  - Draggable interface
  - Device selection dropdown
  - Live device switching without restart
  - Model configuration options
- **Configurable Options**:
  - Source language selection (Auto-detect/English/Japanese)
  - Whisper model size (base/small/medium)
  - Translation model selection
  - Audio device selection with refresh capability

## Requirements

- Windows 10/11
- Python 3.8+
- NVIDIA GPU (recommended) with CUDA support for better performance
- System audio loopback device enabled (e.g., Stereo Mix, What U Hear)

## Installation

1. Clone the repository:
```bash
git clone [repository-url]
cd [repository-name]
```

2. Create and activate a virtual environment (recommended):
```bash
python -m venv venv
.\venv\Scripts\activate
```

3. Install required packages:
```bash
pip install PyQt6 numpy soundcard faster-whisper transformers torch
```

4. Enable system audio loopback:
   - Open Windows Sound settings
   - Enable Stereo Mix or similar loopback device
   - If not visible, right-click in the recording devices area and enable "Show Disabled Devices"

## Usage

1. Run the application:
```bash
python main.py
```

2. Configure the application:
   - Select your preferred Whisper model size (larger models are more accurate but slower)
   - Choose source language or leave as auto-detect
   - Select the appropriate audio capture device from the dropdown
   - Click "Apply Settings" to start

3. The application will:
   - Capture system audio
   - Convert speech to text
   - Translate to Chinese
   - Display results in real-time

4. Features during use:
   - Drag the window by clicking and holding anywhere
   - Switch audio devices on-the-fly using the "Switch Device" button
   - Refresh the device list if new audio devices are connected
   - Apply new settings at any time using the "Apply Settings" button

## Notes

- For optimal performance, use a GPU with CUDA support
- Larger Whisper models provide better accuracy but require more VRAM
- The application requires a working audio loopback device
- Translation quality depends on the selected model and audio clarity

## Troubleshooting

- If no audio devices are shown:
  - Check if Stereo Mix is enabled in Windows sound settings
  - Click "Refresh Devices" to rescan
  - Try running as administrator
- If translations are slow:
  - Try a smaller Whisper model
  - Ensure GPU acceleration is working (if available)
- If the application crashes:
  - Check the logs for error messages
  - Ensure all required packages are installed
  - Verify system audio settings

## License

[Your chosen license]

## Acknowledgments

- [Faster-Whisper](https://github.com/guillaumekln/faster-whisper) for speech recognition
- [Opus-MT](https://github.com/Helsinki-NLP/Opus-MT) for translation
- [SoundCard](https://github.com/bastibe/SoundCard) for audio capture
- [PyQt6](https://www.riverbankcomputing.com/software/pyqt/) for the GUI framework 