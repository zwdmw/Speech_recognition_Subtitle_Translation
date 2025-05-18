import sys
import time
import numpy as np
import soundcard as sc
from PyQt6.QtWidgets import QApplication, QWidget, QLabel, QVBoxLayout, QHBoxLayout, QComboBox, QPushButton
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QMutex, QWaitCondition, QTimer
from PyQt6.QtGui import QPalette, QColor
import faster_whisper
import transformers
import torch
import logging
import queue

# --- Configuration ---
# Whisper model: tiny, base, small, medium, large-v2, large-v3. Adjust based on VRAM/RAM and desired accuracy/speed.
WHISPER_MODEL_SIZE = "base" # Use 'small' model for potentially better accuracy
# Device: "cuda" for GPU (if available and PyTorch with CUDA is installed), "cpu" otherwise
DEVICE = "cuda" 
# Compute type: "float16" for faster GPU inference (if supported), "int8" for CPU, "float32" otherwise
COMPUTE_TYPE = "int8"
# Source language for Whisper (set to None for auto-detect, or e.g., "en", "zh")
# Force English as source since translator only supports en->zh
SOURCE_LANGUAGE = "en"
# Target language for translation (e.g., "zh" for Chinese, "en" for English)
TARGET_LANGUAGE_CODE = "zh" # ISO 639-1 code

# --- Translator Models ---
# Translator model ID from Hugging Face Hub (find appropriate model for your language pair)
# Use the specific En->Zh model
TRANSLATOR_MODEL_ID = f"Helsinki-NLP/opus-mt-en-{TARGET_LANGUAGE_CODE}"
# English to Chinese
# TRANSLATOR_MODEL_ID_EN_ZH = f"Helsinki-NLP/opus-mt-en-{TARGET_LANGUAGE_CODE}"
# Japanese to Chinese
# TRANSLATOR_MODEL_ID_JA_ZH = f"Helsinki-NLP/opus-mt-ja-{TARGET_LANGUAGE_CODE}"
# Add more language pairs here if needed

# Translator model ID from Hugging Face Hub (mBART multilingual model)
# MULTILINGUAL_TRANSLATOR_MODEL_ID = "facebook/mbart-large-50-many-to-many-mmt"

# --- Audio capture settings ---
SAMPLE_RATE = 16000  # Whisper requires 16kHz
BUFFER_SECONDS = 5   # Process audio in chunks of this duration
BLOCK_SIZE = int(SAMPLE_RATE * 0.1) # Small block size for lower latency capture feed
BUFFER_SIZE = SAMPLE_RATE * BUFFER_SECONDS

# --- Audio Devices: Stored here to avoid re-scanning ---
# These lists will be populated at startup and displayed to the user
ALL_AUDIO_DEVICES = []
LOOPBACK_DEVICES = []

# --- List all audio devices at startup ---
try:
    print("--- Available Microphones (including Loopback) ---")
    ALL_AUDIO_DEVICES = sc.all_microphones(include_loopback=True)
    if not ALL_AUDIO_DEVICES:
        print("No microphones or loopback devices found!")
    else:
        for i, mic in enumerate(ALL_AUDIO_DEVICES):
            print(f"  Index {i}: ID='{mic.id}', Name='{mic.name}'")
    
    # Get only loopback devices for the UI dropdown
    LOOPBACK_DEVICES = [mic for mic in ALL_AUDIO_DEVICES if "loopback" in mic.name.lower()]
    if not LOOPBACK_DEVICES:
        # If no explicit loopback found, include all devices for user selection
        print("No explicit 'loopback' devices found; including all audio devices for selection")
        LOOPBACK_DEVICES = ALL_AUDIO_DEVICES
    else:
        print(f"Found {len(LOOPBACK_DEVICES)} loopback devices")
    
    # Also try to find default speaker for info only
    try:
        default_speaker = sc.default_speaker()
        print(f"Default Speaker: {default_speaker.name}")
    except Exception as e:
        print(f"Could not determine default speaker: {e}")

except Exception as e:
    print(f"Error detecting audio devices: {e}. Please check your audio setup.")
    print("Ensure you have a 'Stereo Mix', 'What U Hear', or similar loopback input enabled in your sound settings.")
    sys.exit(1)

# --- Logging Setup ---
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - [%(threadName)s] %(message)s')

# --- Worker Thread ---
class Worker(QThread):
    status_update = pyqtSignal(str)
    translation_update = pyqtSignal(str, str) # (detected_lang, translated_text)
    error_signal = pyqtSignal(str)
    ready_signal = pyqtSignal()

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.stt_model = None
        self.translator = None
        self.translator_tokenizer = None
        self.audio_buffer = np.array([], dtype=np.float32)
        self.buffer_mutex = QMutex()
        self.data_available = QWaitCondition()
        self._is_running = True
        self._models_loaded = False
        self.audio_queue = queue.Queue()

        self.current_device_id = config['device_id']
        self.device_change_requested = False
        self.new_device_id = None
        self.device_mutex = QMutex()
        self.current_recorder = None # Store the SoundCard recorder object

    def stop(self):
        logging.info("Stopping worker thread...")
        self._is_running = False
        # Wake up the run loop if it's waiting on record()
        self.close_current_recorder() # Ensure recorder is closed on stop

    # Modified: 设备切换方法 - 提前关闭旧设备
    def switch_device(self, new_device_id):
        if not self._models_loaded:
            logging.warning("Cannot switch device - models not loaded yet")
            return False

        self.device_mutex.lock()
        current_id = self.current_device_id
        change_in_progress = self.device_change_requested
        self.device_mutex.unlock()

        if new_device_id == current_id and not change_in_progress:
            logging.info(f"Already using device ID: {new_device_id}, no change needed")
            return True
        
        # Avoid triggering multiple switches if one is already pending
        if change_in_progress:
            logging.warning(f"Device switch already requested to {self.new_device_id}, ignoring request for {new_device_id}")
            return False

        logging.info(f"Device switch requested from {current_id} to {new_device_id}")

        # 尝试立即关闭当前录音器
        self.close_current_recorder()

        # 设置标志，让run循环知道需要使用新设备
        self.device_mutex.lock()
        self.device_change_requested = True
        self.new_device_id = new_device_id
        self.device_mutex.unlock()

        return True

    def load_models(self):
        try:
            # --- Load STT Model (Whisper) ---
            self.status_update.emit("Loading STT model...")
            logging.info(f"Loading Whisper model: {self.config['whisper_model_size']} ({self.config['device']}, {self.config['compute_type']}) / Source Lang: {self.config['source_language']}")
            self.stt_model = faster_whisper.WhisperModel(
                self.config['whisper_model_size'],
                device=self.config['device'],
                compute_type=self.config['compute_type'],
            )
            logging.info("STT model loaded.")
            self.status_update.emit("STT model loaded.")

            # --- Load En->Zh Translation Model (Opus-MT) ---
            self.status_update.emit("Loading translation model...")
            logging.info(f"Loading Translator model: {TRANSLATOR_MODEL_ID}")
            cache_directory = "D:/hf_cache"
            self.translator_tokenizer = transformers.AutoTokenizer.from_pretrained(
                TRANSLATOR_MODEL_ID,
                cache_dir=cache_directory
            )
            self.translator = transformers.AutoModelForSeq2SeqLM.from_pretrained(
                TRANSLATOR_MODEL_ID,
                cache_dir=cache_directory
            ).to(DEVICE)
            logging.info("Translation model loaded.")

            self.status_update.emit("All models loaded. Ready.")
            self._models_loaded = True
            self.ready_signal.emit()
            return True

        except Exception as e:
            error_msg = f"Error loading models: {e}"
            logging.exception("Detailed error during model loading:")
            self.error_signal.emit(error_msg)
            self._models_loaded = False
            return False

    def start_audio_capture(self, device_id):
        try:
            # 清空队列中的旧数据
            while not self.audio_queue.empty():
                try:
                    self.audio_queue.get_nowait()
                except queue.Empty:
                    break
            
            logging.info(f"Starting audio capture from device ID: {device_id}")
            target_mic = sc.get_microphone(id=device_id, include_loopback=True)
            logging.info(f"Got microphone object for ID: {device_id}, Name: {target_mic.name}")
            
            # Important: Keep a reference to the recorder object itself
            self.current_recorder = target_mic.recorder(samplerate=SAMPLE_RATE, channels=None, blocksize=BLOCK_SIZE)
            
            # Update state *after* successfully getting the recorder
            self.device_mutex.lock()
            self.current_device_id = device_id
            self.device_change_requested = False # Reset flag now that we are starting
            self.new_device_id = None
            self.device_mutex.unlock()

            return self.current_recorder # Return the recorder object
            
        except Exception as e:
            error_msg = f"Failed to start audio capture from device ID {device_id}: {e}"
            logging.error(error_msg, exc_info=True)
            self.error_signal.emit(error_msg)
            self.current_recorder = None # Ensure recorder is None on failure
            return None

    def close_current_recorder(self):
        recorder_to_close = self.current_recorder
        if recorder_to_close:
            self.current_recorder = None # Set to None immediately to prevent reuse
            try:
                logging.info("Closing current audio recorder context...")
                # Soundcard recorder objects act as context managers
                # Calling __exit__ explicitly might work, or rely on garbage collection
                # Let's try explicitly closing if the object has a close method (it should)
                if hasattr(recorder_to_close, 'close') and callable(recorder_to_close.close):
                     recorder_to_close.close()
                     logging.info("Recorder closed successfully.")
                else:
                     # Fallback or just let garbage collector handle it
                     logging.warning("Recorder object does not have a close method? Letting GC handle it.")
            except Exception as e:
                logging.error(f"Error closing recorder: {e}", exc_info=True)

    def run(self):
        if not self.load_models():
            return

        target_channels = 1 # Whisper works best with mono

        processing_thread = QThread()
        processing_thread.run = self.process_audio_queue
        processing_thread.finished.connect(processing_thread.deleteLater)
        processing_thread.setObjectName("ProcessingThread")
        processing_thread.start()

        active_recorder = None # Keep track of the active recorder context

        try:
            while self._is_running:
                # --- Check for device change request ---
                self.device_mutex.lock()
                change_requested = self.device_change_requested
                target_device_id = self.new_device_id if change_requested else self.current_device_id
                self.device_mutex.unlock()

                # --- Start/Restart Capture Session ---
                # Only start if no recorder is active or a change was requested
                if active_recorder is None or change_requested:
                    # Ensure previous recorder is closed (already done in switch_device, but belt-and-suspenders)
                    if active_recorder:
                        try:
                            active_recorder.__exit__(None, None, None) # Exit context manager
                        except Exception as e:
                            logging.error(f"Error explicitly exiting previous recorder context: {e}")
                        active_recorder = None
                        
                    logging.info(f"Attempting to start capture for device: {target_device_id}")
                    recorder_obj = self.start_audio_capture(target_device_id)
                    if not recorder_obj:
                        logging.warning("Failed to start recorder, retrying after delay...")
                        time.sleep(1)
                        continue
                    
                    # Enter the recorder context
                    try:
                        active_recorder = recorder_obj.__enter__() # Enter context manager
                        logging.info(f"Successfully started recording from: {target_device_id}")
                    except Exception as e:
                        logging.error(f"Failed to enter recorder context: {e}", exc_info=True)
                        self.close_current_recorder() # Ensure cleanup
                        active_recorder = None
                        time.sleep(1)
                        continue

                # --- Inner Recording Loop ---
                if active_recorder:
                    try:
                        # Check again for device change *during* recording
                        self.device_mutex.lock()
                        needs_immediate_switch = self.device_change_requested
                        self.device_mutex.unlock()

                        if needs_immediate_switch:
                            logging.info("Device change requested during recording, exiting inner loop.")
                            # Exit the context manager, outer loop will handle restart
                            active_recorder.__exit__(None, None, None)
                            active_recorder = None
                            continue # Go back to the start of the outer loop

                        # Record audio chunk
                        indata = None
                        try:
                            # Use the entered context 'active_recorder' which is the microphone object
                            indata = active_recorder.record(numframes=BLOCK_SIZE)
                            if indata is not None and indata.shape[0] > 0:
                                logging.debug(f"Recorded {indata.shape[0]} frames. Shape: {indata.shape}")

                        except AttributeError:
                             # This might happen if the recorder context was exited unexpectedly
                             logging.error("Audio recorder context seems invalid, attempting recovery.")
                             active_recorder = None # Force restart in outer loop
                             continue
                        except Exception as record_error:
                            logging.error(f"Error during mic.record(): {record_error}", exc_info=True)
                            # Attempt to recover by restarting the capture in the outer loop
                            active_recorder.__exit__(None, None, None)
                            active_recorder = None
                            time.sleep(0.1)
                            continue

                        if not self._is_running:
                             break # Exit outer loop completely

                        if indata is not None and indata.shape[0] > 0:
                            # Process data (convert to float32, mono)
                            if indata.dtype != np.float32:
                                max_val = np.iinfo(indata.dtype).max
                                if max_val != 0:
                                    indata = indata.astype(np.float32) / max_val
                                else:
                                    indata = indata.astype(np.float32)

                            if indata.ndim > 1 and indata.shape[1] > target_channels:
                                mono_data = np.mean(indata, axis=1)
                            else:
                                mono_data = indata[:, 0] if indata.ndim > 1 else indata

                            self.audio_queue.put(mono_data)
                        else:
                            # No data or error, sleep briefly
                            time.sleep(0.01)

                    except Exception as inner_loop_error:
                        logging.error(f"Error in inner recording loop: {inner_loop_error}", exc_info=True)
                        try:
                            active_recorder.__exit__(None, None, None) # Ensure context exit on error
                        except Exception as exit_err:
                             logging.error(f"Error exiting recorder context during error handling: {exit_err}")
                        active_recorder = None
                        time.sleep(0.5) # Pause before outer loop restarts capture
                else:
                     # No active recorder, wait before trying again
                     time.sleep(0.1)


        except Exception as e:
            error_msg = f"Unhandled error in worker thread run loop: {e}"
            logging.error(error_msg, exc_info=True)
            self.error_signal.emit(error_msg)
        finally:
            logging.info("Worker thread run loop finishing.")
            # Ensure final cleanup
            if active_recorder:
                 try:
                     active_recorder.__exit__(None, None, None)
                 except Exception as e:
                     logging.error(f"Error exiting final recorder context: {e}")
            self.close_current_recorder() # Close any reference held by self.current_recorder
            self._is_running = False

            if processing_thread.isRunning():
                self.audio_queue.put(None)
                processing_thread.quit()
                processing_thread.wait(2000)
            logging.info("Worker thread finished.")

    def process_audio_queue(self):
        """Runs in a separate QThread to process audio without blocking capture."""
        logging.info("Audio processing thread started.")
        segment_buffer = np.array([], dtype=np.float32)

        while self._is_running or not self.audio_queue.empty():
            try:
                # Get audio data from queue, block if empty until new data or stop signal
                try:
                    # Timeout helps to periodically check self._is_running
                    new_data = self.audio_queue.get(block=True, timeout=0.1)
                    # Check for sentinel value to exit gracefully
                    if new_data is None:
                        logging.info("Processing thread received stop signal.")
                        break
                    segment_buffer = np.concatenate((segment_buffer, new_data))
                    self.audio_queue.task_done()
                except queue.Empty:
                    # If queue is empty and we are stopping, break the loop
                    if not self._is_running:
                        logging.debug("Processing queue empty and thread stopping.")
                        break
                    continue # Otherwise, continue waiting for data


                # Process when buffer reaches desired size
                if len(segment_buffer) >= BUFFER_SIZE:
                    logging.debug(f"Processing buffer of size {len(segment_buffer)}")
                    # --- STT ---
                    start_time = time.time()
                    # Check if models are loaded before transcribing
                    if not self._models_loaded or not self.stt_model:
                         logging.warning("STT model not ready, skipping transcription for this buffer.")
                         segment_buffer = np.array([], dtype=np.float32) # Clear buffer
                         continue
                         
                    segments, info = self.stt_model.transcribe(
                        segment_buffer,
                        language=self.config['source_language'], # None for auto-detect
                        beam_size=5,
                        vad_filter=False,
                    )
                    stt_text_parts = []
                    for segment in segments:
                        stt_text_parts.append(segment.text.strip())

                    stt_text = " ".join(stt_text_parts)
                    stt_duration = time.time() - start_time
                    logging.info(f"STT ({stt_duration:.2f}s): Detected language '{info.language}' with probability {info.language_probability:.2f}")
                    logging.info(f"STT Result: {stt_text}")

                    # Clear buffer after processing
                    segment_buffer = np.array([], dtype=np.float32)

                    # --- Translation (using Opus-MT En->Zh) ---
                    if stt_text:
                         # Check if models are loaded before translating
                        if not self._models_loaded or not self.translator or not self.translator_tokenizer:
                             logging.warning("Translation model not ready, skipping translation.")
                             continue
                             
                        self.status_update.emit("Translating...")
                        start_time = time.time()
                        whisper_lang_code = info.language

                        try:
                            logging.info(f"Translating from EN to {TARGET_LANGUAGE_CODE}")
                            inputs = self.translator_tokenizer(stt_text, return_tensors="pt", padding=True, truncation=True, max_length=512).to(DEVICE)
                            translated_tokens = self.translator.generate(
                                **inputs,
                                max_new_tokens=int(len(stt_text)*1.5)+50
                            )
                            translated_text = self.translator_tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]

                            trans_duration = time.time() - start_time
                            logging.info(f"Translation ({trans_duration:.2f}s): {translated_text}")
                            self.translation_update.emit('en', translated_text)
                            self.status_update.emit("Ready.")

                        except Exception as e:
                            logging.error(f"Translation error: {e}", exc_info=True) # Add traceback
                            self.error_signal.emit(f"Translation failed: {e}")
                            self.status_update.emit("Translation Error.")
                    else:
                         logging.debug("STT result was empty, skipping translation.")
                         self.status_update.emit("No speech detected.")

            except Exception as e:
                 logging.error(f"Error in processing loop: {e}", exc_info=True) # Add traceback
                 self.error_signal.emit(f"Processing error: {e}")
                 segment_buffer = np.array([], dtype=np.float32) # Clear buffer on error
                 time.sleep(1) # Avoid fast error loops


        # --- Process remaining buffer on exit (using Opus-MT) ---
        if len(segment_buffer) > SAMPLE_RATE * 0.5 and self._models_loaded and self.stt_model and self.translator and self.translator_tokenizer:
             # ... (Final buffer processing remains the same) ...
            logging.info(f"Processing remaining buffer of size {len(segment_buffer)}")
            try:
                segments, info = self.stt_model.transcribe(segment_buffer,
                                                          language=self.config['source_language'],
                                                          beam_size=5,
                                                          vad_filter=False)
                stt_text = " ".join([seg.text.strip() for seg in segments])
                if stt_text:
                    logging.info(f"Final STT: {stt_text}")
                    whisper_lang_code = info.language # Should be 'en'

                    logging.info(f"Final translation from EN to {TARGET_LANGUAGE_CODE}")
                    inputs = self.translator_tokenizer(stt_text, return_tensors="pt", padding=True, truncation=True, max_length=512).to(DEVICE)
                    translated_tokens = self.translator.generate(
                        **inputs,
                         max_new_tokens=int(len(stt_text)*1.5)+50
                    )
                    translated_text = self.translator_tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]
                    logging.info(f"Final Translation: {translated_text}")
                    self.translation_update.emit('en', translated_text)

            except Exception as e:
                 logging.error(f"Error processing final buffer: {e}", exc_info=True) # Add traceback


        logging.info("Audio processing thread finished.")

# --- Main Window ---
class MainWindow(QWidget):
    # 新增：向Worker发送设备更改的信号 - not needed with direct call
    # device_change_signal = pyqtSignal(str) 
    
    def __init__(self):
        super().__init__()
        self.worker = None
        self.worker_ready = False  # 跟踪Worker是否已准备好接收设备更改
        self.initUI()
        QTimer.singleShot(100, self.apply_settings_and_restart_worker)

    def initUI(self):
        self.setWindowTitle('实时翻译器')
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowStaysOnTopHint | Qt.WindowType.Tool)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)

        # --- Layout ---
        self.main_layout = QVBoxLayout() # Main vertical layout
        self.setLayout(self.main_layout)
        
        # 修改配置行布局为两行
        self.config_layout = QVBoxLayout()  # 改为垂直布局
        self.top_config_layout = QHBoxLayout()  # 顶部行
        self.bottom_config_layout = QHBoxLayout()  # 底部行
        
        # 顶部行 - 模型配置
        self.src_lang_label = QLabel("源语言:")
        self.src_lang_combo = QComboBox()
        self.src_lang_combo.addItems(["自动检测", "英语 (en)", "日语 (ja)"])
        self.top_config_layout.addWidget(self.src_lang_label)
        self.top_config_layout.addWidget(self.src_lang_combo)

        self.whisper_model_label = QLabel("识别模型:")
        self.whisper_model_combo = QComboBox()
        self.whisper_model_combo.addItems(["base", "small", "medium"])
        self.top_config_layout.addWidget(self.whisper_model_label)
        self.top_config_layout.addWidget(self.whisper_model_combo)

        self.translator_model_label = QLabel("翻译模型:")
        self.translator_model_combo = QComboBox()
        self.translator_model_combo.addItems(["Opus-MT (英语->中文)", "M2M100 (多语->中文)"])
        self.top_config_layout.addWidget(self.translator_model_label)
        self.top_config_layout.addWidget(self.translator_model_combo)

        self.apply_button = QPushButton("应用设置")
        self.apply_button.setToolTip("更改模型或语言需要应用设置 (重启工作线程)") # Add tooltip
        self.apply_button.clicked.connect(self.apply_settings_and_restart_worker)
        self.top_config_layout.addWidget(self.apply_button)
        
        # 底部行 - 设备控制
        self.device_label = QLabel("监听设备:")
        self.device_combo = QComboBox()
        # 填充设备列表
        for device in LOOPBACK_DEVICES: # Assuming LOOPBACK_DEVICES is populated at startup
            self.device_combo.addItem(f"{device.name}", device.id)
            
        if self.device_combo.count() == 0:
            self.device_combo.addItem("未找到任何音频设备")
            
        self.bottom_config_layout.addWidget(self.device_label)
        self.bottom_config_layout.addWidget(self.device_combo, 1)  # 设备下拉框可以占据更多空间
        
        # 新增：切换设备按钮
        self.switch_device_button = QPushButton("切换设备")
        self.switch_device_button.setToolTip("实时切换监听的音频设备 (无需重启)") # Add tooltip
        self.switch_device_button.clicked.connect(self.switch_audio_device)
        self.switch_device_button.setEnabled(False)  # 初始时禁用，等待Worker就绪
        self.bottom_config_layout.addWidget(self.switch_device_button)
        
        # 添加刷新按钮以重新扫描设备
        self.refresh_devices_button = QPushButton("刷新设备")
        self.refresh_devices_button.setToolTip("重新扫描系统音频设备") # Add tooltip
        self.refresh_devices_button.clicked.connect(self.refresh_audio_devices)
        self.bottom_config_layout.addWidget(self.refresh_devices_button)
        
        # 将布局组合到主配置布局
        self.config_layout.addLayout(self.top_config_layout)
        self.config_layout.addLayout(self.bottom_config_layout)
        
        # 将配置部分添加到主布局
        self.main_layout.addLayout(self.config_layout)
        
        # --- Translation Display Label ---
        self.translation_label = QLabel("初始化中...")
        self.translation_label.setWordWrap(True)
        self.translation_label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        self.translation_label.setStyleSheet("""
            QLabel {
                color: #E0E0E0;
                background-color: rgba(30, 30, 30, 0.8);
                border-radius: 5px;
                padding: 10px;
                font-size: 12pt;
            }
        """)
        self.main_layout.addWidget(self.translation_label, 1)


        # --- Styling ---
        palette = QPalette()
        palette.setColor(QPalette.ColorRole.Window, QColor(45, 45, 45, 200))
        palette.setColor(QPalette.ColorRole.WindowText, QColor(220, 220, 220))
        palette.setColor(QPalette.ColorRole.ButtonText, QColor(220, 220, 220))
        self.setPalette(palette)
        self.setAutoFillBackground(True)

        combo_and_button_style = """
            QComboBox, QPushButton {
                background-color: rgba(70, 70, 70, 0.8);
                color: #E0E0E0;
                border: 1px solid #555;
                border-radius: 3px;
                padding: 3px;
                min-height: 1.5em;
            }
            QComboBox::drop-down { border: none; }
            QComboBox::down-arrow { image: none; width: 0px; }
            QPushButton:hover { background-color: rgba(90, 90, 90, 0.9); }
            QPushButton:disabled { background-color: rgba(50, 50, 50, 0.7); color: #888; }
            QLabel { background-color: transparent; color: #E0E0E0; padding: 3px; }
        """
        # Apply style to all relevant widgets
        for widget in [self.src_lang_label, self.src_lang_combo, 
                       self.whisper_model_label, self.whisper_model_combo,
                       self.translator_model_label, self.translator_model_combo,
                       self.device_label, self.device_combo,
                       self.apply_button, self.switch_device_button, self.refresh_devices_button]:
            widget.setStyleSheet(combo_and_button_style)


        # 初始大小调整
        self.resize(750, 180)
        self.center()

    def center(self):
        # Center window on screen
        qr = self.frameGeometry()
        cp = self.screen().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())

    def mousePressEvent(self, event):
        # Allow moving the frameless window
        if event.button() == Qt.MouseButton.LeftButton:
            try: # Handle case where window is closed while dragging
                 self.drag_position = event.globalPosition().toPoint() - self.frameGeometry().topLeft()
                 event.accept()
            except AttributeError:
                 pass # Ignore if drag_position is not set or window is closing

    def mouseMoveEvent(self, event):
        # Move window when dragging
        if event.buttons() == Qt.MouseButton.LeftButton:
            try:
                self.move(event.globalPosition().toPoint() - self.drag_position)
                event.accept()
            except AttributeError:
                pass # Ignore if drag_position is not set

    def refresh_audio_devices(self):
        try:
            logging.info("Refreshing audio devices...")
            current_device_id = self.device_combo.currentData()
            
            self.device_combo.clear()
            
            # Re-fetch devices
            refreshed_devices = sc.all_microphones(include_loopback=True)
            loopback_devices = [mic for mic in refreshed_devices if "loopback" in mic.name.lower()]
            if not loopback_devices:
                loopback_devices = refreshed_devices
                
            # Populate combo box
            for device in loopback_devices:
                self.device_combo.addItem(f"{device.name}", device.id)
                
            if self.device_combo.count() == 0:
                self.device_combo.addItem("未找到任何音频设备")
                self.switch_device_button.setEnabled(False) # Disable switch if no devices
            elif self.worker_ready: # Re-enable switch if worker is ready and devices found
                 self.switch_device_button.setEnabled(True)

            # Try to re-select previous device
            if current_device_id:
                index = self.device_combo.findData(current_device_id)
                if index != -1:
                    self.device_combo.setCurrentIndex(index)
                        
            logging.info(f"Refreshed devices. Found {self.device_combo.count()} devices.")
            self.status_update("设备列表已刷新") # Use the status_update method
            
        except Exception as e:
            logging.error(f"Error refreshing devices: {e}", exc_info=True) # Add traceback
            self.show_error(f"刷新设备失败: {e}")

    def switch_audio_device(self):
        if not self.worker or not self.worker_ready:
            self.show_error("工作线程未就绪，无法切换设备")
            return
            
        device_id = self.device_combo.currentData()
        if not device_id or "未找到" in self.device_combo.currentText(): # Check for placeholder text
            self.show_error("请选择有效的音频设备")
            return
            
        logging.info(f"Sending device change request to worker: {device_id}")
        self.status_update("正在切换音频设备...")
        self.switch_device_button.setEnabled(False) # Disable button during switch
        
        # Call the worker's method directly
        success = self.worker.switch_device(device_id)
        
        # Re-enable button after a short delay to allow worker loop to process
        QTimer.singleShot(500, lambda: self.switch_device_button.setEnabled(self.worker_ready))
        
        if success:
             # Status update will happen when the worker actually starts the new device
             # self.status_update(f"切换请求已发送: {self.device_combo.currentText()}")
             pass
        else:
            self.show_error("设备切换请求失败 (可能已在切换中)")
            self.status_update("设备切换失败")

    def apply_settings_and_restart_worker(self):
        logging.info("Apply settings button clicked. Restarting worker...")
        self.translation_label.setText("应用新设置并重启中...")
        QApplication.processEvents()

        self.switch_device_button.setEnabled(False)
        self.worker_ready = False

        selected_src_lang_text = self.src_lang_combo.currentText()
        selected_whisper_model = self.whisper_model_combo.currentText()
        selected_translator_text = self.translator_model_combo.currentText()
        
        selected_device_id = self.device_combo.currentData()
        
        if not selected_device_id or "未找到" in self.device_combo.currentText():
            self.show_error("请选择有效的音频设备以启动")
            return

        # Stop previous worker
        if self.worker and self.worker.isRunning():
            logging.info("Stopping previous worker...")
            self.worker.stop()
            if not self.worker.wait(3000):
                 logging.warning("Worker thread did not stop gracefully.")
            # Disconnect signals safely
            try: self.worker.status_update.disconnect() 
            except TypeError: pass
            try: self.worker.translation_update.disconnect()
            except TypeError: pass
            try: self.worker.error_signal.disconnect()
            except TypeError: pass
            try: self.worker.ready_signal.disconnect()
            except TypeError: pass
            # Try explicitly deleting later to help with resource release
            self.worker.deleteLater() 
            self.worker = None
            logging.info("Previous worker stopped and scheduled for deletion.")

        # Create config
        source_language_map = {"自动检测": None, "英语 (en)": "en", "日语 (ja)": "ja"}
        translator_type_map = {"Opus-MT (英语->中文)": "opus-mt-en-zh", "M2M100 (多语->中文)": "m2m100"}
        
        logging.info(f"Starting with device ID: {selected_device_id}")
        
        config = {
            "whisper_model_size": selected_whisper_model,
            "source_language": source_language_map.get(selected_src_lang_text),
            "translator_type": translator_type_map.get(selected_translator_text, "opus-mt-en-zh"),
            "device_id": selected_device_id
        }
        config["device"] = DEVICE
        config["compute_type"] = COMPUTE_TYPE

        self.start_worker(config)

    def start_worker(self, config):
        logging.info(f"Attempting to start worker with config: {config}")

        if config.get("device_id") is None:
             self.show_error("未找到合适的音频设备ID。请检查系统设置和设备选择。")
             logging.error("Cannot start worker: device_id is None.")
             return

        try:
            self.worker = Worker(config)
            self.worker.setObjectName("WorkerThread")
            # Connect signals
            self.worker.status_update.connect(self.update_status_display) # Use dedicated display slot
            self.worker.translation_update.connect(self.update_translation)
            self.worker.error_signal.connect(self.show_error)
            self.worker.ready_signal.connect(self.on_worker_ready)
            # Connect finished signal to potentially re-enable buttons if worker stops unexpectedly
            self.worker.finished.connect(self.on_worker_finished) 
            
            self.worker.start()
            logging.info("New worker started.")
        except Exception as e:
            logging.exception("Failed to create or start worker thread!")
            self.show_error(f"启动工作线程失败: {e}")
            self.switch_device_button.setEnabled(False) # Ensure button is disabled on failure
            self.worker_ready = False

    def on_worker_ready(self):
        logging.info("Worker thread reported ready")
        self.worker_ready = True
        # Enable switch button only if there are valid devices
        if self.device_combo.count() > 0 and "未找到" not in self.device_combo.currentText():
            self.switch_device_button.setEnabled(True)
        self.status_update("就绪。可以切换设备。") # Use the status_update method

    # New slot to handle worker finishing unexpectedly
    def on_worker_finished(self):
         logging.warning("Worker thread finished.")
         self.worker_ready = False
         self.switch_device_button.setEnabled(False)
         # Optionally show a message to the user
         # self.status_update("工作线程已停止")

    # Renamed from update_status to avoid conflict
    def update_status_display(self, message):
        # This method updates the translation label temporarily with status messages
        logging.info(f"Status Display Update: {message}")
        # Avoid overwriting actual translations with intermediate statuses like "Ready" or "Translating"
        # Maybe use tooltips or a dedicated status bar later?
        # For now, only show critical status or errors persistently
        if "Error" in message or "failed" in message or "停止" in message:
             # Keep error messages visible for longer
             self.translation_label.setText(f"<i>({message})</i>")
        # else:
             # Briefly show non-error status updates? Might be too noisy.
             # QTimer.singleShot(2500, lambda current_text=self.translation_label.text(), msg=message: \
             #                   self.translation_label.setText(current_text) if f"<i>({msg})</i>" == self.translation_label.text() else None)
             # self.translation_label.setText(f"<i>({message})</i>")
             

    # New method for non-persistent status updates (e.g., "设备列表已刷新")
    def status_update(self, message):
         # This provides brief feedback without overwriting the main translation display
         logging.info(f"Status Update: {message}")
         # Example: Use window title or a tooltip on a status icon (not implemented here)
         self.setWindowTitle(f'实时翻译器 - {message}')
         # Reset title after a delay
         QTimer.singleShot(3000, lambda: self.setWindowTitle('实时翻译器') if self.windowTitle().endswith(message) else None)

    def update_translation(self, source_lang, translated_text):
        # Keep label updated with the latest full translation and source lang
        display_text = f"[{source_lang.upper()}] → [ZH]\n{translated_text}"
        self.translation_label.setText(display_text)

    def show_error(self, error_message):
        logging.error(f"GUI Error: {error_message}")
        # Use the status display method to show the error more persistently
        self.update_status_display(f"<font color='red'><b>错误:</b> {error_message}</font>")

    def closeEvent(self, event):
        logging.info("Close event received. Stopping worker...")
        if self.worker:
            self.worker.stop()
            self.worker.wait() # Wait for thread to finish cleanly
        event.accept()

# --- Main Execution ---
if __name__ == '__main__':
    try:
        QThread.currentThread().setObjectName("MainThread")
        app = QApplication(sys.argv)
        main_window = MainWindow()
        main_window.show()
        sys.exit(app.exec())
    except Exception as e:
        logging.exception("Unhandled exception in main execution block.")
        sys.exit(1) 