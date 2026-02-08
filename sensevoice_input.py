#!/usr/bin/env python3
"""
秒输 语音输入工具 - macOS
按住 Option 键说话，松开自动识别并输入
"""

import sherpa_onnx
import sounddevice as sd
import numpy as np
import subprocess
import threading
import os
from pynput import keyboard

# 配置
MODEL_DIR = os.path.expanduser("~/Models/ASR/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17")
SAMPLE_RATE = 16000
USE_INT8 = True  # 使用量化模型，更快

class 秒输Input:
    def __init__(self):
        print("正在加载 秒输 模型...")
        self.recognizer = self._create_recognizer()
        print("模型加载完成！")

        self.is_recording = False
        self.audio_data = []
        self.stream = None

    def _create_recognizer(self):
        model_file = "model.int8.onnx" if USE_INT8 else "model.onnx"

        config = sherpa_onnx.OfflineRecognizerConfig(
            model_config=sherpa_onnx.OfflineModelConfig(
                sense_voice=sherpa_onnx.Offline秒输ModelConfig(
                    model=f"{MODEL_DIR}/{model_file}",
                    language="auto",  # 自动检测语言
                    use_itn=True,  # 使用逆文本正则化
                ),
                tokens=f"{MODEL_DIR}/tokens.txt",
                num_threads=4,
                provider="cpu",
            ),
        )
        return sherpa_onnx.OfflineRecognizer(config)

    def _audio_callback(self, indata, frames, time, status):
        if self.is_recording:
            self.audio_data.append(indata.copy())

    def start_recording(self):
        if self.is_recording:
            return
        self.is_recording = True
        self.audio_data = []
        self.stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype=np.float32,
            callback=self._audio_callback
        )
        self.stream.start()
        print("🎤 开始录音...")

    def stop_recording(self):
        if not self.is_recording:
            return ""

        self.is_recording = False
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None

        if not self.audio_data:
            print("没有录到音频")
            return ""

        print("🔄 正在识别...")
        audio = np.concatenate(self.audio_data, axis=0).flatten()

        # 创建流并识别
        stream = self.recognizer.create_stream()
        stream.accept_waveform(SAMPLE_RATE, audio)
        self.recognizer.decode_stream(stream)

        text = stream.result.text.strip()
        print(f"✅ 识别结果: {text}")
        return text

    def type_text(self, text):
        """使用 AppleScript 输入文字"""
        if not text:
            return
        # 转义特殊字符
        escaped = text.replace('\\', '\\\\').replace('"', '\\"')
        script = f'tell application "System Events" to keystroke "{escaped}"'
        subprocess.run(["osascript", "-e", script], check=True)
        print(f"⌨️ 已输入: {text}")


def main():
    voice_input = 秒输Input()

    print("\n" + "="*50)
    print("秒输 语音输入工具")
    print("="*50)
    print("按住 Option(⌥) 键说话，松开自动识别并输入")
    print("按 Ctrl+C 退出")
    print("="*50 + "\n")

    def on_press(key):
        if key == keyboard.Key.alt:
            voice_input.start_recording()

    def on_release(key):
        if key == keyboard.Key.alt:
            text = voice_input.stop_recording()
            if text:
                voice_input.type_text(text)

    with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
        listener.join()


if __name__ == "__main__":
    main()
