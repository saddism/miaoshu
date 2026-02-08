#!/usr/bin/env python3
"""
秒输 语音输入工具
按住快捷键说话，松开自动识别并粘贴到当前输入框

配置选项（~/.miaoshu_config.json）:
- hotkey: 快捷键，如 "cmd_r", "ctrl_alt", "f13"
- language: 语言，如 "auto", "zh", "en", "ja", "ko", "yue"
- use_itn: 是否启用逆文本规范化（数字转换等）
- auto_punctuation: 是否自动添加标点符号
- custom_punctuation_map: 自定义标点映射
- hotwords: 热词字典，用于替换特定词汇
- hotword_boost: 热词权重增强（如果有模型支持）
- sample_rate: 采样率
- num_threads: 推理线程数
"""

import sherpa_onnx
import sounddevice as sd
import numpy as np
import subprocess
import threading
import queue
import sys
import re
from pynput import keyboard

# macOS 原生 UI
import AppKit
from AppKit import (
    NSApplication, NSWindow, NSTextField, NSColor, NSFont,
    NSWindowStyleMaskBorderless, NSBackingStoreBuffered,
    NSFloatingWindowLevel, NSScreen, NSView, NSBezierPath,
    NSMakeRect, NSTextAlignmentCenter
)
from PyObjCTools import AppHelper
import json
import os

# 配置
DEFAULT_CONFIG = {
    # 基本设置
    "hotkey": "ctrl_alt",  # 可选: cmd_r, ctrl_alt, ctrl_shift, f13, f14, f15
    "language": "auto",    # 可选: auto, zh, en, ja, ko, yue
    "sample_rate": 16000,
    "num_threads": 4,
    
    # 语音识别优化
    "use_itn": True,       # 逆文本规范化（将"一二三"转为"123"）
    "auto_punctuation": True,  # 自动添加标点符号
    
    # 热词/常用词替换（后处理）
    # 用于替换识别错误或不准确的词汇
    "hotwords": {
        # 示例：将特定词汇替换为正确的形式
        # "人名": "正确的名字",
        # "公司名": "正确的公司名",
    },
    
    # 标点符号自定义映射
    # 用于将识别出的标点替换为自定义标点
    "punctuation_map": {
        # "，": ",",  # 将中文逗号转为英文逗号
        # "。": ".",  # 将中文句号转为英文句号
    },
    
    # 文本后处理规则（正则表达式替换）
    "text_rules": [
        # 示例：删除多余的空格
        # {"pattern": "  +", "replace": " "},
        # 示例：转换全角数字为半角
        # {"pattern": "[０-９]", "replace": 对应函数},
    ],
    
    # 模型路径（可选，用于覆盖默认路径）
    "model_dir": "",
}

CONFIG_FILE = os.path.expanduser("~/.miaoshu_config.json")
# 默认模型路径，可通过配置文件覆盖
MODEL_DIR = os.path.expanduser("~/Models/ASR/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17")


class Config:
    """配置管理"""
    def __init__(self):
        self.data = DEFAULT_CONFIG.copy()
        self.load()
    
    def load(self):
        """从配置文件加载"""
        if os.path.exists(CONFIG_FILE):
            try:
                with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
                    user_config = json.load(f)
                    # 递归合并配置
                    self._deep_update(self.data, user_config)
                print(f"已加载配置: {CONFIG_FILE}")
            except Exception as e:
                print(f"加载配置失败: {e}，使用默认配置")
        else:
            # 创建默认配置文件
            self.save()
            print(f"已创建默认配置文件: {CONFIG_FILE}")
    
    def save(self):
        """保存配置到文件"""
        try:
            with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
                json.dump(self.data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"保存配置失败: {e}")
    
    def _deep_update(self, base_dict, update_dict):
        """递归更新字典"""
        for key, value in update_dict.items():
            if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
                self._deep_update(base_dict[key], value)
            else:
                base_dict[key] = value
    
    def get(self, key, default=None):
        """获取配置项"""
        keys = key.split('.')
        value = self.data
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        return value
    
    def set(self, key, value):
        """设置配置项"""
        keys = key.split('.')
        target = self.data
        for k in keys[:-1]:
            if k not in target:
                target[k] = {}
            target = target[k]
        target[keys[-1]] = value
        self.save()


class TextProcessor:
    """文本后处理器 - 处理热词替换、标点映射等"""
    
    def __init__(self, config: Config):
        self.config = config
        self.hotwords = config.get('hotwords', {})
        self.punctuation_map = config.get('punctuation_map', {})
        self.auto_punctuation = config.get('auto_punctuation', True)
    
    def process(self, text: str) -> str:
        """处理识别后的文本"""
        if not text:
            return text
        
        # 1. 热词替换（最长匹配优先）
        text = self._apply_hotwords(text)
        
        # 2. 标点符号映射
        text = self._apply_punctuation_map(text)
        
        # 3. 应用自定义正则规则
        text = self._apply_text_rules(text)
        
        # 4. 标点符号优化（如果启用）
        if self.auto_punctuation:
            text = self._optimize_punctuation(text)
        
        return text.strip()
    
    def _apply_hotwords(self, text: str) -> str:
        """应用热词替换"""
        if not self.hotwords:
            return text
        
        # 按长度降序排序，确保长词优先匹配
        sorted_hotwords = sorted(
            self.hotwords.items(), 
            key=lambda x: len(x[0]), 
            reverse=True
        )
        
        for old_word, new_word in sorted_hotwords:
            # 使用正则表达式进行整词匹配（支持中文分词边界）
            # 在中文中，我们直接替换
            text = text.replace(old_word, new_word)
        
        return text
    
    def _apply_punctuation_map(self, text: str) -> str:
        """应用标点符号映射"""
        if not self.punctuation_map:
            return text
        
        for old_punct, new_punct in self.punctuation_map.items():
            text = text.replace(old_punct, new_punct)
        
        return text
    
    def _apply_text_rules(self, text: str) -> str:
        """应用自定义文本规则"""
        rules = self.config.get('text_rules', [])
        for rule in rules:
            try:
                pattern = rule.get('pattern', '')
                replace = rule.get('replace', '')
                if pattern:
                    text = re.sub(pattern, replace, text)
            except re.error as e:
                print(f"正则规则错误: {e}")
        return text
    
    def _optimize_punctuation(self, text: str) -> str:
        """优化标点符号（去除重复标点、修正间距等）"""
        # 去除重复的标点符号
        text = re.sub(r'([，。！？、；：""''（）【】])\1+', r'\1', text)
        
        # 修正中英文标点间的空格
        text = re.sub(r'([，。！？、；：]) ', r'\1', text)
        text = re.sub(r' ([，。！？、；：])', r'\1', text)
        
        return text
    
    def add_hotword(self, old_word: str, new_word: str, save: bool = True):
        """动态添加热词"""
        self.hotwords[old_word] = new_word
        if save:
            self.config.set('hotwords', self.hotwords)
    
    def remove_hotword(self, old_word: str, save: bool = True):
        """删除热词"""
        if old_word in self.hotwords:
            del self.hotwords[old_word]
            if save:
                self.config.set('hotwords', self.hotwords)


class RoundedView(NSView):
    """圆角背景视图"""
    def drawRect_(self, rect):
        # 半透明黑色背景
        NSColor.colorWithCalibratedRed_green_blue_alpha_(0, 0, 0, 0.8).setFill()
        path = NSBezierPath.bezierPathWithRoundedRect_xRadius_yRadius_(rect, 15, 15)
        path.fill()


class FloatingIndicator:
    """屏幕底部浮动提示条"""
    def __init__(self):
        self.window = None
        self.label = None
        self._setup_window()

    def _setup_window(self):
        # 获取屏幕尺寸
        screen = NSScreen.mainScreen().frame()
        width, height = 200, 40
        x = (screen.size.width - width) / 2
        y = 80  # 距离底部 80 像素

        # 创建无边框窗口
        self.window = NSWindow.alloc().initWithContentRect_styleMask_backing_defer_(
            NSMakeRect(x, y, width, height),
            NSWindowStyleMaskBorderless,
            NSBackingStoreBuffered,
            False
        )
        self.window.setLevel_(NSFloatingWindowLevel)
        self.window.setOpaque_(False)
        self.window.setBackgroundColor_(NSColor.clearColor())
        self.window.setIgnoresMouseEvents_(True)

        # 圆角背景
        bg_view = RoundedView.alloc().initWithFrame_(NSMakeRect(0, 0, width, height))
        self.window.setContentView_(bg_view)

        # 文字标签
        self.label = NSTextField.alloc().initWithFrame_(NSMakeRect(0, 8, width, 24))
        self.label.setStringValue_("🎤 正在录音...")
        self.label.setBezeled_(False)
        self.label.setDrawsBackground_(False)
        self.label.setEditable_(False)
        self.label.setSelectable_(False)
        self.label.setTextColor_(NSColor.whiteColor())
        self.label.setFont_(NSFont.systemFontOfSize_(15))
        self.label.setAlignment_(NSTextAlignmentCenter)
        bg_view.addSubview_(self.label)

    def show(self, text="🎤 正在录音..."):
        def _show():
            self.label.setStringValue_(text)
            self.window.orderFront_(None)
        AppKit.NSApp.activateIgnoringOtherApps_(False)
        AppHelper.callAfter(_show)

    def hide(self):
        def _hide():
            self.window.orderOut_(None)
        AppHelper.callAfter(_hide)

    def update_text(self, text):
        def _update():
            self.label.setStringValue_(text)
        AppHelper.callAfter(_update)


class 秒输Input:
    def __init__(self, indicator=None, config: Config = None):
        self.is_recording = False
        self.audio_queue = queue.Queue()
        self.audio_data = []
        self.recognizer = None
        self.indicator = indicator
        self.config = config or Config()
        self.text_processor = TextProcessor(self.config)
        self.model_dir = self.config.get('model_dir') or MODEL_DIR

    def init_model(self):
        """初始化 秒输 模型"""
        print("正在加载 秒输 模型...")
        print(f"模型路径: {self.model_dir}")
        print(f"语言设置: {self.config.get('language', 'auto')}")
        print(f"ITN 启用: {self.config.get('use_itn', True)}")

        self.recognizer = sherpa_onnx.OfflineRecognizer.from_sense_voice(
            model=f"{self.model_dir}/model.onnx",
            tokens=f"{self.model_dir}/tokens.txt",
            num_threads=self.config.get('num_threads', 4),
            provider="cpu",
            language=self.config.get('language', 'auto'),
            use_itn=self.config.get('use_itn', True),
        )
        print("模型加载完成！")
        
        # 显示热词配置信息
        hotwords = self.config.get('hotwords', {})
        if hotwords:
            print(f"已加载 {len(hotwords)} 个热词:")
            for old, new in list(hotwords.items())[:5]:
                print(f"  '{old}' -> '{new}'")
            if len(hotwords) > 5:
                print(f"  ... 等共 {len(hotwords)} 个")

    def start_recording(self):
        """开始录音"""
        if self.is_recording:
            return

        self.is_recording = True
        self.audio_data = []
        print("🎤 开始录音...")

        # 显示浮动提示
        if self.indicator:
            self.indicator.show("🎤 正在录音...")

        def audio_callback(indata, frames, time, status):
            if self.is_recording:
                self.audio_data.append(indata.copy())

        self.stream = sd.InputStream(
            samplerate=self.config.get('sample_rate', 16000),
            channels=1,
            dtype=np.float32,
            callback=audio_callback
        )
        self.stream.start()

    def stop_recording(self):
        """停止录音并识别"""
        if not self.is_recording:
            return

        self.is_recording = False
        self.stream.stop()
        self.stream.close()
        print("⏹️  停止录音，正在识别...")

        # 更新提示
        if self.indicator:
            self.indicator.update_text("🔄 识别中...")

        # 合并音频数据
        if not self.audio_data:
            print("没有录到音频")
            if self.indicator:
                self.indicator.hide()
            return

        audio = np.concatenate(self.audio_data, axis=0).flatten()

        # 识别
        stream = self.recognizer.create_stream()
        stream.accept_waveform(self.config.get('sample_rate', 16000), audio)
        self.recognizer.decode_stream(stream)

        raw_text = stream.result.text.strip()
        
        # 应用后处理（热词替换等）
        text = self.text_processor.process(raw_text)
        
        # 如果处理后文本有变化，显示原始文本
        if text != raw_text:
            print(f"📝 原始识别: {raw_text}")

        # 隐藏提示
        if self.indicator:
            self.indicator.hide()

        if text:
            print(f"📝 识别结果: {text}")
            self.paste_text(text)
        else:
            print("未识别到文字")

    def paste_text(self, text):
        """粘贴文字到当前输入框"""
        # 复制到剪贴板
        subprocess.run(['pbcopy'], input=text.encode('utf-8'), check=True)

        # 模拟 Cmd+V 粘贴
        subprocess.run([
            'osascript', '-e',
            'tell application "System Events" to keystroke "v" using command down'
        ], check=True)
        print("✅ 已粘贴")


def get_hotkey_config(config: Config):
    """根据配置获取快捷键监听参数"""
    hotkey = config.get('hotkey', 'ctrl_alt')
    
    hotkey_configs = {
        'cmd_r': {
            'description': '右 Command',
            'needs_both': False,
            'key': keyboard.Key.cmd_r
        },
        'ctrl_alt': {
            'description': 'Control + Option',
            'needs_both': True,
            'keys': [keyboard.Key.ctrl, keyboard.Key.ctrl_l, keyboard.Key.alt, keyboard.Key.alt_l]
        },
        'ctrl_shift': {
            'description': 'Control + Shift',
            'needs_both': True,
            'keys': [keyboard.Key.ctrl, keyboard.Key.ctrl_l, keyboard.Key.shift, keyboard.Key.shift_l]
        },
        'f13': {
            'description': 'F13',
            'needs_both': False,
            'key': keyboard.Key.f13
        },
        'f14': {
            'description': 'F14',
            'needs_both': False,
            'key': keyboard.Key.f14
        },
        'f15': {
            'description': 'F15',
            'needs_both': False,
            'key': keyboard.Key.f15
        },
    }
    
    return hotkey_configs.get(hotkey, hotkey_configs['ctrl_alt'])


def main():
    # 加载配置
    config = Config()
    hotkey_config = get_hotkey_config(config)
    
    # 初始化 NSApplication
    NSApplication.sharedApplication()

    # 创建浮动提示
    indicator = FloatingIndicator()

    # 创建语音输入
    app = 秒输Input(indicator=indicator, config=config)
    app.init_model()

    print("\n" + "="*50)
    print("秒输 语音输入工具已启动")
    print(f"快捷键: 按住 {hotkey_config['description']} 说话，松开自动识别")
    print(f"配置文件: {CONFIG_FILE}")
    print("按 Ctrl+C 退出")
    print("="*50 + "\n")

    # 根据快捷键配置设置监听
    if hotkey_config['needs_both']:
        # 组合键模式（如 Ctrl+Option）
        key1_pressed = False
        key2_pressed = False
        keys = hotkey_config['keys']
        
        def on_press(key):
            nonlocal key1_pressed, key2_pressed
            
            if key in keys[:2]:  # 第一个键（ctrl）
                key1_pressed = True
            elif key in keys[2:]:  # 第二个键（alt/shift）
                key2_pressed = True
            
            if key1_pressed and key2_pressed and not app.is_recording:
                app.start_recording()
        
        def on_release(key):
            nonlocal key1_pressed, key2_pressed
            
            if key in keys[:2]:
                key1_pressed = False
                if app.is_recording:
                    app.stop_recording()
            elif key in keys[2:]:
                key2_pressed = False
                if app.is_recording:
                    app.stop_recording()
    else:
        # 单键模式
        target_key = hotkey_config['key']
        
        def on_press(key):
            if key == target_key and not app.is_recording:
                app.start_recording()
        
        def on_release(key):
            if key == target_key and app.is_recording:
                app.stop_recording()

    # 在后台线程监听键盘
    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.start()

    # 运行主循环
    try:
        AppHelper.runConsoleEventLoop()
    except KeyboardInterrupt:
        print("\n再见！")
        listener.stop()


if __name__ == "__main__":
    main()
