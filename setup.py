"""
py2app 打包配置
"""
from setuptools import setup

APP = ['voice_input.py']
DATA_FILES = []
OPTIONS = {
    'argv_emulation': False,
    'iconfile': None,
    'plist': {
        'CFBundleName': '秒输语音输入',
        'CFBundleDisplayName': '秒输语音输入',
        'CFBundleIdentifier': 'com.miaoshu.input',
        'CFBundleVersion': '1.0.0',
        'CFBundleShortVersionString': '1.0.0',
        'LSUIElement': True,  # 无 Dock 图标
        'NSMicrophoneUsageDescription': '需要麦克风权限进行语音识别',
        'NSAppleEventsUsageDescription': '需要辅助功能权限输入文字',
    },
    'packages': ['sherpa_onnx', 'sounddevice', 'pynput', 'numpy'],
}

setup(
    app=APP,
    data_files=DATA_FILES,
    options={'py2app': OPTIONS},
    setup_requires=['py2app'],
)
