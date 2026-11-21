# config.py

# 摄像头ID
CAMERA_ID = 0

# 目标分辨率
TARGET_WIDTH = 640
TARGET_HEIGHT = 360

# YOLO模型路径
MODEL_PATH = 'models/best.pt'

# 检测置信度 - 降到0.15！！！
CONFIDENCE_THRESHOLD = 0.15

# 低光增强
ENABLE_LOWLIGHT_ENHANCE = True
LOWLIGHT_THRESHOLD = 80

# 人体站位参数
SHOULDER_Y_RATIO = 1/3      # 肩膀在1/3高度
FOOT_Y_RATIO = 4/5           # 脚在4/5高度（离底部1/5）
CENTER_X_RATIO = 1/2         # 水平居中

# ISO规则表
ISO_RULES = [
    {'min': 0, 'max': 50, 'value': 1600},
    {'min': 50, 'max': 100, 'value': 800},
    {'min': 100, 'max': 150, 'value': 400},
    {'min': 150, 'max': 255, 'value': 200},
]

# 光圈规则表（单人模式）
APERTURE_RULES = [
    {'min': 0, 'max': 1, 'value': 'f/2.8'},
    {'min': 1, 'max': 2, 'value': 'f/4.0'},
    {'min': 2, 'max': 10, 'value': 'f/5.6'},
]

# 曝光补偿规则表
EV_RULES = [
    {'min': 0, 'max': 60, 'value': 1.0},
    {'min': 60, 'max': 200, 'value': 0.0},
    {'min': 200, 'max': 255, 'value': -1.0},
]

# 快门速度规则表
SHUTTER_RULES = [
    {'min': 0, 'max': 10, 'value': '1/60'},
    {'min': 10, 'max': 30, 'value': '1/125'},
    {'min': 30, 'max': 50, 'value': '1/250'},
    {'min': 50, 'max': 999, 'value': '1/500'},
]