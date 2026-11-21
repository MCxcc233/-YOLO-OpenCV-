# type: ignore

"""
低光图像增强模块
来源：https://github.com/PaulMahdizadeh123/LowLightEnh
简化版 - 只保留核心功能
"""

import cv2
import numpy as np


def enhance_lowlight_image(img):
    """
    对低光图像进行增强
    输入：BGR图像
    输出：增强后的BGR图像
    """
    # 转换为Lab色彩空间
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    # 对亮度通道进行CLAHE增强
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l_enhanced = clahe.apply(l)

    # 合并通道
    lab_enhanced = cv2.merge([l_enhanced, a, b])
    enhanced = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)

    # 自适应伽马校正
    gamma = calculate_adaptive_gamma(img)
    enhanced = adjust_gamma(enhanced, gamma)

    return enhanced


def calculate_adaptive_gamma(img):
    """根据图像亮度计算自适应伽马值"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mean_brightness = np.mean(gray)

    if mean_brightness < 50:
        return 1.5  # 暗图用大伽马
    elif mean_brightness < 100:
        return 1.2
    else:
        return 1.0


def adjust_gamma(image, gamma=1.0):
    """伽马校正"""
    invGamma = 1.0 / gamma
    table = np.array([(i / 255.0) ** invGamma * 255
                      for i in range(256)]).astype("uint8")
    return cv2.LUT(image, table)


# 更高级的增强（可选，如果需要）
def advanced_enhance(img):
    """
    更复杂的低光增强算法
    结合去雾和自适应直方图均衡
    """
    # 暗通道先验去雾（简化版）
    img_f = img.astype(np.float32) / 255.0

    # 自适应直方图均衡
    img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])
    img_eq = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)

    return img_eq