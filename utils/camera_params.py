# type: ignore
"""
参数推荐模块 - 使用整张画面的平均亮度判断环境光线
"""

from config import *


class CameraParamRecommender:
    def recommend(self, features):
        """
        根据环境亮度和荧光棒运动速度推荐拍摄参数
        features: analyzer返回的特征字典
        """
        params = {}

        # 使用整张画面的平均亮度来判断环境光线
        brightness = features.get('avg_frame_brightness', 0)
        velocity = features.get('velocity', 0)

        # ISO推荐（基于环境亮度）
        for rule in ISO_RULES:
            if rule['min'] <= brightness < rule['max']:
                params['iso'] = rule['value']
                break
        if 'iso' not in params:
            params['iso'] = 'AUTO'

        # 曝光补偿推荐（基于环境亮度）
        for rule in EV_RULES:
            if rule['min'] <= brightness < rule['max']:
                params['ev'] = rule['value']
                break
        if 'ev' not in params:
            params['ev'] = 0.0

        # 光圈推荐（单人固定大光圈）
        params['aperture'] = 'f/2.8'

        # 快门速度推荐（基于运动速度）
        for rule in SHUTTER_RULES:
            if rule['min'] <= velocity < rule['max']:
                params['shutter'] = rule['value']
                break
        if 'shutter' not in params:
            params['shutter'] = 'AUTO'

        return params