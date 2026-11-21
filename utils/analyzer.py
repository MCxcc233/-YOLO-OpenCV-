# type: ignore
"""
分析模块 - 分析人体位置和荧光棒特征
"""

import cv2
import numpy as np
from config import ENABLE_LOWLIGHT_ENHANCE, LOWLIGHT_THRESHOLD
from utils.lowlight_enhance import enhance_lowlight_image


class StickAnalyzer:
    def __init__(self):
        self.prev_centers = []  # 用于计算运动速度

    def analyze_frame(self, frame, persons, sticks):
        """
        分析人体位置和荧光棒特征
        """
        # 人体分析
        person_features = self._analyze_persons(persons, frame.shape)

        # 荧光棒分析
        stick_features = self._analyze_sticks(sticks, frame)

        # 计算整张画面的平均亮度（用于环境光判断）
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        avg_frame_brightness = np.mean(gray) if gray.size > 0 else 0

        # 合并特征
        features = {
            'has_person': person_features.get('has_person', False),
            'person_center': person_features.get('person_center'),
            'person_shoulder': person_features.get('person_shoulder'),
            'person_foot': person_features.get('person_foot'),
            'person_height': person_features.get('person_height', 0),
            'target_x': person_features.get('target_x', 0),
            'target_shoulder_y': person_features.get('target_shoulder_y', 0),
            'target_foot_y': person_features.get('target_foot_y', 0),
            'offset_x': person_features.get('offset_x', 0),
            'offset_shoulder_y': person_features.get('offset_shoulder_y', 0),
            'offset_foot_y': person_features.get('offset_foot_y', 0),
            'score': person_features.get('score', 0),
            'is_centered': person_features.get('is_centered', False),
            'has_stick': stick_features.get('has_stick', False),
            'sticks': stick_features.get('sticks', []),
            'stick_count': len(sticks),
            'avg_brightness': stick_features.get('avg_brightness', 0),  # 荧光棒区域的平均亮度
            'avg_frame_brightness': avg_frame_brightness,  # 整张画面的平均亮度
            'avg_length': stick_features.get('avg_length', 0),
            'velocity': stick_features.get('velocity', 0),
            'need_enhance': self._check_need_enhance(frame)
        }

        return features

    def _analyze_persons(self, persons, frame_shape):
        """分析人体位置"""
        h, w = frame_shape[:2]

        if not persons:
            return {
                'has_person': False,
                'person_center': None,
                'person_shoulder': None,
                'person_foot': None,
                'person_height': 0,
                'offset_x': 0,
                'offset_shoulder_y': 0,
                'offset_foot_y': 0,
                'score': 0,
                'is_centered': False
            }

        # 取第一个人
        person = persons[0]

        # 理想位置
        target_x = w // 2
        target_shoulder_y = int(h * 1 / 3)
        target_foot_y = int(h * 4 / 5)

        # 实际位置
        shoulder_x, shoulder_y = person['shoulder']
        foot_x, foot_y = person['foot']

        # 计算偏移
        offset_x = shoulder_x - target_x
        offset_shoulder_y = shoulder_y - target_shoulder_y
        offset_foot_y = foot_y - target_foot_y

        # 评分
        score_x = max(0, 100 * (1 - abs(offset_x) / (w * 0.15)))
        score_shoulder = max(0, 100 * (1 - abs(offset_shoulder_y) / (h * 0.1)))
        score_foot = max(0, 100 * (1 - abs(offset_foot_y) / (h * 0.1)))
        overall_score = (score_x + score_shoulder + score_foot) / 3

        return {
            'has_person': True,
            'person_center': person['center'],
            'person_shoulder': person['shoulder'],
            'person_foot': person['foot'],
            'person_height': person['height'],
            'target_x': target_x,
            'target_shoulder_y': target_shoulder_y,
            'target_foot_y': target_foot_y,
            'offset_x': offset_x,
            'offset_shoulder_y': offset_shoulder_y,
            'offset_foot_y': offset_foot_y,
            'score': overall_score,
            'is_centered': overall_score >= 80
        }

    def _analyze_sticks(self, sticks, frame):
        """分析荧光棒特征"""
        if not sticks:
            return {
                'has_stick': False,
                'sticks': [],
                'avg_brightness': 0,
                'avg_length': 0,
                'velocity': 0
            }

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        brightnesses = []
        centers = []

        for stick in sticks:
            x1, y1, x2, y2 = stick['bbox']
            roi = gray[y1:y2, x1:x2]
            if roi.size > 0:
                brightnesses.append(np.mean(roi))
            centers.append(stick['center'])

        avg_brightness = np.mean(brightnesses) if brightnesses else 0
        avg_length = np.mean([s['length'] for s in sticks]) if sticks else 0

        # 计算运动速度
        velocity = self._calculate_velocity(centers)

        # 计算整张图的平均亮度（已在外部计算，但这里也保留以备后用）
        avg_frame_brightness = np.mean(gray) if gray.size > 0 else 0

        return {
            'has_stick': True,
            'sticks': sticks,
            'avg_brightness': avg_brightness,
            'avg_length': avg_length,
            'velocity': velocity,
            'avg_frame_brightness': avg_frame_brightness  # 备用
        }

    def _calculate_velocity(self, current_centers):
        """计算荧光棒运动速度"""
        if not self.prev_centers or not current_centers:
            self.prev_centers = current_centers
            return 0

        velocities = []
        for i, curr in enumerate(current_centers):
            if i < len(self.prev_centers):
                prev = self.prev_centers[i]
                dist = np.sqrt((curr[0] - prev[0]) ** 2 + (curr[1] - prev[1]) ** 2)
                velocities.append(dist)

        self.prev_centers = current_centers
        return np.mean(velocities) if velocities else 0

    def _check_need_enhance(self, frame):
        """检查是否需要低光增强"""
        if not ENABLE_LOWLIGHT_ENHANCE:
            return False

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        avg_brightness = np.mean(gray)

        return avg_brightness < LOWLIGHT_THRESHOLD