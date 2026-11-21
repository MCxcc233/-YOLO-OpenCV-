# type: ignore
"""
人体站位引导系统
目标：肩膀在画面1/3处，脚在底部1/5处，人居中
"""

import cv2
import numpy as np

FONT = cv2.FONT_HERSHEY_SIMPLEX

class HumanCenteringGuide:
    def __init__(self):
        self.shoulder_y_ratio = 1/3      # 肩膀在1/3高度
        self.foot_y_ratio = 4/5           # 脚在4/5高度（离底部1/5）
        self.center_x_ratio = 1/2         # 水平居中
        self.tolerance = 0.1               # 容忍范围

    def analyze_position(self, person_features, frame_shape):
        """
        分析人体站位
        person_features: analyzer返回的人体特征
        """
        if not person_features.get('has_person', False):
            return {
                'has_person': False,
                'instructions': ['NO PERSON DETECTED'],
                'score': 0,
                'is_centered': False
            }

        h, w = frame_shape[:2]

        # 目标位置
        target_x = person_features.get('target_x', w // 2)
        target_shoulder_y = person_features.get('target_shoulder_y', int(h * self.shoulder_y_ratio))
        target_foot_y = person_features.get('target_foot_y', int(h * self.foot_y_ratio))

        # 实际位置
        shoulder = person_features.get('person_shoulder')
        foot = person_features.get('person_foot')

        if shoulder is None or foot is None:
            return {
                'has_person': True,
                'instructions': ['DETECTING...'],
                'score': 50,
                'is_centered': False
            }

        shoulder_x, shoulder_y = shoulder
        foot_x, foot_y = foot

        # 计算偏移
        offset_x = shoulder_x - target_x
        offset_shoulder_y = shoulder_y - target_shoulder_y
        offset_foot_y = foot_y - target_foot_y

        # 生成移动指令（纯英文，不加符号）
        instructions = []

        # 水平方向
        if abs(offset_x) > w * self.tolerance:
            if offset_x > 0:
                instructions.append("MOVE LEFT")
            else:
                instructions.append("MOVE RIGHT")

        # 肩膀垂直方向
        if abs(offset_shoulder_y) > h * self.tolerance * 0.5:
            if offset_shoulder_y > 0:
                instructions.append("SHOULDER UP")
            else:
                instructions.append("SHOULDER DOWN")

        # 脚垂直方向
        if abs(offset_foot_y) > h * self.tolerance * 0.5:
            if offset_foot_y > 0:
                instructions.append("FEET UP")
            else:
                instructions.append("FEET DOWN")

        if not instructions:
            instructions.append("PERFECT")

        # 评分
        score_x = max(0, 100 * (1 - abs(offset_x) / (w * 0.15)))
        score_shoulder = max(0, 100 * (1 - abs(offset_shoulder_y) / (h * 0.1)))
        score_foot = max(0, 100 * (1 - abs(offset_foot_y) / (h * 0.1)))
        overall_score = (score_x + score_shoulder + score_foot) / 3

        return {
            'has_person': True,
            'shoulder': (shoulder_x, shoulder_y),
            'foot': (foot_x, foot_y),
            'target_shoulder': (target_x, target_shoulder_y),
            'target_foot': (target_x, target_foot_y),
            'instructions': instructions,
            'score': overall_score,
            'is_centered': overall_score >= 80
        }


def draw_human_guide(frame, result):
    """绘制人体站位引导"""
    if not result.get('has_person', False):
        text = "NO PERSON DETECTED"
        text_size = cv2.getTextSize(text, FONT, 0.8, 2)[0]
        text_x = (frame.shape[1] - text_size[0]) // 2
        text_y = (frame.shape[0] - text_size[1]) // 2
        cv2.putText(frame, text, (text_x, text_y), FONT, 0.8, (0, 0, 255), 2)
        return

    h, w = frame.shape[:2]

    # 绘制目标位置（绿色）
    tx, ty_s = result.get('target_shoulder', (w//2, int(h*1/3)))
    tx, ty_f = result.get('target_foot', (w//2, int(h*4/5)))

    # 目标肩膀线
    cv2.line(frame, (tx-30, ty_s), (tx+30, ty_s), (0, 255, 0), 2)
    cv2.putText(frame, "SHOULDER", (tx+40, ty_s-5), FONT, 0.4, (0, 255, 0), 1)

    # 目标脚线
    cv2.line(frame, (tx-30, ty_f), (tx+30, ty_f), (0, 255, 0), 2)
    cv2.putText(frame, "FEET", (tx+40, ty_f-5), FONT, 0.4, (0, 255, 0), 1)

    # 垂直中线
    cv2.line(frame, (tx, 0), (tx, h), (0, 255, 0), 1)

    # 绘制实际位置（红色）
    shoulder = result.get('shoulder')
    foot = result.get('foot')

    if shoulder is not None:
        sx, sy = shoulder
        cv2.circle(frame, (sx, sy), 8, (0, 0, 255), -1)

    if foot is not None:
        fx, fy = foot
        cv2.circle(frame, (fx, fy), 8, (0, 0, 255), -1)

    if shoulder is not None and foot is not None:
        sx, sy = shoulder
        fx, fy = foot
        cv2.line(frame, (sx, sy), (fx, fy), (255, 0, 0), 2)

    # 移动方向箭头
    if not result.get('is_centered', False) and shoulder is not None and foot is not None:
        sx, sy = shoulder
        fx, fy = foot
        cv2.arrowedLine(frame, (sx, sy), (tx, ty_s), (255, 255, 0), 2)
        cv2.arrowedLine(frame, (fx, fy), (tx, ty_f), (255, 255, 0), 2)

    # 分数进度条放在左下角
    bar_x, bar_y = 10, h - 40
    bar_w, bar_h = 100, 8

    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (200, 200, 200), -1)
    fill_w = int(bar_w * result.get('score', 0) / 100)
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), (0, 200, 0), -1)

    cv2.putText(frame, f"SCORE: {result.get('score', 0):.0f}%", (bar_x + bar_w + 10, bar_y + 8),
                FONT, 0.4, (0, 0, 0), 1)

    # 指令放在左下角，分数下面
    y_offset = h - 70
    instructions = result.get('instructions', [])
    for instruction in instructions[:3]:  # 最多显示3条
        cv2.putText(frame, instruction, (10, y_offset), FONT, 0.8, (0, 0, 255), 3)
        y_offset -= 20