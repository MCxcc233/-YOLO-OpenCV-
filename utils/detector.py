# type: ignore
"""
YOLO检测模块 - 双模型：人体 + 荧光棒
"""

from ultralytics import YOLO
import cv2
import numpy as np
from config import CONFIDENCE_THRESHOLD


class StickDetector:
    def __init__(self, stick_model_path, person_model_path='models/yolov8n.pt'):
        """
        初始化两个模型
        stick_model_path: 你的荧光棒模型路径
        person_model_path: 官方人体模型路径
        """
        print(f"加载荧光棒模型: {stick_model_path}")
        self.stick_model = YOLO(stick_model_path)

        print(f"加载人体模型: {person_model_path}")
        self.person_model = YOLO(person_model_path)

        # 从配置文件读取阈值
        self.conf_threshold = CONFIDENCE_THRESHOLD
        print(f"检测置信度阈值: {self.conf_threshold}")

        # COCO数据集的类别（人体是0）
        self.COCO_CLASSES = {
            0: 'person',
            1: 'bicycle',
            2: 'car',
            3: 'motorcycle',
            4: 'airplane',
            5: 'bus',
            6: 'train',
            7: 'truck',
            8: 'boat',
            9: 'traffic light',
            10: 'fire hydrant',
            11: 'stop sign',
            12: 'parking meter',
            13: 'bench',
            14: 'bird',
            15: 'cat',
            16: 'dog',
            17: 'horse',
            18: 'sheep',
            19: 'cow',
            20: 'elephant',
            21: 'bear',
            22: 'zebra',
            23: 'giraffe',
            24: 'backpack',
            25: 'umbrella',
            26: 'handbag',
            27: 'tie',
            28: 'suitcase',
            29: 'frisbee',
            30: 'skis',
            31: 'snowboard',
            32: 'sports ball',
            33: 'kite',
            34: 'baseball bat',
            35: 'baseball glove',
            36: 'skateboard',
            37: 'surfboard',
            38: 'tennis racket',
            39: 'bottle',
            40: 'wine glass',
            41: 'cup',
            42: 'fork',
            43: 'knife',
            44: 'spoon',
            45: 'bowl',
            46: 'banana',
            47: 'apple',
            48: 'sandwich',
            49: 'orange',
            50: 'broccoli',
            51: 'carrot',
            52: 'hot dog',
            53: 'pizza',
            54: 'donut',
            55: 'cake',
            56: 'chair',
            57: 'couch',
            58: 'potted plant',
            59: 'bed',
            60: 'dining table',
            61: 'toilet',
            62: 'tv',
            63: 'laptop',
            64: 'mouse',
            65: 'remote',
            66: 'keyboard',
            67: 'cell phone',
            68: 'microwave',
            69: 'oven',
            70: 'toaster',
            71: 'sink',
            72: 'refrigerator',
            73: 'book',
            74: 'clock',
            75: 'vase',
            76: 'scissors',
            77: 'teddy bear',
            78: 'hair drier',
            79: 'toothbrush'
        }

    def detect_all(self, frame):
        """
        同时检测人体和荧光棒
        返回：人体框列表、荧光棒框列表
        """
        # 人体检测（用官方模型）
        person_results = self.person_model(frame, conf=self.conf_threshold)[0]
        persons = self._extract_person_boxes(person_results)

        # 荧光棒检测（用你的模型）
        stick_results = self.stick_model(frame, conf=self.conf_threshold)[0]
        sticks = self._extract_stick_boxes(stick_results)

        return persons, sticks

    def _extract_person_boxes(self, results):
        """从检测结果中提取人体框（类别ID=0）"""
        persons = []
        if results.boxes is not None:
            for i, box in enumerate(results.boxes):
                class_id = int(box.cls[0])
                if class_id == 0:  # person
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0])

                    # 计算中心点
                    center_x = (x1 + x2) // 2
                    center_y = (y1 + y2) // 2

                    # 估算肩膀位置（人体上半部）
                    shoulder_y = y1 + int((y2 - y1) * 0.2)  # 头顶往下20%

                    # 脚的位置
                    foot_y = y2

                    persons.append({
                        'bbox': (x1, y1, x2, y2),
                        'center': (center_x, center_y),
                        'shoulder': (center_x, shoulder_y),
                        'foot': (center_x, foot_y),
                        'height': y2 - y1,
                        'width': x2 - x1,
                        'confidence': conf,
                        'class_id': class_id,
                        'class_name': 'person'
                    })
        return persons

    def _extract_stick_boxes(self, results):
        """从检测结果中提取荧光棒框（你的模型只有一类）"""
        sticks = []
        if results.boxes is not None:
            for box in results.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])

                # 中心点
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2

                # 长度（用于运动速度计算）
                length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

                sticks.append({
                    'bbox': (x1, y1, x2, y2),
                    'center': (center_x, center_y),
                    'length': length,
                    'confidence': conf
                })
        return sticks

    def detect(self, frame):
        """兼容旧代码，返回人体检测结果"""
        persons, sticks = self.detect_all(frame)
        return persons  # 返回人体列表（兼容旧的接口）

    def get_stick_boxes(self, results):
        """兼容旧代码，返回荧光棒框"""
        # 这个函数在旧代码中调用，但现在我们直接返回sticks
        return []