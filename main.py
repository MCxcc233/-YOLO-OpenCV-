# type: ignore
# spell-checker: disable

import cv2
import os
import time
import threading
import queue
import numpy as np
from datetime import datetime

from config import *
from utils.detector import StickDetector
from utils.analyzer import StickAnalyzer
from utils.camera_params import CameraParamRecommender
from utils.centering_guide import HumanCenteringGuide, draw_human_guide
from utils.lowlight_enhance import enhance_lowlight_image


class VideoRecorder:
    """视频录制器 - 固定帧率10fps"""

    def __init__(self, filename, frame_size):
        self.filename = filename
        self.target_fps = 10  # 固定10fps，确保播放速度正常
        self.frame_size = frame_size
        self.frame_queue = queue.Queue(maxsize=60)
        self.is_recording = False
        self.thread = None
        self.video_writer = None
        self.frames_written = 0
        self.start_time = None

    def start(self):
        """开始录制"""
        self.is_recording = True
        self.start_time = time.time()
        self.thread = threading.Thread(target=self._recording_thread)
        self.thread.daemon = True
        self.thread.start()
        print(f"[Record] Start: {self.filename} (10fps)")

    def stop(self):
        """停止录制"""
        self.is_recording = False
        if self.thread is not None:
            self.thread.join(timeout=2)
        if self.video_writer is not None:
            self.video_writer.release()
        duration = 0
        if self.start_time is not None:
            duration = time.time() - self.start_time
        actual_fps = 0
        if duration > 0:
            actual_fps = self.frames_written / duration
        print(f"[Record] Stop: {self.frames_written} frames, {duration:.1f}s, {actual_fps:.1f}fps")

    def add_frame(self, frame):
        """添加一帧到队列"""
        if self.is_recording:
            try:
                if self.frame_queue.qsize() < self.frame_queue.maxsize:
                    self.frame_queue.put(frame.copy(), block=False)
            except Exception:
                pass

    def _recording_thread(self):
        """录制线程 - 严格按10fps写入"""
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.video_writer = cv2.VideoWriter(
            self.filename, fourcc, self.target_fps, self.frame_size
        )

        if self.video_writer is None or not self.video_writer.isOpened():
            print(f"[Record] Error: Cannot create video file")
            return

        frame_interval = 1.0 / self.target_fps  # 1/10 = 0.1秒
        next_frame_time = time.time()

        while self.is_recording:
            try:
                frame = self.frame_queue.get(timeout=0.1)
                current_time = time.time()
                if current_time >= next_frame_time:
                    self.video_writer.write(frame)
                    self.frames_written += 1
                    next_frame_time += frame_interval
            except queue.Empty:
                continue
            except Exception:
                continue

        if self.video_writer is not None:
            self.video_writer.release()


def main():
    """主函数"""
    print("=" * 60)
    print("FLUORESCENT STICK DANCE ASSISTANT")
    print("=" * 60)

    # 创建输出文件夹
    os.makedirs('outputs/videos', exist_ok=True)
    os.makedirs('outputs/debug', exist_ok=True)

    # 初始化检测器
    print("[1/4] Loading YOLO models...")
    try:
        detector = StickDetector(
            stick_model_path='models/best.pt',
            person_model_path='models/yolov8n.pt'
        )
        print("    ✓ Models loaded successfully")
    except Exception as model_error:
        print(f"    ✗ Error loading models: {model_error}")
        return

    print("[2/4] Initializing analyzer...")
    analyzer = StickAnalyzer()

    print("[3/4] Initializing human centering guide...")
    centering_guide = HumanCenteringGuide()

    print("[4/4] Initializing parameter recommender...")
    recommender = CameraParamRecommender()

    # 打开摄像头
    print("\nOpening camera...")
    cap = cv2.VideoCapture(CAMERA_ID)
    if not cap.isOpened():
        print("Error: Cannot open camera!")
        return

    # 设置摄像头分辨率
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, TARGET_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, TARGET_HEIGHT)

    # 获取实际分辨率
    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    if actual_fps <= 0:
        actual_fps = 30

    print(f"Camera: {actual_width}x{actual_height} @ {actual_fps}fps")
    print(f"Recording will be saved at 10fps (fixed)")

    # 创建窗口
    window_name = 'Fluorescent Stick Dance Assistant'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    # 设置窗口大小为屏幕的一半
    try:
        import tkinter as tk
        root = tk.Tk()
        screen_width = root.winfo_screenwidth()
        root.destroy()
        display_width = screen_width // 2
        display_height = int(display_width * actual_height / actual_width)
        cv2.resizeWindow(window_name, display_width, display_height)
    except Exception:
        # 如果获取屏幕尺寸失败，使用默认大小
        cv2.resizeWindow(window_name, 640, 360)

    # 录制相关
    recorder = None
    is_recording = False

    # UI控制
    show_ui = True
    mirror_mode = True

    # 跳帧处理（提高性能）
    frame_skip = 2
    frame_count = 0
    last_result = None

    # FPS计算
    fps_display = 0
    fps_counter = 0
    fps_timer = time.time()

    print("\n" + "=" * 60)
    print("CONTROLS:")
    print("  SPACE - Start/Stop recording")
    print("  S     - Screenshot")
    print("  R     - Reset tracking")
    print("  H     - Hide/Show UI")
    print("  M     - Mirror mode")
    print("  Q     - Quit")
    print("=" * 60 + "\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Cannot read frame")
            break

        # 镜像翻转（自拍模式）
        if mirror_mode:
            frame = cv2.flip(frame, 1)

        # 保存原始帧用于录制
        original_frame = frame.copy()

        frame_count += 1
        current_time = time.time()

        # 计算显示FPS
        fps_counter += 1
        if current_time - fps_timer >= 1.0:
            fps_display = fps_counter
            fps_counter = 0
            fps_timer = current_time

        # 双模型检测（跳帧处理）
        if frame_count % frame_skip == 0 or last_result is None:
            persons, sticks = detector.detect_all(frame)
            features = analyzer.analyze_frame(frame, persons, sticks)
            params = recommender.recommend(features)

            last_result = {
                'features': features,
                'params': params,
                'persons': persons,
                'sticks': sticks
            }

        features = last_result['features']
        params = last_result['params']

        # 创建显示用画面
        display_frame = original_frame.copy()

        # 低光增强
        if features.get('need_enhance', False) and ENABLE_LOWLIGHT_ENHANCE:
            display_frame = enhance_lowlight_image(display_frame)

        # 绘制UI
        if show_ui:
            # 人体站位引导
            if features.get('has_person', False):
                guide_result = centering_guide.analyze_position(features, display_frame.shape)
                draw_human_guide(display_frame, guide_result)

            # 荧光棒信息
            stick_count = features.get('stick_count', 0)
            if stick_count > 0:
                cv2.putText(display_frame, f"STICKS: {stick_count}",
                            (10, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

            # 显示推荐参数
            show_camera_params(display_frame, params, features)

            # FPS显示
            cv2.putText(display_frame, f"FPS: {fps_display}", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

            # 操作提示
            controls = [
                "SPACE: Record",
                "S: Screenshot",
                "R: Reset",
                "H: Hide UI",
                "M: Mirror",
                "Q: Quit"
            ]
            for i, text in enumerate(controls):
                cv2.putText(display_frame, text, (10, 50 + i * 18),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)

            # 镜像模式提示（右上角，黑色字体）
            if mirror_mode:
                cv2.putText(display_frame, "MIRROR",
                            (display_frame.shape[1] - 80, 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)  # 改为黑色

        # 录制状态显示（红点在时间左边）
        if is_recording and recorder is not None:
            if recorder.start_time is not None:
                elapsed = int(current_time - recorder.start_time)
                time_text = f"{elapsed // 60:02d}:{elapsed % 60:02d}"

                # 计算文字大小和位置
                text_size = cv2.getTextSize(time_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                text_x = display_frame.shape[1] - text_size[0] - 45  # 留出红点位置
                text_y = 65  # 放在MIRROR下面，避免重叠

                # 先画时间（红色）
                cv2.putText(display_frame, time_text, (text_x, text_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

                # 红点在时间左边（闪烁）
                if int(current_time * 2) % 2 == 0:
                    cv2.circle(display_frame, (text_x - 15, text_y - 8), 8, (0, 0, 255), -1)

        # 显示画面
        cv2.imshow(window_name, display_frame)

        # 录制 - 添加原始帧（不加任何UI）
        if is_recording and recorder is not None:
            recorder.add_frame(original_frame)

        # 按键处理
        key = cv2.waitKey(1) & 0xFF

        # 空格键：开始/停止录制
        if key == 32:
            if not is_recording:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f'outputs/videos/recording_{timestamp}_10fps.mp4'  # 文件名标注fps
                recorder = VideoRecorder(
                    filename,
                    (original_frame.shape[1], original_frame.shape[0])
                )
                recorder.start()
                is_recording = True
            else:
                if recorder is not None:
                    recorder.stop()
                is_recording = False

        # S键：截图
        elif key == ord('s') or key == ord('S'):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f'outputs/debug/screenshot_{timestamp}.jpg'
            cv2.imwrite(filename, display_frame)
            print(f"Screenshot saved: {filename}")

        # R键：重置追踪
        elif key == ord('r') or key == ord('R'):
            analyzer.prev_centers = []
            print("Tracking reset")

        # H键：隐藏/显示UI
        elif key == ord('h') or key == ord('H'):
            show_ui = not show_ui

        # M键：镜像模式切换
        elif key == ord('m') or key == ord('M'):
            mirror_mode = not mirror_mode

        # Q键：退出
        elif key == ord('q') or key == ord('Q'):
            print("Quitting...")
            break

    # 清理资源
    if is_recording and recorder is not None:
        recorder.stop()
    cap.release()
    cv2.destroyAllWindows()
    print("Program closed")


def show_camera_params(frame, params, features):
    """在画面上显示推荐的拍摄参数"""
    h, w = frame.shape[:2]

    # 面板位置（右下角）
    x = w - 200
    y = h - 170
    w_panel = 190
    h_panel = 140

    # 半透明背景
    overlay = frame.copy()
    cv2.rectangle(overlay, (x, y), (x + w_panel, y + h_panel), (255, 255, 255), -1)
    cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)

    # 边框
    cv2.rectangle(frame, (x, y), (x + w_panel, y + h_panel), (100, 100, 100), 1)

    # 标题
    cv2.putText(frame, "CAMERA SETTINGS", (x + 10, y + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)

    # 参数列表
    y_text = y + 45
    texts = [
        f"ISO: {params.get('iso', 'AUTO')}",
        f"Aperture: {params.get('aperture', 'AUTO')}",
        f"EV: {params.get('ev', 0):+1.1f}",
        f"Shutter: {params.get('shutter', 'AUTO')}"
    ]

    for i, text in enumerate(texts):
        cv2.putText(frame, text, (x + 10, y_text + i * 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)

    # 当前亮度（改为黑色）
    brightness = features.get('avg_brightness', 0)
    cv2.putText(frame, f"Brightness: {brightness:.0f}",
                (x + 10, y_text + 90), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nUser interrupted")
    except Exception as main_error:
        print(f"\nError: {main_error}")
        import traceback

        traceback.print_exc()
    finally:
        print("Program exited")