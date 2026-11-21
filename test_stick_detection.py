# type: ignore
"""
荧光棒模型单独测试脚本
用于检测YOLO模型是否能正确识别荧光棒
"""

import cv2
import os
import glob
from ultralytics import YOLO
import numpy as np


def test_model_on_image(model_path, image_path, output_dir='test_output'):
    """
    测试模型在单张图片上的表现
    """
    print(f"\n{'=' * 60}")
    print(f"测试图片: {image_path}")
    print(f"{'=' * 60}")

    # 加载模型
    try:
        model = YOLO(model_path)
        print(f"✓ 模型加载成功: {model_path}")
    except Exception as e:
        print(f"✗ 模型加载失败: {e}")
        return False

    # 读取图片
    img = cv2.imread(image_path)
    if img is None:
        print(f"✗ 无法读取图片: {image_path}")
        return False

    print(f"✓ 图片尺寸: {img.shape}")

    # 尝试不同置信度阈值
    conf_thresholds = [0.1, 0.2, 0.3, 0.4, 0.5]

    for conf in conf_thresholds:
        print(f"\n--- 置信度阈值: {conf} ---")

        # 推理
        results = model(img, conf=conf)[0]

        # 检查检测结果
        if results.boxes is not None and len(results.boxes) > 0:
            print(f"✓ 检测到 {len(results.boxes)} 个物体")

            # 显示每个检测到的物体
            for i, box in enumerate(results.boxes):
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf_score = float(box.conf[0])
                class_id = int(box.cls[0])

                print(f"  物体 {i + 1}: 位置=({x1},{y1},{x2},{y2}), 置信度={conf_score:.3f}, 类别ID={class_id}")

                # 在图片上画框
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(img, f"{conf_score:.2f}", (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        else:
            print(f"✗ 未检测到任何物体")

    # 保存结果
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, os.path.basename(image_path))
    cv2.imwrite(output_path, img)
    print(f"\n✓ 结果已保存到: {output_path}")

    return True


def test_model_on_video(model_path, video_path, output_dir='test_output', frame_skip=30):
    """
    测试模型在视频上的表现（每隔30帧检测一次）
    """
    print(f"\n{'=' * 60}")
    print(f"测试视频: {video_path}")
    print(f"{'=' * 60}")

    # 加载模型
    try:
        model = YOLO(model_path)
        print(f"✓ 模型加载成功: {model_path}")
    except Exception as e:
        print(f"✗ 模型加载失败: {e}")
        return False

    # 打开视频
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"✗ 无法打开视频: {video_path}")
        return False

    # 获取视频信息
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"✓ 视频信息: {width}x{height}, {fps}fps, 总帧数={total_frames}")

    frame_count = 0
    detection_count = 0
    frames_with_detection = 0

    os.makedirs(output_dir, exist_ok=True)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 每隔 frame_skip 帧检测一次
        if frame_count % frame_skip == 0:
            # 推理
            results = model(frame, conf=0.3)[0]

            if results.boxes is not None and len(results.boxes) > 0:
                frames_with_detection += 1
                detection_count += len(results.boxes)

                print(f"帧 {frame_count}: 检测到 {len(results.boxes)} 个物体")

                # 保存这一帧
                output_path = os.path.join(output_dir, f"frame_{frame_count}.jpg")
                cv2.imwrite(output_path, frame)

        frame_count += 1

        # 进度提示
        if frame_count % 300 == 0:
            print(f"已处理 {frame_count}/{total_frames} 帧...")

    cap.release()

    total_checked = total_frames // frame_skip
    print(f"\n{'=' * 60}")
    print(f"测试结果:")
    print(f"  检查了 {total_checked} 帧")
    print(f"  其中有检测结果的帧数: {frames_with_detection}")
    print(f"  总共检测到 {detection_count} 个物体")

    if frames_with_detection > 0:
        print(f"✓ 模型能识别荧光棒！")
    else:
        print(f"✗ 模型未能识别任何荧光棒")

    return frames_with_detection > 0


def main():
    """主测试函数"""
    print("=" * 60)
    print("荧光棒模型测试工具")
    print("=" * 60)

    # 模型路径
    model_path = 'models/best.pt'

    # 检查模型文件
    if not os.path.exists(model_path):
        print(f"✗ 模型文件不存在: {model_path}")
        print("请确认 models/best.pt 文件存在")
        return

    # === 图片和视频路径 ===
    photo_dir = r'D:\StickHelper\outputs\photo4test'
    video_dir = r'D:\StickHelper\outputs\video4test'

    # 测试图片
    if os.path.exists(photo_dir):
        # 测试指定的4张图片
        test_images = ['1.jpg', '2.jpg', '3.jpg', '4.jpg']
        print(f"\n{'=' * 60}")
        print(f"测试图片文件夹: {photo_dir}")
        print(f"{'=' * 60}")

        for img_name in test_images:
            img_path = os.path.join(photo_dir, img_name)
            if os.path.exists(img_path):
                test_model_on_image(model_path, img_path, output_dir='outputs/test_output/photos')
            else:
                print(f"✗ 图片不存在: {img_path}")
    else:
        print(f"✗ 图片文件夹不存在: {photo_dir}")

    # 测试视频
    if os.path.exists(video_dir):
        video_path = os.path.join(video_dir, '1.mp4')
        if os.path.exists(video_path):
            test_model_on_video(model_path, video_path, output_dir='outputs/test_output/video', frame_skip=30)
        else:
            print(f"✗ 视频不存在: {video_path}")
    else:
        print(f"✗ 视频文件夹不存在: {video_dir}")

    print("\n" + "=" * 60)
    print("测试完成！")
    print("请查看 test_output 文件夹中的结果图片")
    print("=" * 60)


if __name__ == "__main__":
    main()