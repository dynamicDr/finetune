import cv2
import os
from pathlib import Path

# 视频文件夹路径
video_dir = Path.home() / "dataset" / "vsi_bench"

# 存储所有视频的帧数
frame_counts = []

# 遍历所有mp4文件
for video_file in video_dir.glob("*.mp4"):
    # 打开视频
    cap = cv2.VideoCapture(str(video_file))

    # 获取帧数
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_counts.append(frame_count)

    print(f"{video_file.name}: {frame_count} 帧")

    # 释放视频
    cap.release()

# 统计数据
if frame_counts:
    print("\n统计数据:")
    print(f"视频总数: {len(frame_counts)}")
    print(f"总帧数: {sum(frame_counts)}")
    print(f"平均帧数: {sum(frame_counts) / len(frame_counts):.2f}")
    print(f"最小帧数: {min(frame_counts)}")
    print(f"最大帧数: {max(frame_counts)}")
else:
    print("没有找到mp4文件")