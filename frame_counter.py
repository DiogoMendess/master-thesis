import cv2
import os
import pandas as pd

CAMERAS = ["1.03", "1.04", "1.05", "2.05"]
SAVE_PATH = "results/frame_counter"


def get_video_numbers_sorted(path):
    videos = os.listdir(path)
    video_numbers = []
    for video_name in videos:
        result = video_name[14:].split("_")
        video_numbers.append((int(result[0]), int(result[1])))
    return sorted(video_numbers)


def count_frames_tocsv(camera):
    videos_dir_path = f"C:/Users/diogo/Desktop/Tese/Dados/Videos/14.03.2022/20220314_{camera}_blurred"
    video_numbers = get_video_numbers_sorted(videos_dir_path)
    name_and_length = []
    for n1, n2 in video_numbers:
        video_name = f"20220314_{camera}_{n1}_{n2}_blurred"
        video = cv2.VideoCapture(f"{videos_dir_path}/{video_name}.mp4")
        length = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        name_and_length.append([video_name, length])
    pd.DataFrame(name_and_length).to_csv(f"{SAVE_PATH}/frame_count_{camera}.csv", header=["video_name", "length"], index=False)


for camera in CAMERAS:
    count_frames_tocsv(camera)