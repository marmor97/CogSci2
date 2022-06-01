import ffmpeg
import sys
from pprint import pprint
import glob
import os
import numpy as np
import shutil
from tqdm import tqdm

VIDEO_DIR = "D:\\mosei_dataset\\MOSEI\\CMU_MOSEI\\Raw\\Videos\\Segmented\\Combined\\"
# FILTER_VIDEOS_DIR = "D:\\mosei_dataset\\MOSEI\\CMU_MOSEI\\Raw\\Videos\\Full\\Combined\\Filtered"

def retrieve_video_meta(pattern):
    files_found = []
    total_files = 0
    print("Getting all segmented videos names...")
    unique_found = set()
    for filename in glob.iglob(VIDEO_DIR + "*" + pattern, recursive=True):
        video_unique = filename.split(os.sep)[-1].split("_")[-2]
        unique_found.add(video_unique)
        total_files += 1
        files_found.append(filename)
    return list(unique_found)

unique_videos = retrieve_video_meta(".mp4")
np.savetxt("segmented_videos.csv", np.array([f_v.split(os.sep)[-1] for f_v in unique_videos], dtype=str), delimiter=",", fmt="%s")