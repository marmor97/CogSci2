# conda install -c conda-forge ffmpeg-python
# Tutorial: https://www.thepythoncode.com/article/extract-media-metadata-in-python

from lib2to3.pgen2.token import STRING
from pprint import pprint
import glob
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
from moviepy.editor import *
from datetime import datetime, timedelta


FILTER_VIDEOS_DIR = "D:\\mosei_dataset\\MOSEI\\CMU_MOSEI\\Raw\\Videos\\Full\\Combined\\Filtered"
OUTPUT_DIR = "clips"

def clip_videos(csv_path):
    annotated_file = pd.read_csv(csv_path, delimiter=",")
    annotated_file = annotated_file.drop(annotated_file[annotated_file.Accept != 1].index)
    annotated_file["start"] = annotated_file.TimeSegment.str.replace("[","").str.replace("]","")

    start_times = []
    end_times = []
    duration = []
    print(annotated_file)
    for i, (row_i, row) in enumerate(annotated_file.iterrows()):
        print(i)
        time = annotated_file['start'].iloc[i]
        print(time)
        if type(time) == str:
            time_interval = time.split(",")[0]
            split = time_interval.split(";")
            print(split)
            start = datetime.strptime(split[0],"%M:%S")
            end = datetime.strptime(split[1],"%M:%S")
            start_times.append(start - datetime(1900, 1, 1))
            end_times.append(end - datetime(1900, 1, 1))
            duration.append(end-start)
        else:
            start_times.append(-1)
            end_times.append(-1)
            duration.append(-1)
            continue

    start_times = [-1 if t == -1 else t.total_seconds() for t in start_times]
    end_times = [-1 if t == -1 else t.total_seconds() for t in end_times]
    duration = [-1 if t == -1 else t.total_seconds() for t in duration]

    annotated_file["start"] = start_times
    annotated_file["end"] = end_times
    annotated_file["duration"] = duration


    for row_i, row in annotated_file.iterrows():
        video = row.Filename+".mp4"
        output_path = os.path.join(FILTER_VIDEOS_DIR, OUTPUT_DIR, "clip_"+video)
        settings_path = os.path.join(FILTER_VIDEOS_DIR, OUTPUT_DIR, "settings_"+video.replace(".mp4","")+".csv")
        print(row)
        if row.start == -1:
            clip = VideoFileClip(os.path.join(FILTER_VIDEOS_DIR,video))
            duration = int(round(clip.duration))
            print(duration)
            ffmpeg_extract_subclip(os.path.join(FILTER_VIDEOS_DIR,video), 0, duration, targetname=output_path)
            np.savetxt(settings_path, np.array([0, duration, duration]), delimiter=",")
        else:
            ffmpeg_extract_subclip(os.path.join(FILTER_VIDEOS_DIR,video), row.start, row.end, targetname=output_path)
            np.savetxt(settings_path, np.array([row.start, row.end, row.duration]), delimiter=",")
        print(output_path)
    np.savetxt(os.path.join(FILTER_VIDEOS_DIR, OUTPUT_DIR, "video_clips.csv"),annotated_file.Filename.to_numpy(dtype=str), delimiter="," , fmt='%s')
clip_videos("filter_videos2.csv")