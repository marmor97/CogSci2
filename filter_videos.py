# conda install -c conda-forge ffmpeg-python
# Tutorial: https://www.thepythoncode.com/article/extract-media-metadata-in-python

from lib2to3.pgen2.token import STRING
import ffmpeg
import sys
from pprint import pprint
import glob
from os import path
import os
import numpy as np
import shutil
from tqdm import tqdm

VIDEO_DIR = "D:\\mosei_dataset\\MOSEI\\CMU_MOSEI\\Raw\\Videos\\Full\\Combined\\"
FILTER_VIDEOS_DIR = "D:\\mosei_dataset\\MOSEI\\CMU_MOSEI\\Raw\\Videos\\Full\\Combined\\Filtered"

POSITIVE_PATH = os.path.join(FILTER_VIDEOS_DIR, "positive")
NEGATIVE_PATH = os.path.join(FILTER_VIDEOS_DIR, "negative")
NEUTRAL_PATH = os.path.join(FILTER_VIDEOS_DIR, "neutral")

def retrieve_video_meta(pattern):
    files_found = []
    total_files = 0
    print("Retrieving Metadata and Filtering Videos...")
    files_found = []
    for filename in glob.iglob(VIDEO_DIR + "*" + pattern, recursive=True):
        total_files += 1
        files_found.append(filename)

    for filename in tqdm(files_found):
        filepath = path.abspath(filename)
        metadata = ffmpeg.probe(filepath)["streams"]
        for dict in metadata:
            if ('duration' in dict.keys()
            and 'coded_height' in dict.keys()
            and 'avg_frame_rate' in dict.keys()
            and 'codec_type' in dict.keys()):
                if dict['codec_type'] == "audio":
                    continue
                duration = float(dict['duration'])
                coded_height = float(dict['coded_height'])
                frames, seconds = dict['avg_frame_rate'].split("/")
                frames, seconds = float(frames), float(seconds)
                frame_rate = frames/seconds
                if np.abs(frame_rate-30) > 0.05:
                    pass
                    #print("Different Framerate!",frame_rate)
                if (duration > 300 
                and duration < 700
                and coded_height >= 720):
                    files_found.append(filename)
                    break

    return files_found, total_files

#filtered_videos, total_videos = retrieve_video_meta("mp4")
#print(f"Found {len(filtered_videos)}/{total_videos} according to the criteria")
pos_videos = ['VDjnKdFxuBE',
 '6-0bcijTR8k',
 'N-NnCI6U52c',
 'HXuRR2eBHHQ',
 'rWjPSlzUc-8',
 'FGnHDWGYsA8',
 '8YsIvz4qnk4',
 'veHYwR7ge6Y',
 'sWsDAG6x9n0',
 'pjWLxORDDP4',
 'mSWwz7dXUP8',
 'lWXjdWvB0VI',
 'aFhT0px8AMw',
 'VbmSzYg_lmg',
 'R-17BoOk4kc',
 'x-zGBp7axao',
 'U8VYG_g6yVE',
 '9K4aWWoXuyU',
 'hE-sA5umuCk',
 'WGfGxoI05ew',
 '3hOlJf_JQDs',
 'tkzdanzsA0A',
 'XVSDfgstFUQ',
 '1LkYxsqRPZM',
 'hfvmQuhqMmY']

neg_videos = ['Zb7bdrjWyEY',
 'XQOUZhWI1B0',
 '262226',
 '113265',
 '24202',
 '273237',
 '92533',
 '275248',
 '243341',
 '79935',
 '28006',
 '245276',
 '56276',
 'zrFZAofNGi4',
 'NVlxq3jxlAk',
 '208299',
 '92496',
 'Es9MkKsMsjU',
 '139032',
 '135623',
 'sfaWfZ2-4c0',
 '265959',
 'A8plGi4rbxM',
 '3ocpWAmSNUE',
 '96350',
 '288714',
 '45186',
 '46497',
 '81668',
 '224263',
 '278474',
 '2W-U94hXuK0',
 '45184',
 '73447',
 '7oFimEZJQ_A',
 '_K0okiLaF9I',
 'o2bNnLOEEC0',
 'pVs1daijYTw',
 '46618',
 '126831',
 '254298',
 'Va54WZgPTdY',
 'jYT3-RQFy1U',
 '130426',
 '63951',
 '243338',
 '154449',
 '92331',
 '266938',
 '107585']

neutral_videos = ['-tANM6ETl_M',
 '9K5mYSaoBL4',
 'liJO1yIFsJs',
 '221153',
 '8NPaDkOiXw4',
 'vrgwwihTxVQ',
 '5xa0Ac2krGs',
 'oNCvb0F-c88',
 'URP2YlSNuZ4',
 'OwYfPi9St0w',
 'kj63uF0RhW8',
 'd3EeIRaMbbk',
 'ymPQOY2O_nw',
 'vJDDEuE-FlY',
 'iXiMifHNKPI']

def copy_videos(video_list, output_dir, filename="filtered_videos"):
    print("Saving Videos to HTML...")
    with open(filename+".html", "w") as f:
        f.write(""" <!DOCTYPE html>
                        <html>
                        <body> """)
        for f_v in tqdm(video_list):
            file_name = f_v.split(os.sep)[-1]
            f_v =  os.path.join(output_dir, f_v+".mp4") 
            string = f"""<a href="file:///{f_v}">{file_name}</a><br>"""
            f.write(string + "\n")
        f.write("""</body> 
                    </html>""")

    print("Copying files...")
    for video in tqdm(video_list):
        path_video = os.path.join(VIDEO_DIR,video+".mp4")
        dst = path.join(output_dir, path_video.split(os.sep)[-1])
        shutil.copyfile(path_video, dst)
    np.savetxt(filename+".csv", np.array([f_v.split(os.sep)[-1] for f_v in video_list], dtype=str), delimiter=",", fmt="%s")

copy_videos(pos_videos, POSITIVE_PATH, "positive")
copy_videos(neg_videos, NEGATIVE_PATH, "negative")
copy_videos(neutral_videos, NEUTRAL_PATH, "neutral")
