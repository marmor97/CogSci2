import os
import numpy as np

CLIP_DIR = "clips"
CLIP_NAMES = np.loadtxt(os.path.join(CLIP_DIR,"video_clips.csv"), dtype=str, delimiter=",")
for clip_name in CLIP_NAMES:
    clip_name_no_ext = clip_name.split(".")[0]
    os.system(f"""OpenPoseDemo.exe --video D:\mosei_dataset\openpose-1.7.0-binaries-win64-gpu-python3.7-flir-3d_recommended\openpose\clips\clip_{clip_name_no_ext}.mp4 --write_json OpenPoseFeatures\{clip_name_no_ext} --hand -number_people_max 1 --net_resolution \"-1x256\" --display 0 --render_pose 1 -write_video OpenPoseFeatures\pose_{clip_name_no_ext}.avi """)