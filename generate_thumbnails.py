from moviepy.editor import *
import os

VIDEO_DIR = "clips"
THUMBNAIL_DIR = os.path.join(VIDEO_DIR,"thumbnails")

for filename in os.listdir(VIDEO_DIR):
    if "mp4" in filename or "avi" in filename:
        name = filename.split(".")[0].replace("clip_","") + "_thumb.jpg"
        clip = VideoFileClip(os.path.join(VIDEO_DIR,filename))
        clip.save_frame(os.path.join(THUMBNAIL_DIR,name), t=5.00)