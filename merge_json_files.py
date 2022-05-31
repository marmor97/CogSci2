import json
import os
import numpy as np
from pprint import pprint

def directory_merge_json_files(path):
    directories_to_merge = [dir_name for dir_name in os.listdir(path) if ".csv" not in dir_name]
    print(directories_to_merge)
    for video_names in directories_to_merge:
        merge_json_files(os.path.join(path,video_names))

def merge_json_files(path):
    files_to_merge = os.listdir(path)
    filename = path.split(os.sep)[-1]
    pose_keypoints = []
    hand_right_keypoints = []
    hand_left_keypoints = []
    for f in files_to_merge:
        if ".json" in f:
            print(f)
            file = open(os.path.join(path, f))
            data = json.load(file)
            if len(data['people']):
                first_person = data['people'][0]
                pose_keypoints.append(np.array(first_person['pose_keypoints_2d']))
                hand_right_keypoints.append(np.array(first_person['hand_right_keypoints_2d']))
                hand_left_keypoints.append(np.array(first_person['hand_left_keypoints_2d']))
            else:
                # Person wasn't tracked = Fill with 0s
                pose_keypoints.append(np.zeros(75))
                hand_right_keypoints.append(np.zeros(63))
                hand_left_keypoints.append(np.zeros(63)) 
    
    pose_keypoints = np.array(pose_keypoints)
    hand_left_keypoints = np.array(hand_left_keypoints)
    hand_right_keypoints = np.array(hand_right_keypoints)

    np.savetxt(os.path.join(path, filename+"_pose_features.csv"), pose_keypoints, delimiter=",")
    np.savetxt(os.path.join(path, filename+"_hand_l_features.csv"), hand_left_keypoints, delimiter=",")
    np.savetxt(os.path.join(path, filename+"_hand_r_features.csv"), hand_right_keypoints, delimiter=",")
    return pose_keypoints, hand_left_keypoints, hand_right_keypoints
 
#pose, hand_l, hand_r = merge_json_files("iRBcNs9oI8")
directory_merge_json_files("OpenPoseFeatures")