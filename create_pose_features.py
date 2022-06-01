#create_toy_computational_sequence.py
#this example shows how to create two toy computational sequences and put them together in a dataset

from mmsdk import mmdatasdk
import numpy as np
import os

OPEN_POSE_DIR_NAME = os.path.join("OpenPoseFeatures","TOP10")

def pose_features(compseq, num_f = 5, confidence_threshold = 0.5):
	for vid_key in vid_keys:
		start, end, duration = np.loadtxt(os.path.join(OPEN_POSE_DIR_NAME,"settings_"+vid_key+".csv"), delimiter=",")
		compseq[vid_key]={}
		pose_array = np.loadtxt(os.path.join(OPEN_POSE_DIR_NAME,vid_key,vid_key+"_pose_features.csv"), delimiter=",")
		n_points,_ = pose_array.shape
		n_frames = num_f
		cut_off = n_points-(n_points%n_frames)

		pose_x = pose_array[:cut_off, ::3]
		pose_y = pose_array[:cut_off, 1::3]
		pose_c = pose_array[:cut_off, 2::3]
		pose_x = pose_x.reshape(-1,n_frames, 25)
		pose_y = pose_y.reshape(-1,n_frames, 25)
		pose_c = pose_c.reshape(-1,n_frames, 25)
		n_points = pose_x.shape[0]

		features = np.zeros((n_points,n_frames,2,25))
		features[:,:,0,:] = pose_x
		features[:,:,1,:] = pose_y
		pose_c[pose_c < confidence_threshold] = 0 
		new_pose = features.std(axis=1).mean(axis=1) * pose_c.mean(axis=1)
		compseq[vid_key]["features"] = new_pose
		num_entries = compseq[vid_key]["features"].shape[0]
		#compseq[vid_key]["hand_r"]= np.loadtxt(vid_name+"_hand_r_features.csv", delimiter=",")
		#compseq[vid_key]["hand_l"]= np.loadtxt(vid_name+"_hand_r_features.csv", delimiter=",")
		#let's assume each video is one minute, hence 60 seconds. 
		compseq[vid_key]["intervals"]=np.arange(start=start,stop=end+0.000001,step=duration/((2*num_entries)-1)).reshape([num_entries,2])

def hand_r_features(compseq, num_f=5, confidence_threshold = 0.5):
	for vid_key in vid_keys:
		start, end, duration = np.loadtxt(os.path.join(OPEN_POSE_DIR_NAME,"settings_"+vid_key+".csv"), delimiter=",")
		compseq[vid_key]={}
		hand_r_array = np.loadtxt(os.path.join(OPEN_POSE_DIR_NAME,vid_key,vid_key+"_hand_r_features.csv"), delimiter=",")

		n_points,_ = hand_r_array.shape
		n_frames = num_f
		cut_off = n_points-(n_points%n_frames)

		hand_r_x = hand_r_array[:cut_off, ::3]
		hand_r_y = hand_r_array[:cut_off, 1::3]
		hand_r_c = hand_r_array[:cut_off, 2::3]

		hand_r_x = hand_r_x.reshape(-1,n_frames,21)
		hand_r_y = hand_r_y.reshape(-1,n_frames,21)
		hand_r_c = hand_r_c.reshape(-1,n_frames,21)
		
		n_points = hand_r_c.shape[0]
		features = np.zeros((n_points,n_frames,2,21))
		features[:,:,0,:] = hand_r_x
		features[:,:,1,:] = hand_r_y
		hand_r_c[hand_r_c < confidence_threshold] = 0 
		compseq[vid_key]["features"] = features.std(axis=1).mean(axis=1) * hand_r_c.mean(axis=1)
		num_entries = compseq[vid_key]["features"].shape[0]
		#let's assume each video is one minute, hence 60 seconds. 
		compseq[vid_key]["intervals"]=np.arange(start=start,stop=end+0.000001,step=duration/((2*num_entries)-1)).reshape([num_entries,2])

def hand_l_features(compseq, num_f=5, confidence_threshold=0.5):
	for vid_key in vid_keys:
		start, end, duration = np.loadtxt(os.path.join(OPEN_POSE_DIR_NAME,"settings_"+vid_key+".csv"), delimiter=",")
		compseq[vid_key]={}
		hand_l_array = np.loadtxt(os.path.join(OPEN_POSE_DIR_NAME,vid_key,vid_key+"_hand_l_features.csv"), delimiter=",")

		n_points,_ = hand_l_array.shape
		n_frames = num_f
		cut_off = n_points-(n_points%n_frames)

		hand_l_x = hand_l_array[:cut_off, ::3]
		hand_l_y = hand_l_array[:cut_off, 1::3]
		hand_l_c = hand_l_array[:cut_off, 2::3]

		hand_l_x = hand_l_x.reshape(-1,n_frames,21)
		hand_l_y = hand_l_y.reshape(-1,n_frames,21)
		hand_l_c = hand_l_c.reshape(-1,n_frames,21)
		
		n_points = hand_l_c.shape[0]
		features = np.zeros((n_points,n_frames,2,21))
		features[:,:,0,:] = hand_l_x
		features[:,:,1,:] = hand_l_y
		hand_l_c[hand_l_c < confidence_threshold] = 0 
		compseq[vid_key]["features"] = features.std(axis=1).mean(axis=1) * hand_l_c.mean(axis=1)

		num_entries = compseq[vid_key]["features"].shape[0]
		#let's assume each video is one minute, hence 60 seconds. 
		compseq[vid_key]["intervals"]=np.arange(start=start,stop=end+0.000001,step=duration/((2*n_points)-1)).reshape([n_points,2])


if __name__=="__main__":
	vid_keys= [dir_name for dir_name in os.listdir(OPEN_POSE_DIR_NAME) if ".csv" not in dir_name or dir_name == "nottop10"]
	pos_features_data = {}
	pose_features(pos_features_data)
	hand_r_features_data = {}
	hand_r_features(hand_r_features_data)
	hand_l_features_data = {}
	hand_l_features(hand_l_features_data)
	compseq_1 = mmdatasdk.computational_sequence("pose_features")
	compseq_1.setData(pos_features_data,"pose_features")
	compseq_2 = mmdatasdk.computational_sequence("hand_r_features")
	compseq_2.setData(hand_r_features_data,"hand_r_features")
	compseq_3 = mmdatasdk.computational_sequence("hand_l_features")
	compseq_3.setData(hand_l_features_data,"hand_l_features")
	#NOTE: if you don't want to manually input the metdata, set it by creating a metdata key-value dictionary based on mmsdk/mmdatasdk/configurations/metadataconfigs.py
	compseq_1.deploy("pose_features.csd")
	compseq_2.deploy("hand_r_features.csd")
	compseq_3.deploy("hand_l_features.csd")

	#now creating a toy dataset from the toy compseqs
	mydataset_recipe={"pose_features":"pose_features.csd", "hand_r_features":"hand_r_features.csd", "hand_l_features":"hand_l_features.csd"}
	mydataset = mmdatasdk.mmdataset(mydataset_recipe)
	#let's also see if we can align to compseq_1
	mydataset.align("pose_features")



