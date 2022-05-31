import numpy as np
import matplotlib.pyplot as plt

pose = np.loadtxt("tkzdanzsA0A_pose_features.csv", delimiter=",")
hand_r_array = np.loadtxt("tkzdanzsA0A_hand_r_features.csv", delimiter=",")
n_points,_ = pose.shape
n_frames = 5
cut_off = n_points-(n_points%n_frames)

pose_x = pose[:cut_off, ::3]
pose_y = pose[:cut_off, 1::3]
pose_c = pose[:cut_off, 2::3]
pose_x = pose_x.reshape(-1,n_frames, 25)
pose_y = pose_y.reshape(-1,n_frames, 25)
pose_c = pose_c.reshape(-1,n_frames, 25)
n_points = pose_x.shape[0]

features = np.zeros((n_points,n_frames,2,25))
features[:,:,0,:] = pose_x
features[:,:,1,:] = pose_y
pose_c[pose_c < 0.5] = 0 
new_pose = features.std(axis=1).mean(axis=1) * pose_c.mean(axis=1)

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

features_h_r = features.std(axis=1).mean(axis=1) * hand_r_c.mean(axis=1)

POSE_BODY_25_BODY_PARTS = {
     0:  "Nose",
     1:  "Neck",
     2:  "RShoulder",
     3:  "RElbow",
     4:  "RWrist",
     5:  "LShoulder",
     6:  "LElbow",
     7:  "LWrist",
     8:  "MidHip",
     9:  "RHip",
     10: "RKnee",
     11: "RAnkle",
     12: "LHip",
     13: "LKnee",
     14: "LAnkle",
     15: "REye",
     16: "LEye",
     17: "REar",
     18: "LEar",
     19: "LBigToe",
     20: "LSmallToe",
     21: "LHeel",
     22: "RBigToe",
     23: "RSmallToe",
     24: "RHeel",
     25: "Background"}

hand_r_c[hand_r_c < 0.4] = 0
features = new_pose

plt_1 = plt.figure(figsize=(6,4))
ax = plt.subplot(111)
points_to_plot = 30
for point in range(25):
    ax.plot(np.arange(points_to_plot), features[:points_to_plot,point], label=POSE_BODY_25_BODY_PARTS[point], alpha=0.8)
    i_plot = features[:points_to_plot,point].argmax()
    if features[i_plot,point] != 0:
        ax.text(i_plot+0.1, features[i_plot,point] , POSE_BODY_25_BODY_PARTS[point])
#plt.legend()
plt.title(f"Pose Features ({n_frames} Frames Standard Deviation)\nVideo: tkzdanzsA0A [0:05-0:10]")
plt.ylabel(f"Average Standard Deviation ({n_frames} Frames)")
plt.xlabel("Second in Segment")
x_ticks = np.arange(0,points_to_plot+1,n_frames)
x_labels = []
ticks_per_second = int(30/n_frames)
for tick in x_ticks:
    if tick % 6 == 0:
        value = int(tick/ticks_per_second)
        x_labels.append(f"{tick/ticks_per_second}")
    else:
        x_labels.append("")
plt.xticks(x_ticks,x_labels)
#box = ax.get_position()
#ax.set_position([box.x0, box.y0 + box.height * 0.1,
#                 box.width, box.height * 0.9])
#ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.10),
#          fancybox=True, ncol=10)
plt.show()
plt.clf()

plt_1 = plt.figure(figsize=(6,4))
ax = plt.subplot(111)
for point in range(21):
    plt.plot(np.arange(points_to_plot), features_h_r[:points_to_plot,point], label=point, alpha=0.8)
    i_plot = features_h_r[:points_to_plot,point].argmax()
    if features[i_plot,point] != 0:
        plt.text(i_plot, features_h_r[i_plot,point], point)
#plt.legend()
plt.title(f"Right Hand Features ({n_frames} Frames Standard Deviation)\nVideo: tkzdanzsA0A [0:05-0:10]")
plt.ylabel(f"Average Standard Deviation ({n_frames} Frames)")
plt.xlabel("Second in Segment")
x_ticks = np.arange(0,points_to_plot+1,n_frames)
x_labels = []
ticks_per_second = int(30/n_frames)
for tick in x_ticks:
    if tick % 6 == 0:
        value = int(tick/ticks_per_second)
        x_labels.append(f"{tick/ticks_per_second}")
    else:
        x_labels.append("")
plt.xticks(x_ticks,x_labels)
#box = ax.get_position()
#ax.set_position([box.x0, box.y0 + box.height * 0.1,
#                 box.width, box.height * 0.9])
#ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.10),
#          fancybox=True, ncol=10)        
plt.show()