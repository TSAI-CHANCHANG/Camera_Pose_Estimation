import math
from math import log
import numpy as np

index = 0
k = 20
path = './'
scene_name = 'office'
seq = 'seq-02'
for index in range(1000):
    if index + k >= 1000:
        break
    if index % 50 == 0:
        print("Has saved " + str(index))
    if index != 0:
        prefix_num = 5 - math.floor(log(index, 10))
    else:
        prefix_num = 5
    temp1 = str(index)
    temp2 = str(index + k)
    while len(temp1) < 6:
        temp1 = '0' + temp1
    while len(temp2) < 6:
        temp2 = '0' + temp2
    this_frame_pose_path = path + scene_name + '/' + seq + '/frame-' + temp1 + '.pose.txt'
    pair_frame_pose_path = path + scene_name + '/' + seq + '/frame-' + temp2 + '.pose.txt'
    this_frame_pose = np.loadtxt(this_frame_pose_path).reshape(4, 4)
    pair_frame_pose = np.loadtxt(pair_frame_pose_path).reshape(4, 4)
    this_frame_rot = this_frame_pose[0:3, 0:3]
    pair_frame_rot = pair_frame_pose[0:3, 0:3]
    this_frame_trans = this_frame_pose[0:3, 3]
    pair_frame_trans = pair_frame_pose[0:3, 3]
    relative_rot = this_frame_rot.T @ pair_frame_rot
    relative_trans = pair_frame_trans - this_frame_trans
    relative_matrix = np.zeros((4, 4), dtype=np.float64)
    relative_matrix[0:3, 0:3] = relative_rot
    relative_matrix[0:3, 3] = relative_trans
    relative_matrix[3, 3] = 1
    savePath = './'+scene_name+'./'+seq+'./'+'pair_'+str(k)+'/'+str(index)+' '+str(index+k)+'.txt'
    np.savetxt(savePath, relative_matrix, fmt='%1.7e')
# loadMatrix = np.loadtxt(path)
# print(loadMatrix)
# print(relative_rot)
# print(relative_trans)
# print(relative_matrix)