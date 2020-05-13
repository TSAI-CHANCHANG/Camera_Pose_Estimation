import numpy as np
import cv2
from invert import tran_matrix_2_vec

index = 0
k = 1
path = './'
scene_name = 'office'
seq = 'seq-01'
sampleNum = 1000
rangeNum = sampleNum - k
running_trans_loss = 0
running_rot_loss = 0
for index in range(999):
    separate_prediction = np.loadtxt("prediction_Net_office_seq-01_Pred_office_seq-01.txt").reshape(1000, 6)
    ground_truth_path = path + scene_name + '/' + seq + '/' + 'pair_' + str(k) + '/' + str(index) + ' ' + str(
            index + k) + '.txt'
    ground_truth_trans_mat = np.loadtxt(ground_truth_path).reshape(4, 4)
    this_frame_rot_vec = separate_prediction[index, 0:3]
    this_frame_rot_matrix = cv2.Rodrigues(this_frame_rot_vec)[0]
    pair_frame_rot_vec = separate_prediction[index+k, 0:3]
    pair_frame_rot_matrix = cv2.Rodrigues(pair_frame_rot_vec)[0]
    this_frame_trans_vec = separate_prediction[index, 3:6]
    pair_frame_trans_vec = separate_prediction[index+k, 3:6]

    # calculate the relative rot vec and trans vec from two separate prediction
    relative_rot_matrix = this_frame_rot_matrix.T @ pair_frame_rot_matrix
    relative_trans = pair_frame_trans_vec - this_frame_trans_vec
    ground_truth_trans_vec = ground_truth_trans_mat[0:3, 3]

    # calculate the rot loss
    g_t_matrix = ground_truth_trans_mat[0:3, 0:3]
    pred_matrix = relative_rot_matrix
    R = g_t_matrix.T @ pred_matrix
    rot_loss = np.linalg.norm(cv2.Rodrigues(R)[0])

    # calculate the trans loss
    trans_loss = np.abs(ground_truth_trans_vec - relative_trans).sum()/3
    print('[%4d] rot_loss: %.3f trans_loss: %.3f' %
          (index + 1, rot_loss, trans_loss))
    running_rot_loss += rot_loss
    running_trans_loss += trans_loss
    if index % 1000 == 998:
        print('Average: k = %d rot_loss: %.3f trans_loss: %.3f' %
              (k, running_rot_loss / 999, running_trans_loss / 999))