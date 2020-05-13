import cv2
import torch
import torch.nn as nn

from kornia.geometry import conversions
from invert import tran_matrix_2_vec

class CombineLoss(nn.Module):
    def __init__(self):
        super(CombineLoss, self).__init__()
        return

    def forward(self, prediction, ground_truth_matrix):
        pred_rot_vec = prediction[0:3]
        pred_trans_vec = prediction[3:6]
        g_t_trans_vec = ground_truth_matrix[3, 0:3].t()
        g_t_matrix = ground_truth_matrix[0:3, 0:3]
        pred_matrix = conversions.angle_axis_to_rotation_matrix(pred_rot_vec).reshape(3, 3)
        R = torch.matmul(g_t_matrix.T, pred_matrix)
        # g_t_matrix denote as m1, pred_matrix as m2
        # since m1, m2 are all rotation matrices
        # m1.t = m1^-1 (inverse matrix)
        # we want to obtain the loss has physical loss
        # then we can choose relative transform matrix R
        # m1 @ R = m2
        # so R = m1.t @ m2
        # and R is a 3x3 matrix, we need to transfer it back to rot_vec by Rodrigues
        rot_loss = np.linalg.norm(cv2.Rodrigues(R)[0])
        # rot_loss = torch.from_numpy(np.array(rot_loss))  # unit: radian
        # trans_loss = np.linalg.norm(g_t_trans_vec - pred_trans_vec)
        loss = rot_loss
        return loss


# input = torch.rand(1, 3, 3)
# output = conversions.rotation_matrix_to_angle_axis(input)  # Nx3
# print(input)
# import numpy as np
# mat = np.loadtxt("./office/seq-01/pair_1/1 2.txt")
# rot_vec, trans_vec = tran_matrix_2_vec(mat)
# print(mat)
# print(rot_vec)
# print(trans_vec)
# print(cv2.Rodrigues(rot_vec)[0])
# print(cv2.Rodrigues(mat[0:3, 0:3])[0])
# mat_tensor = torch.from_numpy(mat)
# mat_t = conversions.angle_axis_to_rotation_matrix(torch.from_numpy(rot_vec)).reshape(3, 3)
# print(mat_t)
# temp = mat_tensor[0:3, 0:3].clone().reshape(1, 3, 3)
# print(mat_tensor)
# # print(temp.stride())
# print(temp)
#
# rot_tensor = conversions.rotation_matrix_to_angle_axis(temp.reshape(1, 3, 3))
# print(rot_tensor)
# print("finish!")
# import numpy as np
# a = [[1,2,3],
#      [4,5,6]]
# a_np = np.array(a)
# tensorA = torch.from_numpy(a_np)
# b = [[1],
#      [4],
#      [5]]
# b_np = np.array(b)
# tensorB = torch.from_numpy(b_np)
# print(torch.matmul(tensorA, tensorB))

