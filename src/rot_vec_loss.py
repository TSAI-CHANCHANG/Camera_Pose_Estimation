import cv2
import torch
import torch.nn as nn
import numpy as np


class RotVecLoss(nn.Module):
    def __init__(self):
        super(RotVecLoss, self).__init__()
        return

    def forward(self, prediction, ground_truth_matrix):
        pred_vec = prediction.numpy()
        g_t_matrix = ground_truth_matrix#.numpy()
        pred_matrix = cv2.Rodrigues(pred_vec)[0]
        R = g_t_matrix.T @ pred_matrix
        # g_t_matrix denote as m1, pred_matrix as m2
        # since m1, m2 are all rotation matrices
        # m1.t = m1^-1 (inverse matrix)
        # we want to obtain the loss has physical loss
        # then we can choose relative transform matrix R
        # m1 @ R = m2
        # so R = m1.t @ m2
        # and R is a 3x3 matrix, we need to transfer it back to rot_vec by Rodrigues
        loss = np.linalg.norm(cv2.Rodrigues(R)[0])
        loss = torch.from_numpy(np.array(loss))  # unit: radian
        return loss
