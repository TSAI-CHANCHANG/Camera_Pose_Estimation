from __future__ import print_function
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision import models
import numpy as np
from invert import tran_matrix_2_vec
from rot_vec_loss import RotVecLoss
from torch.hub import load_state_dict_from_url
from PIL import Image

class ModifiedResNet18(models.ResNet):
    def __init__(self, num_classes=6, pretrained=False, **kwargs):
        super().__init__(block=models.resnet.BasicBlock,
                         layers=[2, 2, 2, 2],
                         num_classes=num_classes, **kwargs)
        if pretrained:
            state_dict = load_state_dict_from_url(models.resnet.model_urls["resnet18"], progress=True)
            self.load_state_dict(state_dict)
        # self.avgpool = nn.AvgPool2d((7, 7))
        self.last_conv = nn.Conv2d(in_channels=self.fc.in_features,
                                         out_channels=num_classes,
                                         kernel_size=1)
        self.last_conv.weight.data.copy_(self.fc.weight.data.view(*self.fc.weight.data.shape, 1, 1))
        self.last_conv.bias.data.copy_(self.fc.bias.data)

    def _forward_impl(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = self.last_conv(x)
        return x
preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])
        ])
index = 0
k = 50
path = './'
scene_name = 'office'
model_scene_name = 'office'
seq = 'seq-02'
model_seq = 'seq-01'
model_k = 1
sampleNum = 1000
rangeNum = sampleNum - k
# step 1: load the trained model
model = torch.load(model_scene_name + '_' + model_seq + '_pair_' + str(model_k) + '_Pose_estimate_net.pkl')
# step 2: set the model into evaluation mode and disable all the change
model.eval()
torch.set_grad_enabled(False)
if torch.cuda.is_available():
    model.cuda()
# step 3: prepare the loss function
loss_func = torch.nn.L1Loss()
rot_loss_func = RotVecLoss()
running_loss = torch.zeros(1)
running_rot_loss = torch.zeros(1)
running_trans_loss = torch.zeros(1)
for index in range(rangeNum):
    if index + k >= sampleNum:
        break
    temp1 = str(index)
    temp2 = str(index + k)
    while len(temp1) < 6:
        temp1 = '0' + temp1
    while len(temp2) < 6:
        temp2 = '0' + temp2
    this_frame_path = path + scene_name + '/' + seq + '/frame-' + temp1 + '.color.png'
    pair_frame_path = path + scene_name + '/' + seq + '/frame-' + temp2 + '.color.png'
    ground_truth_path = path + scene_name + '/' + seq + '/' + 'pair_' + str(k) + '/' + str(index) + ' ' + str(
        index + k) + '.txt'
    this_frame = np.array(Image.open(this_frame_path))
    pair_frame = np.array(Image.open(pair_frame_path))
    final_img = np.append(this_frame, pair_frame).reshape(960, 640, 3)
    input_img = Image.fromarray(final_img)
    # Image._show(input_img)
    input_tensor = preprocess(input_img)
    input_batch = input_tensor.unsqueeze(0).cuda()
    ground_truth_trans_mat = np.loadtxt(ground_truth_path).reshape(4, 4)
    rot_vec, trans_vec = tran_matrix_2_vec(ground_truth_trans_mat)
    pose_data = torch.from_numpy(np.append(rot_vec, trans_vec)).cuda()

    if torch.cuda.is_available():
        input_batch = input_batch.to('cuda:0')
        pose_data = pose_data.to('cuda:0')
    prediction = model(input_batch)

    loss = loss_func(prediction.double().view(6), pose_data).cuda()
    rot_loss = rot_loss_func(prediction.double().view(6)[0:3].cpu(), ground_truth_trans_mat[0:3, 0:3])
    trans_loss = loss_func(prediction.double().view(6)[3:6], pose_data[3:6])

    running_loss += loss
    running_rot_loss += rot_loss
    running_trans_loss += trans_loss
    if (index % 1000 == 998) | (index + k == 999):
        print('[%4d] loss: %.3f rot_loss: %.3f trans_loss: %.3f\n[%4dth] pose_data:' %
              (index + 1, running_loss / 999, running_rot_loss / 999, running_trans_loss / 999, index + 1))
        print(pose_data)
        print('prediction:')
        print(prediction.view(6))

        running_loss = 0
        running_rot_loss = 0
        running_trans_loss = 0