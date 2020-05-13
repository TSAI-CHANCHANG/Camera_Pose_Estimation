#  network struct reference: https://iambigboss.top/post/59955_1_1.html
import torch
import torch.nn as nn
import cv2
import numpy as np
from torchvision import models
import torchvision.transforms as transforms
from torch.hub import load_state_dict_from_url
from PIL import Image
import matplotlib.pyplot as plt
from combine_loss import CombineLoss
from rot_vec_loss import RotVecLoss
from invert import tran_matrix_2_vec


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
# some parameters
index = 0
k = 1
path = './'
scene_name = 'office'
seq = 'seq-01'
# step1: prepare network
MyResNet18 = ModifiedResNet18().cuda()
# from torchsummary import summary
# summary(MyResNet18, (3, 960, 640))
# print(MyResNet18)
# step2: select optimizer and loss function
optimizer = torch.optim.SGD(MyResNet18.parameters(), lr=0.01, momentum=0.9)
trans_loss_func = torch.nn.L1Loss()
rot_loss_func = RotVecLoss()
combine_loss_func = torch.nn.L1Loss()
running_loss = torch.zeros(1).cuda()
running_rot_loss = torch.zeros(1).cuda()
running_trans_loss = torch.zeros(1).cuda()
loss_list = []
rot_loss_list = []
trans_loss_list = []
# step3: start training
for epoch in range(10):
    for index in range(1000):
        # first, load training data
        if index + k >= 1000:
            break
        temp1 = str(index)
        temp2 = str(index + k)
        while len(temp1) < 6:
            temp1 = '0' + temp1
        while len(temp2) < 6:
            temp2 = '0' + temp2
        this_frame_path = path + scene_name + '/' + seq + '/frame-' + temp1 + '.color.png'
        pair_frame_path = path + scene_name + '/' + seq + '/frame-' + temp2 + '.color.png'
        ground_truth_path = path + scene_name + '/' + seq + '/' + 'pair_' + str(k) + '/' + str(index) + ' ' + str(index+k) + '.txt'
        this_frame = np.array(Image.open(this_frame_path))
        pair_frame = np.array(Image.open(pair_frame_path))
        final_img = np.append(this_frame, pair_frame).reshape(960, 640, 3)
        input_img = Image.fromarray(final_img)
        # Image._show(input_img)
        input_tensor = preprocess(input_img)
        input_batch = input_tensor.unsqueeze(0).cuda()

        # second, prepare ground truth data
        ground_truth_trans_mat = np.loadtxt(ground_truth_path).reshape(4, 4)
        rot_vec, trans_vec = tran_matrix_2_vec(ground_truth_trans_mat)
        pose_data = torch.from_numpy(np.append(rot_vec, trans_vec)).cuda()
        # print(trans_mat)

        # third, feed the data to the network
        optimizer.zero_grad()
        prediction = MyResNet18(input_batch)
        combine_loss = combine_loss_func(prediction.double().view(6), pose_data).cuda()
        trans_loss = trans_loss_func(prediction.double().view(6)[3:6], pose_data[3:6]).cuda()
        combine_loss.backward()
        optimizer.step()

        # print loss
        rot_loss = rot_loss_func(prediction.double().view(6)[0:3].cpu().detach(), ground_truth_trans_mat[0:3, 0:3])
        running_loss += combine_loss.item()
        running_rot_loss += rot_loss.item()
        running_trans_loss += trans_loss.item()
        if (index % 200 == 199) | (index + k == 999):
            print('[%d, %5d] loss: %.3f' % (epoch+1, index+1, running_loss/200))
            print('[%d, %5d] rot loss: %.3f' % (epoch+1, index+1, running_rot_loss/200))
            print('[%d, %5d] trans loss: %.3f' % (epoch+1, index+1, running_trans_loss/200))
            loss_list.append(running_loss / 200)
            rot_loss_list.append(running_rot_loss / 200)
            trans_loss_list.append(running_trans_loss / 200)
            running_loss = 0
            running_rot_loss = 0
            running_trans_loss = 0
torch.save(MyResNet18, scene_name + '_' + seq + '_pair_' + str(k) + '_Pose_estimate_net.pkl')
x = []
for i in range(50):
    x.append(0.2 * (i + 1))
plt.figure(figsize=(8, 4))
plt.plot(x, loss_list, label="$loss$", color="red", linewidth=2)
plt.xlabel("epoch")
plt.ylabel("loss")
plt.savefig(scene_name + '_' + seq + '_pair_' + str(k) + "_loss.png")
plt.show()

plt.figure(figsize=(8, 4))
plt.plot(x, rot_loss_list, label="$loss$", color="red", linewidth=2)
plt.xlabel("epoch")
plt.ylabel("rot_loss")
plt.savefig(scene_name + '_' + seq + '_pair_' + str(k) + "_rot_loss.png")
plt.show()

plt.figure(figsize=(8, 4))
plt.plot(x, trans_loss_list, label="$loss$", color="red", linewidth=2)
plt.xlabel("epoch")
plt.ylabel("trans_loss")
plt.savefig(scene_name + '_' + seq + '_pair_' + str(k) + "_trans_loss.png")
plt.show()