import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import models
import utils.config as config


class VGG16TensorflowAlign(nn.Module):

    def __init__(self, n_classes_input=3):
        super(VGG16TensorflowAlign, self).__init__()

        self.features_dim = 960
        # this is to align with tensorflow padding (with stride)
        # https://bugxch.github.io/tf%E4%B8%AD%E7%9A%84padding%E6%96%B9%E5%BC%8FSAME%E5%92%8CVALID%E6%9C%89%E4%BB%80%E4%B9%88%E5%8C%BA%E5%88%AB/
        self.same_padding = nn.ZeroPad2d(1)
        self.tf_padding = nn.ZeroPad2d((0, 1, 0, 1))
        self.tf_padding_2 = nn.ZeroPad2d((1, 2, 1, 2))

        self.conv0_1 = nn.Conv2d(n_classes_input, 16, 3, stride=1, padding=0)
        self.conv0_2 = nn.Conv2d(16, 16, 3, stride=1, padding=0)

        self.conv1_1 = nn.Conv2d(16, 32, 3, stride=2, padding=0)  # 224 -> 112
        self.conv1_2 = nn.Conv2d(32, 32, 3, stride=1, padding=0)
        self.conv1_3 = nn.Conv2d(32, 32, 3, stride=1, padding=0)

        self.conv2_1 = nn.Conv2d(32, 64, 3, stride=2, padding=0)  # 112 -> 56
        self.conv2_2 = nn.Conv2d(64, 64, 3, stride=1, padding=0)
        self.conv2_3 = nn.Conv2d(64, 64, 3, stride=1, padding=0)

        self.conv3_1 = nn.Conv2d(64, 128, 3, stride=2, padding=0)  # 56 -> 28
        self.conv3_2 = nn.Conv2d(128, 128, 3, stride=1, padding=0)
        self.conv3_3 = nn.Conv2d(128, 128, 3, stride=1, padding=0)

        self.conv4_1 = nn.Conv2d(128, 256, 5, stride=2, padding=0)  # 28 -> 14
        self.conv4_2 = nn.Conv2d(256, 256, 3, stride=1, padding=0)
        self.conv4_3 = nn.Conv2d(256, 256, 3, stride=1, padding=0)

        self.conv5_1 = nn.Conv2d(256, 512, 5, stride=2, padding=0)  # 14 -> 7
        self.conv5_2 = nn.Conv2d(512, 512, 3, stride=1, padding=0)
        self.conv5_3 = nn.Conv2d(512, 512, 3, stride=1, padding=0)
        self.conv5_4 = nn.Conv2d(512, 512, 3, stride=1, padding=0)

    def forward(self, img):
        img = F.relu(self.conv0_1(self.same_padding(img)))
        img = F.relu(self.conv0_2(self.same_padding(img)))

        img = F.relu(self.conv1_1(self.tf_padding(img)))
        img = F.relu(self.conv1_2(self.same_padding(img)))
        img = F.relu(self.conv1_3(self.same_padding(img)))

        img = F.relu(self.conv2_1(self.tf_padding(img)))
        img = F.relu(self.conv2_2(self.same_padding(img)))
        img = F.relu(self.conv2_3(self.same_padding(img)))
        img2 = img

        img = F.relu(self.conv3_1(self.tf_padding(img)))
        img = F.relu(self.conv3_2(self.same_padding(img)))
        img = F.relu(self.conv3_3(self.same_padding(img)))
        img3 = img

        img = F.relu(self.conv4_1(self.tf_padding_2(img)))
        img = F.relu(self.conv4_2(self.same_padding(img)))
        img = F.relu(self.conv4_3(self.same_padding(img)))
        img4 = img

        img = F.relu(self.conv5_1(self.tf_padding_2(img)))
        img = F.relu(self.conv5_2(self.same_padding(img)))
        img = F.relu(self.conv5_3(self.same_padding(img)))
        img = F.relu(self.conv5_4(self.same_padding(img)))
        img5 = img

        return [img2, img3, img4, img5]


class MultiViewVGG16TensorflowAlign(nn.Module):

    def __init__(self, n_classes_input=3):
        super(MultiViewVGG16TensorflowAlign, self).__init__()

        self.features_dim = 16 + 32 + 64
        # this is to align with tensorflow padding (with stride)
        # https://bugxch.github.io/tf%E4%B8%AD%E7%9A%84padding%E6%96%B9%E5%BC%8FSAME%E5%92%8CVALID%E6%9C%89%E4%BB%80%E4%B9%88%E5%8C%BA%E5%88%AB/
        self.same_padding = nn.ZeroPad2d(1)
        self.tf_padding = nn.ZeroPad2d((0, 1, 0, 1))
        self.tf_padding_2 = nn.ZeroPad2d((1, 2, 1, 2))

        self.conv0_1 = nn.Conv2d(n_classes_input, 16, 3, stride=1, padding=0)
        self.conv0_2 = nn.Conv2d(16, 16, 3, stride=1, padding=0)

        self.conv1_1 = nn.Conv2d(16, 32, 3, stride=2, padding=0)  # 224 -> 112
        self.conv1_2 = nn.Conv2d(32, 32, 3, stride=1, padding=0)
        self.conv1_3 = nn.Conv2d(32, 32, 3, stride=1, padding=0)

        self.conv2_1 = nn.Conv2d(32, 64, 3, stride=2, padding=0)  # 112 -> 56
        self.conv2_2 = nn.Conv2d(64, 64, 3, stride=1, padding=0)
        self.conv2_3 = nn.Conv2d(64, 64, 3, stride=1, padding=0)

        self.conv3_1 = nn.Conv2d(64, 128, 3, stride=2, padding=0)  # 56 -> 28
        self.conv3_2 = nn.Conv2d(128, 128, 3, stride=1, padding=0)
        self.conv3_3 = nn.Conv2d(128, 128, 3, stride=1, padding=0)

        self.conv4_1 = nn.Conv2d(128, 256, 5, stride=2, padding=0)  # 28 -> 14
        self.conv4_2 = nn.Conv2d(256, 256, 3, stride=1, padding=0)
        self.conv4_3 = nn.Conv2d(256, 256, 3, stride=1, padding=0)

        self.conv5_1 = nn.Conv2d(256, 512, 5, stride=2, padding=0)  # 14 -> 7
        self.conv5_2 = nn.Conv2d(512, 512, 3, stride=1, padding=0)
        self.conv5_3 = nn.Conv2d(512, 512, 3, stride=1, padding=0)
        self.conv5_4 = nn.Conv2d(512, 512, 3, stride=1, padding=0)

    def forward(self, img):
        img = F.relu(self.conv0_1(self.same_padding(img)))
        img = F.relu(self.conv0_2(self.same_padding(img)))
        img0 = img

        img = F.relu(self.conv1_1(self.tf_padding(img)))
        img = F.relu(self.conv1_2(self.same_padding(img)))
        img = F.relu(self.conv1_3(self.same_padding(img)))
        img1 = img

        img = F.relu(self.conv2_1(self.tf_padding(img)))
        img = F.relu(self.conv2_2(self.same_padding(img)))
        img = F.relu(self.conv2_3(self.same_padding(img)))
        img2 = img

        img = F.relu(self.conv3_1(self.tf_padding(img)))
        img = F.relu(self.conv3_2(self.same_padding(img)))
        img = F.relu(self.conv3_3(self.same_padding(img)))
        img3 = img

        img = F.relu(self.conv4_1(self.tf_padding_2(img)))
        img = F.relu(self.conv4_2(self.same_padding(img)))
        img = F.relu(self.conv4_3(self.same_padding(img)))
        img4 = img

        img = F.relu(self.conv5_1(self.tf_padding_2(img)))
        img = F.relu(self.conv5_2(self.same_padding(img)))
        img = F.relu(self.conv5_3(self.same_padding(img)))
        img = F.relu(self.conv5_4(self.same_padding(img)))
        img5 = img

        return [img0, img1, img2]#, img3, img4, img5]
