"""
pytorch implementation of the resnet.py
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

# Xavier : nn.init.xavier_normal_
# He : nn.init.kaiming_normal_
# Normal : nn.init.normal_
# l2_decay : nn.MSELoss

weight_init = nn.init.kaiming_normal_
weight_regularizer = nn.MSELoss()

##################################################################################
# Layer
##################################################################################


class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=4, stride=2, padding=1, use_bias=True):
        super(ConvLayer, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=use_bias)
        weight_init(self.conv.weight)
        if use_bias:
            nn.init.constant_(self.conv.bias, 0)

    def forward(self, x):
        # print("x.shape in ConvLayer, line21, resnet.py: ", x.shape)
        # [2, 24, 172, 172]
        return self.conv(x)


class FullyConnectedLayer(nn.Module):
    def __init__(self, in_features, out_features, use_bias=True):
        super(FullyConnectedLayer, self).__init__()
        self.fc = nn.Linear(in_features, out_features, bias=use_bias)
        weight_init(self.fc.weight)
        if use_bias:
            nn.init.constant_(self.fc.bias, 0)

    def forward(self, x):
        return self.fc(x.view(x.size(0), -1))


# use when resnet_n < 50
class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, downsample=False):
        super(ResBlock, self).__init__()
        self.downsample = downsample

        # error
        self.conv1 = ConvLayer(in_channels, out_channels, kernel_size=3, stride=2 if downsample else 1)

        self.conv2 = ConvLayer(out_channels, out_channels, kernel_size=3, stride=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.downsample_conv = ConvLayer(in_channels, out_channels, kernel_size=1, stride=2 if downsample else 1) if downsample else None

    def forward(self, x):
        #
        # print("x.shape in ResBlock forward, line 60, resnet.py: ", x.shape)
        # [2, 24, 172, 172]
        residual = self.downsample_conv(x) if self.downsample else x

        # error
        out = self.bn1(self.conv1(x))
        out = self.bn2(self.conv2(out))
        out += residual
        out = self.relu(out)
        return out


# use when resnet_n >= 50
class BottleResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, downsample=False):
        super(BottleResBlock, self).__init__()
        self.downsample = downsample
        self.conv1x1_front = ConvLayer(in_channels, out_channels, kernel_size=1)
        self.conv3x3 = ConvLayer(out_channels, out_channels, kernel_size=3, stride=2 if downsample else 1, padding=1)
        self.conv1x1_back = ConvLayer(out_channels, out_channels * 4, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.bn3 = nn.BatchNorm2d(out_channels * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample_conv = ConvLayer(in_channels, out_channels * 4, kernel_size=1, stride=2 if downsample else 1) if downsample else ConvLayer(in_channels, out_channels * 4, kernel_size=1)

    def forward(self, x):
        residual = self.downsample_conv(x)
        out = self.bn1(self.conv1x1_front(x))
        out = self.relu(out)
        out = self.bn2(self.conv3x3(out))
        out = self.relu(out)
        out = self.bn3(self.conv1x1_back(out))
        out += residual
        out = self.relu(out)
        return out


def get_residual_layer(res_n):
    if res_n == 18:
        return [2, 2, 2, 2]
    elif res_n == 34:
        return [3, 4, 6, 3]
    elif res_n == 50:
        return [3, 4, 6, 3]
    elif res_n == 101:
        return [3, 4, 23, 3]
    elif res_n == 152:
        return [3, 8, 36, 3]
    else:
        raise ValueError('Invalid ResNet depth')

##################################################################################
# Sampling
##################################################################################


def global_avg_pooling(x):
    return F.adaptive_avg_pool2d(x, (1, 1))


def avg_pooling(x):
    return F.avg_pool2d(x, kernel_size=2, stride=2, padding=0)


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

##################################################################################
# Loss function
##################################################################################


class ClassificationLoss(nn.Module):
    def __init__(self):
        super(ClassificationLoss, self).__init__()

    def forward(self, logit, label):
        loss = F.cross_entropy(logit, label)
        prediction = torch.argmax(logit, dim=-1) == label # torch.argmax(label, dim=-1)
        accuracy = torch.mean(prediction.float())
        return loss, accuracy

##################################################################################
# ResNet
##################################################################################


class ResNet(nn.Module):
    def __init__(self, res_n=18, feature_size=126, ch=64):
        super(ResNet, self).__init__()
        self.res_n = res_n
        self.feature_size = feature_size
        self.ch = ch
        self.residual_block = ResBlock if res_n < 50 else BottleResBlock
        self.residual_list = get_residual_layer(res_n)
        self.conv1 = ConvLayer(3, ch, kernel_size=3, stride=1)
        self.layers = self._make_layers()
        self.bn = nn.BatchNorm2d(ch * 8)
        self.relu = nn.ReLU(inplace=True)
        self.global_pool = global_avg_pooling
        self.fc = FullyConnectedLayer(ch * 8, feature_size)

    def _make_layers(self):
        layers = []
        in_channels = self.ch
        for i in range(len(self.residual_list)):
            layers.append(self.residual_block(in_channels, self.ch * (2**i), downsample=True if i > 0 else False))
            in_channels = self.ch * (2**i)
            for j in range(1, self.residual_list[i]):
                layers.append(self.residual_block(in_channels, in_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.layers(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.global_pool(x)
        x = self.fc(x)
        return x
