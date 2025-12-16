import torch.nn as nn
import torch
import torch.nn.functional as F

def conv_block(in_channels, out_channels, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)

class ResidualBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None, drop_prob=0.0, use_dropblock=False, block_size=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = conv_block(in_channels, out_channels)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.activation = nn.LeakyReLU(0.1)
        self.conv2 = conv_block(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = conv_block(out_channels, out_channels)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.pool = nn.MaxPool2d(stride)
        self.downsample = downsample
        self.stride = stride
        self.drop_prob = drop_prob
        self.use_dropblock = use_dropblock
        self.block_size = block_size
        self.batch_count = 0

    def forward(self, x):
        self.batch_count += 1
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.activation(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.activation(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.activation(out)
        out = self.pool(out)
        if self.drop_prob > 0:
            out = F.dropout(out, p=self.drop_prob, training=self.training, inplace=True)
        return out

class CustomResNet(nn.Module):
    def __init__(self, block=ResidualBlock, dropout_prob=1.0, use_avg_pool=False, drop_prob=0.0, dropblock_size=5):
        super(CustomResNet, self).__init__()
        self.in_channels = 3
        self.layer1 = self._build_layer(block, 64, stride=2, drop_prob=drop_prob)
        self.layer2 = self._build_layer(block, 160, stride=2, drop_prob=drop_prob)
        self.layer3 = self._build_layer(block, 320, stride=2, drop_prob=drop_prob, use_dropblock=True, block_size=dropblock_size)
        self.layer4 = self._build_layer(block, 640, stride=2, drop_prob=drop_prob, use_dropblock=True, block_size=dropblock_size)
        self.use_avg_pool = use_avg_pool
        if use_avg_pool:
            self.avgpool = nn.AvgPool2d(5, stride=1)
        self.dropout_prob = dropout_prob
        self.dropout = nn.Dropout(p=1 - self.dropout_prob, inplace=False)
        self.drop_prob = drop_prob

        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

    def _build_layer(self, block, planes, stride=1, drop_prob=0.0, use_dropblock=False, block_size=1):
        downsample = None
        if stride != 1 or self.in_channels != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, planes * block.expansion, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.in_channels, planes, stride, downsample, drop_prob, use_dropblock, block_size))
        self.in_channels = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        if self.use_avg_pool:
            x = self.avgpool(x)
        return x

def custom_resnet12():
    return CustomResNet()
