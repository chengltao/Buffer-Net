import torch
import torch.nn as nn
import torch.nn.functional as F
from math import ceil, floor

class BufferBlock(nn.Module):
    def __init__(self, in_planes, planes, stride=1):
        super(BufferBlock, self).__init__()
        self.downsampling = nn.Conv3d(in_planes, planes, kernel_size=(3,3,3),
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        self.buffering1 = nn.Conv3d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(planes)
        self.buffering2 = nn.Conv3d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn3 = nn.BatchNorm3d(planes)
        self.buffering3 = nn.Conv3d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn4 = nn.BatchNorm3d(planes)

    def forward(self, x):
        out = F.relu(self.bn1(self.downsampling(x)))
        out = F.relu(self.bn2(self.buffering1(out)))
        out = F.relu(self.bn3(self.buffering2(out)))
        out = F.relu(self.bn4(self.buffering3(out)))
        return out


class BufferNet(nn.Module):
    def __init__(self, block, num_classes=11):
        super(BufferNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv3d(1, 64, kernel_size=(5,1,1),
                               stride=(2,1,1), padding=(0,0,0), bias=False)
        self.bn1 = nn.BatchNorm3d(64)

        self.BufferBlock1 = self._block(block, 64, 64,  stride=2)
        self.BufferBlock2 = self._block(block, 64, 128, stride=2)
        self.BufferBlock3 = self._block(block, 128, 256,  stride=2)
        self.BufferBlock4 = self._block(block, 256, 512, stride=2)
        self.linear = nn.Linear(1536, num_classes)
    def _block(self, block, in_planes, planes, stride):
        #layers = []
        #layers.append(block(self.in_planes, planes, stride))
        return nn.Sequential(block(in_planes, planes, stride))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))

        out = self.BufferBlock1(out)

        out = self.BufferBlock2(out)

        out = self.BufferBlock3(out)

        out = self.BufferBlock4(out)

        out = F.avg_pool3d(out, 2)


        out = out.view(out.size(0), -1)

        out = self.linear(out)

        return F.log_softmax(out, dim=1)

def Buffer():
    return BufferNet(BufferBlock)

print(Buffer())