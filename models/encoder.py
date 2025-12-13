import torch
import torch.nn as nn
import torch.nn.functional as F

def conv3x3(in_ch, out_ch, stride=1):
    return nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False)

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = conv3x3(in_ch, out_ch, stride)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = conv3x3(out_ch, out_ch, 1)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.down = None
        if stride != 1 or in_ch != out_ch:
            self.down = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )

    def forward(self, x):
        identity = x
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        if self.down is not None:
            identity = self.down(identity)
        out = F.relu(out + identity, inplace=True)
        return out

class SmallResNet(nn.Module):
    def __init__(self, layers=(2,2,2,2), in_ch=1):
        super().__init__()
        self.inplanes = 64
        self.conv1 = conv3x3(in_ch, 64, stride=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(64, layers[0], stride=1)
        self.layer2 = self._make_layer(128, layers[1], stride=2)
        self.layer3 = self._make_layer(256, layers[2], stride=2)
        self.layer4 = self._make_layer(512, layers[3], stride=2)
        self.avg = nn.AdaptiveAvgPool2d((1,1))

    def _make_layer(self, planes, blocks, stride):
        layers = [BasicBlock(self.inplanes, planes, stride=stride)]
        self.inplanes = planes
        for _ in range(1, blocks):
            layers.append(BasicBlock(self.inplanes, planes, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)), inplace=True)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avg(x)
        return torch.flatten(x, 1)

class ResNetMNISTEncoder(nn.Module):
    def __init__(self, emb_dim: int = 256):
        super().__init__()
        self.backbone = SmallResNet()
        self.proj = nn.Sequential(
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, emb_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.backbone(x)
        z = self.proj(h)
        return F.normalize(z, dim=-1)
