import torch
import torch.nn as nn

def conv_block(in_channels, out_channels, kernel_size, stride, padding):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.SiLU()
    )

class CSPBlock(nn.Module):
    def __init__(self, in_channels, out_channels, ratio=0.5):
        super().__init__()
        mid_channels = int(in_channels * ratio)
        self.branch1 = conv_block(in_channels, mid_channels, 1, 1, 0)
        self.branch2 = nn.Sequential(
            conv_block(in_channels, mid_channels, 1, 1, 0),
            conv_block(mid_channels, mid_channels, 3, 1, 1)
        )
        self.bottleneck = conv_block(mid_channels + mid_channels, out_channels, 1, 1, 0)

    def forward(self, x):
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        return self.bottleneck(torch.cat([b1, b2], dim=1))

class DarknetBackbone(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()
        # P3: 96x160
        # P4: 48x80
        # P5: 24x40
        self.stem = conv_block(in_channels, 32, 3, 2, 1)   # 192x320
        self.layer1 = conv_block(32, 64, 3, 2, 1)          # 96x160 (P3)
        self.layer2 = CSPBlock(64, 128)                    # 96x160
        self.layer3 = conv_block(128, 256, 3, 2, 1)        # 48x80 (P4)
        self.layer4 = CSPBlock(256, 512)                   # 48x80
        self.layer5 = conv_block(512, 1024, 3, 2, 1)       # 24x40 (P5)

    def forward(self, x):
        x = self.stem(x)
        p3 = self.layer1(x)
        p3_ext = self.layer2(p3)
        p4 = self.layer3(p3_ext)
        p4_ext = self.layer4(p4)
        p5 = self.layer5(p4_ext)
        return [p3, p4, p5]
