"""
Lightweight CSP-Darknet backbone for YOLOP.

Output feature maps (post-CSP):
    p3_ext: stride-4,  channels=128   (96 x 160 for 384x640 input)
    p4_ext: stride-8,  channels=512   (48 x 80)
    p5:     stride-16, channels=1024  (24 x 40)

Bug fixed: previously returned pre-CSP p3/p4 instead of post-CSP p3_ext/p4_ext.
"""

import torch
import torch.nn as nn


def conv_block(in_ch, out_ch, k, s, p):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, k, s, p, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.SiLU(inplace=True),
    )


class CSPBlock(nn.Module):
    def __init__(self, in_ch, out_ch, ratio=0.5):
        super().__init__()
        mid = int(in_ch * ratio)
        self.branch1 = conv_block(in_ch, mid, 1, 1, 0)
        self.branch2 = nn.Sequential(
            conv_block(in_ch, mid, 1, 1, 0),
            conv_block(mid, mid, 3, 1, 1),
        )
        self.bottleneck = conv_block(mid * 2, out_ch, 1, 1, 0)

    def forward(self, x):
        return self.bottleneck(torch.cat([self.branch1(x), self.branch2(x)], dim=1))


class DarknetBackbone(nn.Module):
    """
    Input: [B, 3, 384, 640]
    stem   → 32ch  192x320
    layer1 → 64ch   96x160  (P3 raw)
    layer2 → 128ch  96x160  (P3 ext, returned as p3)
    layer3 → 256ch  48x80   (stride 2 downsample)
    layer4 → 512ch  48x80   (P4 ext, returned as p4)
    layer5 → 1024ch 24x40   (P5, returned as p5)
    """

    def __init__(self, in_channels=3):
        super().__init__()
        self.stem   = conv_block(in_channels, 32,   3, 2, 1)   # /2
        self.layer1 = conv_block(32,  64,  3, 2, 1)            # /4  → 96x160
        self.layer2 = CSPBlock(64,  128)                        # stays 96x160
        self.layer3 = conv_block(128, 256, 3, 2, 1)            # /8  → 48x80
        self.layer4 = CSPBlock(256, 512)                        # stays 48x80
        self.layer5 = conv_block(512, 1024, 3, 2, 1)           # /16 → 24x40

    def forward(self, x):
        x     = self.stem(x)
        _p3   = self.layer1(x)
        p3    = self.layer2(_p3)   # 128ch  96x160
        _p4   = self.layer3(p3)
        p4    = self.layer4(_p4)   # 512ch  48x80
        p5    = self.layer5(p4)    # 1024ch 24x40
        return [p3, p4, p5]
