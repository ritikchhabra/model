"""
YOLOP – You Only Look Once for Panoptic driving Perception.

Three heads:
    Detection Head         – objectness heatmap per FPN scale
    Drivable Area Head     – 3-class segmentation (bg / direct / alternative)
    Lane Detection Head    – binary lane mask

Backbone output channels (post-CSP fix):
    p3: 128ch   96x160
    p4: 512ch   48x80
    p5: 1024ch  24x40
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from yolop.backbone.darknet import DarknetBackbone


def conv_block(in_ch, out_ch, k=3, s=1, p=1):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, k, s, p, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.SiLU(inplace=True),
    )


class DetHead(nn.Module):
    """Per-scale objectness heatmap head. Output: [B, 1, H_s, W_s]."""
    def __init__(self, in_ch, num_classes=10):
        super().__init__()
        self.conv = conv_block(in_ch, in_ch // 2, 3, 1, 1)
        self.out  = nn.Conv2d(in_ch // 2, 1 + num_classes, 1)

    def forward(self, x):
        return self.out(self.conv(x))


class SegHead(nn.Module):
    """
    Drivable Area segmentation head.
    Upsamples p3 (stride-4) to full input size via bilinear + 1x1 conv.
    Output: [B, num_classes, H, W]
    """
    def __init__(self, in_ch, num_classes=3):
        super().__init__()
        self.conv1 = conv_block(in_ch, 64, 3, 1, 1)
        self.conv2 = nn.Conv2d(64, num_classes, 1)

    def forward(self, x, target_size):
        x = F.interpolate(x, size=target_size, mode='bilinear', align_corners=False)
        x = self.conv1(x)
        return self.conv2(x)


class LaneHead(nn.Module):
    """
    Lane detection head.
    Upsamples p3 (stride-4) to full input size.
    Output: [B, 1, H, W]
    """
    def __init__(self, in_ch):
        super().__init__()
        self.conv1 = conv_block(in_ch, 32, 3, 1, 1)
        self.conv2 = nn.Conv2d(32, 1, 1)

    def forward(self, x, target_size):
        x = F.interpolate(x, size=target_size, mode='bilinear', align_corners=False)
        x = self.conv1(x)
        return self.conv2(x)


class YOLOP(nn.Module):
    """
    Args:
        num_classes_det: detection classes (10 for BDD100K)
        num_classes_seg: drivable area classes (3: bg/direct/alternative)
        img_size: (H, W) model input size
    """

    def __init__(self, num_classes_det=10, num_classes_seg=3, img_size=(384, 640)):
        super().__init__()
        self.img_h, self.img_w = img_size

        self.backbone = DarknetBackbone()

        # Detection heads per scale
        self.det_head = nn.ModuleDict({
            'p3': DetHead(128,  num_classes_det),   # 128ch
            'p4': DetHead(512,  num_classes_det),   # 512ch
            'p5': DetHead(1024, num_classes_det),   # 1024ch
        })

        # Drivable area head (uses p3)
        self.drv_head  = SegHead(128, num_classes_seg)

        # Lane head (uses p3)
        self.lane_head = LaneHead(128)

    def forward(self, x):
        target_h, target_w = x.shape[2], x.shape[3]
        p3, p4, p5 = self.backbone(x)

        det_out = {
            'p3': self.det_head['p3'](p3),
            'p4': self.det_head['p4'](p4),
            'p5': self.det_head['p5'](p5),
        }

        drv_out  = self.drv_head(p3,  (target_h, target_w))
        lane_out = self.lane_head(p3, (target_h, target_w))

        return {
            'det':  det_out,
            'drv':  drv_out,    # [B, 3, H, W]
            'lane': lane_out,   # [B, 1, H, W]
        }
