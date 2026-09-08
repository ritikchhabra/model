import torch
import torch.nn as nn
from yolop.backbone.darknet import DarknetBackbone

class YOLOP(nn.Module):
    def __init__(self, num_classes_det=10, num_classes_seg=19, num_lanes=4):
        super().__init__()
        self.backbone = DarknetBackbone()
        
        # Detection Head
        # p3: 64, p4: 256, p5: 1024
        self.det_head = nn.ModuleDict({
            'p3': nn.Conv2d(64, num_classes_det + 5, 1),
            'p4': nn.Conv2d(256, num_classes_det + 5, 1),
            'p5': nn.Conv2d(1024, num_classes_det + 5, 1)
        })
        
        # Drivable Area Head
        # p3: 64, p4: 256
        self.drv_head = nn.ModuleDict({
            'p3': nn.Conv2d(64, num_classes_seg, 1),
            'p4': nn.Conv2d(256, num_classes_seg, 1)
        })
        
        # Lane Head
        self.lane_head = nn.Conv2d(64, num_lanes, 1)

    def forward(self, x):
        features = self.backbone(x)
        p3, p4, p5 = features[0], features[1], features[2]
        
        det_out = {
            'p3': self.det_head['p3'](p3),
            'p4': self.det_head['p4'](p4),
            'p5': self.det_head['p5'](p5)
        }
        
        drv_out = {
            'p3': self.drv_head['p3'](p3),
            'p4': self.drv_head['p4'](p4)
        }
        
        lane_out = self.lane_head(p3)
        
        return {
            "det": det_out,
            "drv": drv_out,
            "lane": lane_out
        }
