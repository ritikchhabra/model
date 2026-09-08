import torch
import torch.nn as nn
import torch.nn.functional as F

class YOLOPLoss(nn.Module):
    def __init__(self, lambda_det=1.0, lambda_drv=1.0, lambda_lane=1.0):
        super().__init__()
        self.lambda_det = lambda_det
        self.lambda_drv = lambda_drv
        self.lambda_lane = lambda_lane

        # Losses
        self.det_loss_fn = nn.MSELoss()
        self.drv_loss_fn = nn.CrossEntropyLoss()
        self.lane_loss_fn = nn.MSELoss()

    def forward(self, pred, target):
        # pred: {'det': {'p3': ..., 'p4': ..., 'p5': ...}, 'drv': {'p3': ..., 'p4': ...}, 'lane': ...}
        # target: same structure
        
        # 1. Detection Loss
        det_loss = 0.0
        for scale in ['p3', 'p4', 'p5']:
            det_loss += self.det_loss_fn(pred['det'][scale], target['det'][scale])
        
        # 2. Drivable Area Loss
        drv_loss = 0.0
        for scale in ['p3', 'p4']:
            # Target for segmentation is [B, H, W] (class indices)
            # pred is [B, C, H, W]
            drv_loss += self.drv_loss_fn(pred['drv'][scale], target['drv'][scale].long())
            
        # 3. Lane Loss
        lane_loss = self.lane_loss_fn(pred['lane'], target['lane'])

        total_loss = (self.lambda_det * det_loss + 
                      self.lambda_drv * drv_loss + 
                      self.lambda_lane * lane_loss)

        return total_loss, det_loss, drv_loss, lane_loss
