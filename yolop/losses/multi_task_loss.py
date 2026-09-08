"""
YOLOP Multi-Task Loss.

    total_loss = lambda_det * det_loss
               + lambda_drv * drv_loss
               + lambda_lane * lane_loss

det_loss:  BCEWithLogitsLoss on objectness heatmap (per scale, averaged)
drv_loss:  CrossEntropyLoss  on 3-class drivable area segmentation
lane_loss: BCEWithLogitsLoss on binary lane mask
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class YOLOPLoss(nn.Module):
    def __init__(self, lambda_det=1.0, lambda_drv=1.0, lambda_lane=1.0,
                 num_classes_det=10):
        super().__init__()
        self.lambda_det  = lambda_det
        self.lambda_drv  = lambda_drv
        self.lambda_lane = lambda_lane
        self.num_cls     = num_classes_det

        # pos_weight for det: heatmaps are very sparse
        self.det_loss_fn  = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([10.0]))
        self.drv_loss_fn  = nn.CrossEntropyLoss(ignore_index=255)
        self.lane_loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([10.0]))

    def forward(self, pred, target):
        """
        pred:   {'det': {'p3':..,'p4':..,'p5':..}, 'drv': .., 'lane': ..}
        target: {'det': {'p3':..,'p4':..,'p5':..}, 'drv': .., 'lane': ..}

        det pred shape:   [B, 1+num_cls, H_s, W_s]
        det target shape: [B,          1, H_s, W_s]   (objectness only for now)
        drv pred shape:   [B,          3, H,   W  ]
        drv target shape: [B,             H,   W  ]   (int64 class indices)
        lane pred shape:  [B,          1, H,   W  ]
        lane target shape:[B,          1, H,   W  ]   (float32 0/1)
        """
        # ── Detection loss (objectness channel only for simplicity) ──
        det_loss = torch.tensor(0.0, device=pred['drv'].device)
        for scale in ['p3', 'p4', 'p5']:
            pred_obj = pred['det'][scale][:, :1]          # [B,1,H,W]
            tgt_obj  = target['det'][scale]               # [B,1,H,W]

            # Move pos_weight to correct device
            pw = self.det_loss_fn.pos_weight
            if pw.device != pred_obj.device:
                self.det_loss_fn.pos_weight = pw.to(pred_obj.device)

            det_loss = det_loss + self.det_loss_fn(pred_obj, tgt_obj)
        det_loss = det_loss / 3.0

        # ── Drivable area loss ────────────────────────────────────────
        drv_pred = pred['drv']                             # [B,3,H,W]
        drv_tgt  = target['drv'].long()                   # [B,H,W]
        drv_loss = self.drv_loss_fn(drv_pred, drv_tgt)

        # ── Lane loss ─────────────────────────────────────────────────
        lane_pred = pred['lane']                           # [B,1,H,W]
        lane_tgt  = target['lane']                         # [B,1,H,W]

        pw = self.lane_loss_fn.pos_weight
        if pw.device != lane_pred.device:
            self.lane_loss_fn.pos_weight = pw.to(lane_pred.device)

        lane_loss = self.lane_loss_fn(lane_pred, lane_tgt)

        total_loss = (self.lambda_det  * det_loss +
                      self.lambda_drv  * drv_loss +
                      self.lambda_lane * lane_loss)

        return total_loss, det_loss.detach(), drv_loss.detach(), lane_loss.detach()
