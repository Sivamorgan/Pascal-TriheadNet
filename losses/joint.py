import torch.nn as nn
from .detection import FCOSLoss
from .segmentation import SemanticSegmentationLoss, InstanceSegmentationLoss

class JointTrainingLoss(nn.Module):
    """Combined loss for joint detection and segmentation training.
    
    Weights and combines losses from all three heads.
    
    Args: det_weight, sem_weight, inst_weight
    """
    
    def __init__(self,
                 det_weight: float = 1.0,
                 sem_weight: float = 1.0,
                 inst_weight: float = 1.0,
                 boundary_weight: float = 2.0,
                 ignore_index: int = 255):
        """Initialize joint training loss.
        
        Args: 
            det_weight: Weight for detection loss
            sem_weight: Weight for semantic loss
            inst_weight: Weight for instance loss
            boundary_weight: Extra weight for boundary pixels in semantic loss
            ignore_index: Label to ignore in segmentation
        """
        super().__init__()
        self.det_loss = FCOSLoss(bbox_weight=4)
        self.sem_loss = SemanticSegmentationLoss(
            ignore_index=ignore_index,
            boundary_weight=boundary_weight
        )
        self.inst_loss = InstanceSegmentationLoss()
        
        self.det_weight = det_weight
        self.sem_weight = sem_weight
        self.inst_weight = inst_weight
    
    def forward(self,det_pred=None,det_targets=None,sem_pred=None,sem_target=None,inst_pred=None,inst_target=None):
        """Compute joint training loss.
        
        Args: All prediction and target pairs (optional based on available annotations)
        Returns: Dict with all losses and total
        """
        losses = {}
        total = 0.0
        
        # Detection loss
        if det_pred is not None and det_targets is not None:
            det_losses = self.det_loss(det_pred, det_targets)
            losses['det_cls'] = det_losses['cls_loss']
            losses['det_bbox'] = det_losses['bbox_loss']
            losses['det_centerness'] = det_losses['centerness_loss']
            losses['det_total'] = det_losses['total']
            total = total + self.det_weight * det_losses['total']
        
        # Semantic segmentation loss
        if sem_pred is not None and sem_target is not None:
            sem_losses = self.sem_loss(sem_pred, sem_target)
            losses['sem_ce'] = sem_losses['ce_loss']
            losses['sem_dice'] = sem_losses['dice_loss']
            losses['sem_total'] = sem_losses['total']
            total = total + self.sem_weight * sem_losses['total']
        
        # Instance segmentation loss
        if inst_pred is not None and inst_target is not None:
            inst_losses = self.inst_loss(inst_pred, inst_target)
            losses['inst_bce'] = inst_losses['bce']
            losses['inst_dice'] = inst_losses['dice']
            losses['inst_total'] = inst_losses['total']
            total = total + self.inst_weight * inst_losses['total']
        
        losses['total'] = total
        return losses
