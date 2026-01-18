import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(nn.Module):
    """Dice Loss for segmentation.
    
    Supports both multiclass (softmax) and binary (sigmoid) modes.
    Handles ignore_index for semantic segmentation.
    
    Args:
        mode: 'multiclass' or 'binary'
        ignore_index: Label to ignore (for multiclass)
        smooth: Smoothing factor to avoid division by zero
    """
    
    def __init__(self, mode: str = 'multiclass', ignore_index= 255, smooth= 1.0):
        super().__init__()
        assert mode in ['multiclass', 'binary']
        self.mode = mode
        self.ignore_index = ignore_index
        self.smooth = smooth
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute Dice loss.
        
        Args: 
            pred: (N, C, H, W) logits for multiclass, or (N, 1, H, W) logits for binary
            target: (N, H, W) labels for multiclass, or (N, 1, H, W) mask for binary
        Returns: 
            Scalar loss
        """
        if self.mode == 'multiclass':
            num_classes = pred.shape[1]
            pred_soft = F.softmax(pred, dim=1)
            
            # Create valid mask
            valid_mask = target != self.ignore_index
            if not valid_mask.any():
                return pred.sum() * 0.0
            
            # Handle ignore_index
            target_valid = target.clone()
            target_valid[~valid_mask] = 0
            
            # One-hot encode: (N, C, H, W)
            target_onehot = F.one_hot(target_valid.long(), num_classes).permute(0, 3, 1, 2).float()
            
            # Mask out invalid regions
            valid_mask_expanded = valid_mask.unsqueeze(1).expand_as(pred_soft)
            pred_soft = pred_soft * valid_mask_expanded
            target_onehot = target_onehot * valid_mask_expanded
            dims = (2, 3) 
            intersection = (pred_soft * target_onehot).sum(dim=dims)
            #union = pred_soft.sum(dim=dims) + target_onehot.sum(dim=dims)
            cardinality = (pred_soft + target_onehot).sum(dim=dims)
            
            # Dice per sample per class
            dice_score = (2 * intersection + self.smooth) / (cardinality + self.smooth)
            #dice_score= (2*intersection+self.smooth)/(union+self.smooth)
            valid_reduction_mask = cardinality > 0
            
            if valid_reduction_mask.any():
                # Only average over valid sample-class pairs
                final_loss = 1 - dice_score[valid_reduction_mask].mean()
            else:
                final_loss = pred.sum() * 0.0 # Gradient safety
            
            return final_loss
            
        else: # binary
            pred_sigmoid = torch.sigmoid(pred)
            target_f = target.float()
            dims = (2, 3)
            intersection = (pred_sigmoid * target_f).sum(dim=dims)
            cardinality = (pred_sigmoid + target_f).sum(dim=dims)
            
            dice_score = (2 * intersection + self.smooth) / (cardinality + self.smooth)
            return 1 - dice_score.mean()


def detect_semantic_boundaries(target, ignore_index = 255):
    """Detect boundary pixels where adjacent pixels have different classes.
    
    Args:
        target: (N, H, W) semantic labels
        ignore_index: Label to ignore
    
    Returns:
        boundary_mask: (N, H, W) boolean mask, True at boundary pixels
    """
    N, H, W = target.shape
    device = target.device
    
    # Create padded version for neighbor comparison
    target_padded = F.pad(target.float(), (1, 1, 1, 1), mode='replicate')
    
    # Get shifts for 4-connected neighbors (up, down, left, right)
    up = target_padded[:, :-2, 1:-1]
    down = target_padded[:, 2:, 1:-1]
    left = target_padded[:, 1:-1, :-2]
    right = target_padded[:, 1:-1, 2:]
    
    # Boundary = any neighbor has different label
    center = target.float()
    boundary = (center != up) | (center != down) | (center != left) | (center != right)
    
    # Ignore boundary pixels at ignore_index
    valid_mask = target != ignore_index
    boundary = boundary & valid_mask
    
    return boundary


class SemanticSegmentationLoss(nn.Module):
    """Combined loss for semantic segmentation: CE + Dice + Boundary Weighting.
    
    Args: 
        ce_weight: Weight for cross-entropy loss
        dice_weight: Weight for dice loss
        boundary_weight: Extra weight multiplier for boundary pixels (1.0 = no extra weight)
        ignore_index: Label to ignore
    """
    
    def __init__(self, 
                 ce_weight= 1.0,
                 dice_weight= 1.0,
                 boundary_weight= 2.0,
                 ignore_index= 255):
        super().__init__()
        self.ce_loss_fn = nn.CrossEntropyLoss(ignore_index=ignore_index, reduction='none')
        self.dice_loss = DiceLoss(mode='multiclass', ignore_index=ignore_index)
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.boundary_weight = boundary_weight
        self.ignore_index = ignore_index
    
    def forward(self, pred, target):
        # Detect boundary pixels
        boundary_mask = detect_semantic_boundaries(target, self.ignore_index)
        
        # CE Loss with boundary weighting (normalized to keep loss magnitude consistent)
        ce_per_pixel = self.ce_loss_fn(pred, target.long())  # (N, H, W)
        
        # Create weight map: boundary pixels get extra weight
        weight_map = torch.ones_like(ce_per_pixel)
        weight_map[boundary_mask] = self.boundary_weight
        
        # Normalize weights so mean weight = 1.0 (preserves loss magnitude)
        valid_mask = target != self.ignore_index
        if valid_mask.any():
            mean_weight = weight_map[valid_mask].mean()
            weight_map = weight_map / mean_weight  # Normalize
            
            ce_weighted = ce_per_pixel * weight_map
            ce = ce_weighted[valid_mask].mean()
        else:
            ce = pred.sum() * 0.0
        
        dice = self.dice_loss(pred, target)
        total = self.ce_weight * ce + self.dice_weight * dice
        if torch.isnan(total):
            total = pred.sum() * 0.0
            ce = pred.sum() * 0.0
            dice = pred.sum() * 0.0
        
        return {
            'ce_loss': ce,
            'dice_loss': dice,
            'total': total,
        }


class InstanceSegmentationLoss(nn.Module):
    """Loss for instance segmentation masks: BCE + Dice with per-mask weighting."""
    
    def __init__(self):
        super().__init__()
        self.dice_loss = DiceLoss(mode='binary')
    
    def forward(self,pred_masks,target_masks):
        """
        Args:
            pred_masks: (N, 1, H, W) predicted mask logits
            target_masks: (N, 1, H, W) target binary masks
        """
        if pred_masks.numel() == 0:
            return {'bce': pred_masks.sum() * 0.0, 'dice': pred_masks.sum() * 0.0, 'total': pred_masks.sum() * 0.0}
        bce_per_pixel = F.binary_cross_entropy_with_logits(
            pred_masks, target_masks.float(), reduction='none'
        )  # (N, 1, H, W)
        
        # Get mask areas for weighting
        mask_areas = target_masks.sum(dim=(1, 2, 3)) + 1.0  # (N,) + 1 to avoid div by 0
        mask_weights = 1.0 / mask_areas  # Smaller masks get higher weight
        mask_weights = mask_weights / mask_weights.mean()  # Normalize to mean=1
        
        # Weighted mean: average per-mask loss, then weight by mask area
        bce_per_mask = bce_per_pixel.mean(dim=(1, 2, 3))  # (N,)
        bce = (bce_per_mask * mask_weights).mean()
        
        # Dice Loss (already handles per-mask naturally)
        dice = self.dice_loss(pred_masks, target_masks)
        
        total = bce + dice
        
        return {
            'bce': bce,
            'dice': dice,
            'total': total,
        }
