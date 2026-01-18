import torch
import torch.nn as nn
import torch.nn.functional as F
from losses.loss_utils import compute_locations

class FocalLoss(nn.Module):
    """Focal Loss for dense object detection (FCOS).
    
    Args: alpha, gamma, reduction
    """
    
    def __init__(self, alpha = 0.25, gamma = 2.0, reduction:str = 'sum'):
        """Args: alpha (foreground weight), gamma (focusing parameter), reduction
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, pred, target):
        """
        Args: pred (logits), target (class indices or one-hot)
        Returns: Focal loss
        """
        pred_sigmoid = torch.sigmoid(pred)
        if target.dim() == 1 or target.shape[-1] != pred.shape[-1]:
            num_classes = pred.shape[-1]
            target_onehot = F.one_hot(target.long(), num_classes).float()
        else:
            target_onehot = target
        # Compute focal weights
        pt = pred_sigmoid * target_onehot + (1 - pred_sigmoid) * (1 - target_onehot)
        focal_weight = (1 - pt) ** self.gamma
        # Compute alpha weights
        alpha_t = self.alpha * target_onehot + (1 - self.alpha) * (1 - target_onehot)
        # Compute BCE loss
        bce = F.binary_cross_entropy_with_logits(pred, target_onehot, reduction='none')
        # Apply focal and alpha weights
        loss = alpha_t * focal_weight * bce
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


class GIoULoss(nn.Module):
    """Generalized IoU Loss for bounding box regression.
    Args: reduction
    """
    
    def __init__(self, reduction: str = 'mean'):
        """Initialize GIoU Loss.
        
        Args: reduction
        """
        super().__init__()
        self.reduction = reduction
    
    def forward(self, pred, target):
        """Compute GIoU loss.
        
        Args: pred (N, 4) predicted boxes [x1, y1, x2, y2]
              target (N, 4) target boxes [x1, y1, x2, y2]
        Returns: GIoU loss
        """
        # Ensure valid boxes (x2 > x1, y2 > y1)
        pred_x1, pred_y1, pred_x2, pred_y2 = pred.unbind(-1)
        target_x1, target_y1, target_x2, target_y2 = target.unbind(-1)
        
        # Compute intersection
        inter_x1 = torch.max(pred_x1, target_x1)
        inter_y1 = torch.max(pred_y1, target_y1)
        inter_x2 = torch.min(pred_x2, target_x2)
        inter_y2 = torch.min(pred_y2, target_y2)
        
        inter_area = (inter_x2 - inter_x1).clamp(min=0) * (inter_y2 - inter_y1).clamp(min=0)
        
        # Compute union
        pred_area = (pred_x2 - pred_x1) * (pred_y2 - pred_y1)
        target_area = (target_x2 - target_x1) * (target_y2 - target_y1)
        union_area = pred_area + target_area - inter_area
        
        # Compute IoU
        iou = inter_area / (union_area + 1e-7)
        
        # Compute enclosing box
        enclose_x1 = torch.min(pred_x1, target_x1)
        enclose_y1 = torch.min(pred_y1, target_y1)
        enclose_x2 = torch.max(pred_x2, target_x2)
        enclose_y2 = torch.max(pred_y2, target_y2)
        enclose_area = (enclose_x2 - enclose_x1) * (enclose_y2 - enclose_y1)
        
        # Compute GIoU
        giou = iou - (enclose_area - union_area) / (enclose_area + 1e-7)
        
        loss = 1 - giou
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


class FCOSTargetAssigner(nn.Module):
    """Assigns targets for FCOS training."""
    
    def __init__(self, 
                 fpn_strides=[4, 8, 16, 32], 
                 center_sampling_radius=2):
        super().__init__()
        self.fpn_strides = fpn_strides
        self.center_sampling_radius = center_sampling_radius

    def forward(self, locations, gt_boxes, gt_labels):
        """
        Args:
            locations: list of (N_loc_i, 2)
            gt_boxes: list of (M, 4)
            gt_labels: list of (M,)
        Returns:
            labels, reg_targets, centerness_targets
        """
        # Prepare strides for all locations (constant across batch)
        strides_list = []
        for locs, stride in zip(locations, self.fpn_strides):
            strides_list.append(torch.full((len(locs),), stride, device=locations[0].device))
        all_strides = torch.cat(strides_list, dim=0) # (L_total,)

        # Iterate over batch
        labels_list = []
        reg_targets_list = []
        centerness_targets_list = []
        
        for _, (boxes, labels) in enumerate(zip(gt_boxes, gt_labels)):
            # Concatenate locations from all levels
            all_locs = torch.cat(locations, dim=0)
            
            # Ensure device consistency
            boxes = boxes.to(all_locs.device)
            labels = labels.to(all_locs.device)
            
            if len(boxes) == 0:
                # No objects
                num_locs = all_locs.shape[0]
                labels_list.append(torch.zeros(num_locs, dtype=torch.long, device=all_locs.device)) # Background = 0
                reg_targets_list.append(torch.zeros(num_locs, 4, dtype=torch.float32, device=all_locs.device))
                centerness_targets_list.append(torch.zeros(num_locs, dtype=torch.float32, device=all_locs.device))
                continue
                
            # Compute distances (L, M): l=x-x1, t=y-y1, r=x2-x, b=y2-y
            l = all_locs[:, 0][:, None] - boxes[:, 0][None, :]
            t = all_locs[:, 1][:, None] - boxes[:, 1][None, :]
            r = boxes[:, 2][None, :] - all_locs[:, 0][:, None]
            b = boxes[:, 3][None, :] - all_locs[:, 1][:, None]
            
            reg_targets_per_im = torch.stack([l, t, r, b], dim=2) # (L, M, 4)
            
            # Determine which boxes are valid for each location (inside box)
            is_in_box = reg_targets_per_im.min(dim=2)[0] > 0 # (L, M)
            
            # Center sampling: locations must be near the center of the box
            if self.center_sampling_radius > 0:
                gt_centers_x = (boxes[:, 0] + boxes[:, 2]) / 2
                gt_centers_y = (boxes[:, 1] + boxes[:, 3]) / 2
                
                # Expand strides to (L, M)
                strides_expanded = all_strides[:, None].expand(-1, len(boxes))
                radius = self.center_sampling_radius * strides_expanded
                
                # Check if loc is within center box (Rectangular region - FCOS standard)
                dist_x = (all_locs[:, 0][:, None] - gt_centers_x[None, :]).abs()
                dist_y = (all_locs[:, 1][:, None] - gt_centers_y[None, :]).abs()
                
                is_in_center = (dist_x < radius) & (dist_y < radius)
                is_in_box = is_in_box & is_in_center

            # FPN Level assignment: Check regression range for each stride.
            num_locs_per_level = [len(loc) for loc in locations]
            
            # Max regression target for each location
            max_reg_targets_per_im = reg_targets_per_im.max(dim=2)[0] # (L, M)
            
            # Level limits: Adjusted for 224 input 
            level_limits = [[0, 32], [32, 64], [64, 128], [128, 999999]]
            
            is_in_level = torch.zeros_like(is_in_box, dtype=torch.bool)
            
            start_idx = 0
            for level_idx, num_locs in enumerate(num_locs_per_level):
                end_idx = start_idx + num_locs
                limit_range = level_limits[level_idx] if level_idx < len(level_limits) else [0, 999999]
                
                # Check if max reg target is within range for ANY box
                lvl_max_reg = max_reg_targets_per_im[start_idx:end_idx]
                mask = (lvl_max_reg > limit_range[0]) & (lvl_max_reg <= limit_range[1])
                is_in_level[start_idx:end_idx] = mask
                
                start_idx = end_idx
            
            # Valid locations: inside box (and center) AND correct level
            valid_locs = is_in_box & is_in_level # (L, M)
            
            # Match locations to boxes with minimal area
            box_areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]) # (M,)
            locations_to_gt_area = box_areas[None, :].repeat(all_locs.shape[0], 1) # (L, M)
            locations_to_gt_area[~valid_locs] = float('inf')
            
            min_area, min_area_inds = locations_to_gt_area.min(dim=1) # (L,)
            
            # Assignments
            labels_per_im = torch.zeros(all_locs.shape[0], dtype=torch.long, device=all_locs.device)
            reg_targets_final = torch.zeros(all_locs.shape[0], 4, dtype=torch.float32, device=all_locs.device)
            centerness_final = torch.zeros(all_locs.shape[0], dtype=torch.float32, device=all_locs.device)
            
            # Positive locations
            pos_inds = min_area < float('inf')
            
            if pos_inds.any():
                # Assign labels (VOC 1-20)
                labels_per_im[pos_inds] = labels[min_area_inds[pos_inds]]
                
                # Assign regression targets (indexing handled properly with explicit arange)
                pos_loc_indices = torch.arange(all_locs.shape[0], device=all_locs.device)[pos_inds]
                pos_reg_targets = reg_targets_per_im[pos_loc_indices, min_area_inds[pos_inds]]
                reg_targets_final[pos_inds] = pos_reg_targets
                
                # Compute centerness
                l, t, r, b = pos_reg_targets.unbind(1)
                
                # Clamp l, t, r, b to prevent negative values (edge cases)
                l = l.clamp(min=1e-6)
                t = t.clamp(min=1e-6)
                r = r.clamp(min=1e-6)
                b = b.clamp(min=1e-6)
                
                # Compute centerness: sqrt((min(l,r)/max(l,r)) * (min(t,b)/max(t,b)))
                centerness = torch.sqrt(
                    (torch.min(l, r) / torch.max(l, r)) * 
                    (torch.min(t, b) / torch.max(t, b))
                )
                centerness_final[pos_inds] = centerness
            
            labels_list.append(labels_per_im)
            reg_targets_list.append(reg_targets_final)
            centerness_targets_list.append(centerness_final)
        return (
            torch.stack(labels_list),
            torch.stack(reg_targets_list),
            torch.stack(centerness_targets_list)
        )


class FCOSLoss(nn.Module):
    """Combined FCOS loss: Focal + GIoU + BCE Centerness.
    
    Args: cls_weight, bbox_weight, centerness_weight
    """
    
    def __init__(self, cls_weight = 1.0,bbox_weight= 1.0,
                 centerness_weight = 1.0,strides = [4, 8, 16, 32]):
        """Initialize FCOS Loss.
        Args: cls_weight, bbox_weight, centerness_weight
        """
        super().__init__()
        self.focal_loss = FocalLoss()
        # Use reduction='none' to allow element-wise weighting by centerness
        self.giou_loss = GIoULoss(reduction='none')
        self.cls_weight = cls_weight
        self.bbox_weight = bbox_weight
        self.centerness_weight = centerness_weight
        self.strides = strides
        self.assigner = FCOSTargetAssigner(fpn_strides=strides)
    
    def forward(self,pred,targets):
        """Compute FCOS loss.
        
        Args: 
            pred - Dict with cls, bbox, centerness predictions per level
            targets - Dict with gt_boxes, gt_labels
        Returns: Dict with cls_loss, bbox_loss, centerness_loss, total
        """
        cls_preds = pred['cls']
        bbox_preds = pred['bbox']
        centerness_preds = pred['centerness']
        gt_boxes = targets['gt_boxes']
        gt_labels = targets['gt_labels']
        
        # Compute locations for each level
        locations = []
        for i, stride in enumerate(self.strides):
            feat = cls_preds[i] # (N, C, H, W)
            locs = compute_locations(feat, stride)
            locations.append(locs)
            
        #  Assign targets
        target_labels, target_bboxes, target_centerness = self.assigner(locations, gt_boxes, gt_labels)
        
        # Flatten predictions for all levels
        flat_cls = []
        flat_bbox = []
        flat_center = []
        flat_locs = []
        
        for i in range(len(cls_preds)):
            N, C, H, W = cls_preds[i].shape
            flat_cls.append(cls_preds[i].permute(0, 2, 3, 1).reshape(N, -1, C))
            flat_bbox.append(bbox_preds[i].permute(0, 2, 3, 1).reshape(N, -1, 4))
            flat_center.append(centerness_preds[i].permute(0, 2, 3, 1).reshape(N, -1))
            
            # Flatten locations for this level (repeat for batch)
            locs_lvl = locations[i].unsqueeze(0).repeat(N, 1, 1)
            flat_locs.append(locs_lvl)
        
        flat_cls = torch.cat(flat_cls, dim=1) # (N, L_total, num_classes)
        flat_bbox = torch.cat(flat_bbox, dim=1) # (N, L_total, 4)
        flat_center = torch.cat(flat_center, dim=1).squeeze(-1) # (N, L_total)
        flat_locs = torch.cat(flat_locs, dim=1) # (N, L_total, 2)
        flat_cls = flat_cls.reshape(-1, flat_cls.shape[-1])
        flat_bbox = flat_bbox.reshape(-1, 4)
        flat_center = flat_center.reshape(-1)
        flat_locs = flat_locs.reshape(-1, 2)
        
        target_labels = target_labels.reshape(-1)
        target_bboxes = target_bboxes.reshape(-1, 4)
        target_centerness = target_centerness.reshape(-1)
        
        # Pos mask
        pos_inds = target_labels > 0
        num_pos = pos_inds.sum().item()
        
        # Create one-hot targets (N_samples, Num_classes)
        num_classes = flat_cls.shape[1]
        cls_targets_onehot = torch.zeros_like(flat_cls)
        if num_pos > 0:
            pos_labels = target_labels[pos_inds] - 1
            cls_targets_onehot[pos_inds, pos_labels] = 1.0
        cls_loss = self.focal_loss(flat_cls, cls_targets_onehot) / max(num_pos, 1.0)
        
        # Regression and Centerness Loss (Only positive samples)
        if num_pos > 0:
            pos_bbox_preds = flat_bbox[pos_inds]
            pos_bbox_targets = target_bboxes[pos_inds]
            pos_center_preds = flat_center[pos_inds]
            pos_center_targets = target_centerness[pos_inds]
            pos_locs = flat_locs[pos_inds]
            
            # GIoU Needs xyxy format. Convert Predictions.
            pred_l, pred_t, pred_r, pred_b = pos_bbox_preds.unbind(1)
            pred_boxes = torch.stack([
                pos_locs[:, 0] - pred_l,
                pos_locs[:, 1] - pred_t,
                pos_locs[:, 0] + pred_r,
                pos_locs[:, 1] + pred_b
            ], dim=1)
            
            # Convert Targets to xyxy used for GIoU
            target_l, target_t, target_r, target_b = pos_bbox_targets.unbind(1)
            target_boxes = torch.stack([
                pos_locs[:, 0] - target_l,
                pos_locs[:, 1] - target_t,
                pos_locs[:, 0] + target_r,
                pos_locs[:, 1] + target_b
            ], dim=1)
            giou = self.giou_loss(pred_boxes, target_boxes)
            weighted_giou = giou * pos_center_targets
            # Normalize by sum of weights (not num_pos) to preserve loss magnitude
            bbox_loss = weighted_giou.sum() / max(pos_center_targets.sum().item(), 1.0)
            centerness_loss = F.binary_cross_entropy_with_logits(pos_center_preds, pos_center_targets, reduction='mean')
        else:
            # Ensure gradient flow even for empty batches
            bbox_loss = flat_bbox.sum() * 0.0
            centerness_loss = flat_center.sum() * 0.0
            
        total_loss = (
            self.cls_weight * cls_loss +
            self.bbox_weight * bbox_loss +
            self.centerness_weight * centerness_loss
        )
        
        return {
            'cls_loss': cls_loss,
            'bbox_loss': bbox_loss,
            'centerness_loss': centerness_loss,
            'total': total_loss,
        }
