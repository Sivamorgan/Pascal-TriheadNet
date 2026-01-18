import torch
import torch.nn.functional as F
from losses.loss_utils import compute_locations
from torchvision.ops import batched_nms

def postprocess_detection(
    preds,
    strides = [4, 8, 16, 32],
    score_thresh: float = 0.2,
    nms_thresh: float = 0.5,
    max_detections: int = 100
):
    """Convert raw FCOS predictions to final bounding boxes.
    
    Args:
        preds: Dict with 'cls', 'bbox', 'centerness' per level
        strides: FPN strides
        score_thresh: Threshold to filter low confidence
        nms_thresh: IoU threshold for NMS
        max_detections: Max boxes per image
        
    Returns:
        results: List of dicts [{'boxes': (N,4), 'scores': (N,), 'labels': (N,)}]
    """
    cls_preds = preds['cls']
    bbox_preds = preds['bbox']
    centerness_preds = preds['centerness']
    
    batch_size = cls_preds[0].shape[0]
    results = []
    for batch_idx in range(batch_size):
        all_boxes = []
        all_scores = []
        all_labels = []
        for i, stride in enumerate(strides):
            cls_p = cls_preds[i][batch_idx]       # (C, H, W)
            bbox_p = bbox_preds[i][batch_idx]     # (4, H, W)
            center_p = centerness_preds[i][batch_idx] # (1, H, W)
            C,H, W = cls_p.shape
            locs = compute_locations(cls_p, stride) # (H*W, 2)
            cls_p = cls_p.permute(1, 2, 0).reshape(-1, C).sigmoid() # (H*W, C)
            bbox_p = bbox_p.permute(1, 2, 0).reshape(-1, 4) # (H*W, 4)
            center_p = center_p.permute(1, 2, 0).reshape(-1).sigmoid() # (H*W,)
            scores = cls_p * center_p.unsqueeze(1) # (H*W, C)
            max_scores,max_classes = scores.max(dim=1)
            keep= max_scores>score_thresh
            if not keep.any():
                continue
            valid_scores = max_scores[keep]
            valid_classes = max_classes[keep]
            valid_bbox_deltas = bbox_p[keep]
            valid_locs = locs[keep]
            l, t, r, b = valid_bbox_deltas.unbind(1)
            x_center, y_center = valid_locs.unbind(1)
            
            boxes = torch.stack([
                x_center-l,
                y_center-t,
                x_center + r,
                y_center + b
            ],dim=1)
            
            all_boxes.append(boxes)
            all_scores.append(valid_scores)
            all_labels.append(valid_classes + 1) # Shift 0-19 -> 1-20 to match VOC labels
            
        if len(all_boxes) == 0:
            results.append({
                'boxes': torch.zeros((0, 4), device=cls_preds[0].device),
                'scores': torch.zeros((0,), device=cls_preds[0].device),
                'labels': torch.zeros((0,), dtype=torch.long, device=cls_preds[0].device)
            })
            continue
            
        all_boxes = torch.cat(all_boxes, dim=0)
        all_scores = torch.cat(all_scores, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        keep = batched_nms(all_boxes, all_scores, all_labels, nms_thresh)
        keep = keep[:max_detections]
        
        results.append({
            'boxes': all_boxes[keep],
            'scores': all_scores[keep],
            'labels': all_labels[keep]
        })
        
    return results


def paste_masks_in_image(masks,boxes,image_shape,threshold: float = 0.5):
    """Paste region-based masks onto full image canvas.
    
    Args:
        masks: (N, 1, 28, 28) - per-box masks from instance head
        boxes: (N, 4) - [x1, y1, x2, y2] in absolute coordinates
        image_shape: (H, W) - target image size
        threshold: Binary threshold for masks
    
    Returns:
        masks_full: (N, H, W) - binary masks at image resolution
    """
    H, W = image_shape
    N = len(boxes)
    device = masks.device
    if N == 0:
        return torch.zeros((0, H, W), device=device)
    masks_full = torch.zeros((N, H, W), device=device, dtype=torch.uint8)
    
    for i in range(N):
        # Get box coordinates
        x1, y1, x2, y2 = boxes[i].int().clamp(min=0)
        x1, x2 = x1.item(), x2.item()
        y1, y2 = y1.item(), y2.item()
        x1 = max(0, min(x1, W))
        x2 = max(0, min(x2, W))
        y1 = max(0, min(y1, H))
        y2 = max(0, min(y2, H))
        
        box_h = y2 - y1
        box_w = x2 - x1
        
        if box_h <= 0 or box_w <= 0:
            continue  
        # Resize mask from 28×28 to box size
        mask_resized = F.interpolate(
            masks[i:i+1],  # (1, 1, 28, 28)
            size=(box_h, box_w),
            mode='bilinear',
            align_corners=False
        )[0, 0]  # (box_h, box_w)
        mask_binary = (mask_resized > threshold).to(dtype=torch.uint8)
        masks_full[i, y1:y2, x1:x2] = mask_binary
    
    return masks_full


def postprocess_instance_segmentation(
    inst_preds,
    det_results,
    image_size,
    mask_threshold: float = 0.5
):
    """Convert region masks to full-image instance segmentation.
    
    Args:
        inst_preds: Instance masks (N_total_boxes, 1, 28, 28) from instance head
        det_results: Postprocessed detections from postprocess_detection()
                     List[{'boxes': (N,4), 'scores': (N,), 'labels': (N,)}]
        image_size: (H, W) - image dimensions
        mask_threshold: Threshold for binary masks
    
    Returns:
        List of dicts per image: {
            'boxes': (N, 4),
            'labels': (N,),
            'scores': (N,),
            'masks': (N, H, W)  # Full-image binary masks
        }
    """
    results = []
    mask_offset = 0 
    
    for det_result in det_results:
        boxes = det_result['boxes']
        scores = det_result['scores']
        labels = det_result['labels']
        N = len(boxes)
        
        if N == 0:
            results.append({
                'boxes': boxes,
                'labels': labels,
                'scores': scores,
                'masks': torch.zeros((0, *image_size), device=boxes.device)
            })
            continue
        inst_masks = inst_preds[mask_offset:mask_offset + N]  # (N, 1, 28, 28)
        masks_full = paste_masks_in_image(
            inst_masks,
            boxes,
            image_size,
            mask_threshold
        )
        
        results.append({
            'boxes': boxes,
            'labels': labels,
            'scores': scores,
            'masks': masks_full  # (N, H, W)
        })
        
        mask_offset += N
    
    return results
