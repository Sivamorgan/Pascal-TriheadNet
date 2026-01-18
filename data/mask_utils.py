import torch
import torch.nn.functional as F

def crop_and_resize_instance_masks(binary_masks,boxes,mask_size=28) :
    """Crop each binary mask to its corresponding box and resize.
    Args:
        binary_masks: (N, H, W) binary masks, one per instance
        boxes: (N, 4) bounding boxes in [x1, y1, x2, y2] format
        mask_size: output mask resolution (default 28)
    
    Returns:
        per_box_masks: (N, 1, mask_size, mask_size) cropped and resized masks
    """
    N = boxes.shape[0]
    device = boxes.device
    
    if N == 0:
        return torch.zeros((0, 1, mask_size, mask_size), device=device)
    if binary_masks.device != device:
        binary_masks = binary_masks.to(device)
    H, W = binary_masks.shape[-2:]
    per_box_masks = []
    
    for i in range(N):
        x1, y1, x2, y2 = boxes[i].int()
        x1 = max(0, x1.item())
        y1 = max(0, y1.item())
        x2 = min(W, x2.item())
        y2 = min(H, y2.item())
        if x2 <= x1 or y2 <= y1:
            per_box_masks.append(torch.zeros((1, mask_size, mask_size), device=device))
            continue
        crop = binary_masks[i, y1:y2, x1:x2].float()
        crop = crop.unsqueeze(0).unsqueeze(0)  # (1, 1, h, w)
        resized = F.interpolate(crop, size=(mask_size, mask_size), mode='nearest')
        per_box_masks.append(resized.squeeze(0))  # (1, mask_size, mask_size)
    
    return torch.stack(per_box_masks, dim=0)  # (N, 1, mask_size, mask_size)
