import torch
import numpy as np
import cv2

# VOC Classes
VOC_CLASSES = (
    "background", "aeroplane", "bicycle", "bird", "boat", "bottle",
    "bus", "car", "cat", "chair", "cow", "diningtable", "dog",
    "horse", "motorbike", "person", "pottedplant", "sheep", "sofa",
    "train", "tvmonitor"
)

# Colors (random fixed)
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(21, 3), dtype=np.uint8)

def draw_boxes(image_tensor, boxes, labels, scores=None, color_override=None):
    """Draw boxes on image.
    
    Args:
        image_tensor: (C, H, W) float tensor
        boxes: (N, 4)
        labels: (N,)
        scores: (N,) optional
        color_override: Tuple (B, G, R) optional
    """
    # Denormalize
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(image_tensor.device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(image_tensor.device)
    
    img = image_tensor * std + mean
    img = img.permute(1, 2, 0).cpu().numpy()
    img = (img * 255).astype(np.uint8).copy()
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) # OpenCV uses BGR
    
    if boxes is None or len(boxes) == 0:
        return img
        
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = box.int().tolist()
        label = labels[i].item()
        
        # Color
        color = COLORS[label].tolist() if color_override is None else color_override
        
        # Draw box
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        
        # Label text
        class_name = VOC_CLASSES[label] if label < len(VOC_CLASSES) else str(label)
        text = f"{class_name}"
        if scores is not None:
            text += f" {scores[i]:.2f}"
            
        # Draw text background
        (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        
        # Check if text fits above box
        if y1 - 20 > 0:
            text_bg_top = y1 - 20
            text_bg_bottom = y1
            text_pos = (x1, y1 - 5)
        else:
            # Draw inside box (at top)
            text_bg_top = y1
            text_bg_bottom = y1 + 20
            text_pos = (x1, y1 + 15)
            
        cv2.rectangle(img, (x1, text_bg_top), (x1 + w, text_bg_bottom), color, -1)
        cv2.putText(img, text, text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
    return img

def visualize_segmentation(image_tensor, mask, alpha=0.5):
    """Overlay segmentation mask."""
    # Denormalize
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(image_tensor.device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(image_tensor.device)
    
    img = image_tensor * std + mean
    img = img.permute(1, 2, 0).cpu().numpy()
    img = (img * 255).astype(np.uint8).copy()
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    
    if mask is None:
        return img
        
    mask = mask.squeeze().cpu().numpy()
    h, w = mask.shape
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)
    
    unique_labels = np.unique(mask)
    for l in unique_labels:
        if l == 0 or l == 255: continue 
        color_mask[mask == l] = COLORS[l % 21] 
        
    mask_bool = (mask > 0) & (mask != 255)
    if mask_bool.any():
        img[mask_bool] = cv2.addWeighted(img[mask_bool], 1 - alpha, color_mask[mask_bool], alpha, 0)
    
    return img

def visualize_mask_only(mask):
    """Visualize segmentation mask as pure colored mask (no background image).
    
    Args:
        mask: (H, W) tensor with class labels
    
    Returns:
        np.ndarray: (H, W, 3) BGR image with colored segmentation
    """
    if mask is None:
        # Return black image
        return np.zeros((224, 224, 3), dtype=np.uint8)
    
    mask = mask.squeeze().cpu().numpy()
    h, w = mask.shape
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)
    
    # Assign colors to each class
    unique_labels = np.unique(mask)
    for l in unique_labels:
        if l == 0:  # Background - keep black
            continue
        if l == 255:  # Ignore - keep black
            continue
        color_mask[mask == l] = COLORS[l % 21]
    
    return color_mask


def visualize_instance_masks(
    image_tensor: torch.Tensor,
    masks: torch.Tensor,
    labels: torch.Tensor = None,
    alpha: float = 0.5
) -> np.ndarray:
    """Visualize instance segmentation masks with unique colors per instance.
    
    Args:
        image_tensor: (3, H, W) normalized image tensor
        masks: (N, H, W) binary instance masks
        labels: (N,) class labels (optional, for labeling)
        alpha: Transparency for overlay
    
    Returns:
        np.ndarray: (H, W, 3) BGR image with colored instances
    """
    # Denormalize image
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(image_tensor.device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(image_tensor.device)
    
    img = image_tensor * std + mean
    img = img.permute(1, 2, 0).cpu().numpy()
    img = (img * 255).astype(np.uint8).copy()
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    
    if masks is None or len(masks) == 0:
        return img
    
    # Generate consistent but distinct colors for each instance
    np.random.seed(42)  # For consistency
    instance_colors = np.random.randint(50, 255, size=(len(masks), 3))
    
    # Create overlay
    overlay = img.copy()
    
    for i, mask in enumerate(masks):
        # Get mask as numpy
        mask_np = mask.cpu().numpy() > 0.5
        
        if not mask_np.any():
            continue
        
        # Apply colored mask
        color = instance_colors[i]
        overlay[mask_np] = overlay[mask_np] * (1 - alpha) + color * alpha
    
    return overlay.astype(np.uint8)


def visualize_instance_masks_only(masks: torch.Tensor) -> np.ndarray:
    """Visualize instance masks as pure colored masks (no background image).
    
    Args:
        masks: (N, H, W) binary instance masks
    
    Returns:
        np.ndarray: (H, W, 3) BGR image with colored instances
    """
    if masks is None or len(masks) == 0:
        return np.zeros((224, 224, 3), dtype=np.uint8)
    
    masks_np = masks.cpu().numpy()
    H, W = masks_np.shape[1:]
    color_mask = np.zeros((H, W, 3), dtype=np.uint8)
    
    # Generate consistent colors
    np.random.seed(42)
    instance_colors = np.random.randint(50, 255, size=(len(masks), 3))
    
    for i, mask in enumerate(masks_np):
        mask_bool = mask > 0.5
        if mask_bool.any():
            color_mask[mask_bool] = instance_colors[i]
    
    return color_mask
