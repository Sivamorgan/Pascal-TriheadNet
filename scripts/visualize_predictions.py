
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn.functional as F
import numpy as np
import cv2
import hydra
from omegaconf import DictConfig
from tqdm import tqdm
from pathlib import Path
from PIL import Image

# Fix path to include root
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from data.Dataset import PascalUnifiedDataset
from models.architectures import JointModel
from utils.postprocess import postprocess_detection, postprocess_instance_segmentation, paste_masks_in_image
from utils.visualization import draw_boxes, visualize_segmentation, visualize_mask_only, visualize_instance_masks, visualize_instance_masks_only

def denormalize(tensor):
    """Denormalize image tensor for visualization.
    Args: tensor (3, H, W)
    Returns: numpy (H, W, 3) in [0, 255]
    """
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(tensor.device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(tensor.device)
    image = tensor * std + mean
    image = torch.clamp(image, 0, 1)
    return (image.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)


@hydra.main(version_base=None, config_path="../configs", config_name="joint_training")
def main(cfg: DictConfig):
    # To disable hydra logging: python script.py hydra.output_subdir=null hydra.run.dir=.
    
    device = torch.device(cfg.training.device if torch.cuda.is_available() else 'cpu')
    output_dir = Path("outputs/visualization")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load Model
    print("Initializing model...")
    model = JointModel(cfg).to(device)
    
    # Load Checkpoint
    checkpoint_path = cfg.training.resume or os.path.join(cfg.training.checkpoint_dir, 'checkpoint_epoch_20.pth')
    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location=device)
        state_dict = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
        
        try:
            model.load_state_dict(state_dict)
        except RuntimeError as e:
             print(f"Warning: State dict mismatch, trying non-strict load: {e}")
             model.load_state_dict(state_dict, strict=False)
    else:
        print(f"Checkpoint not found at {checkpoint_path}, using random weights for demo.")
        
    model.eval()
    
    # Dataset
    print("Loading dataset...")
    dataset = PascalUnifiedDataset(
        data_root=cfg.data.root,
        split='val',
        subset='segmentation', # Visualize only segmented samples
        task='all',
        limit=cfg.data.limit if cfg.data.limit is not None else 20 
    )
    
    print(f"Visualizing {len(dataset)} samples...")
    print(f"Saving outputs to {output_dir.resolve()}")
    
    for i in tqdm(range(len(dataset))):
        sample = dataset[i]
        img_id = sample['image_id']
        
        # Prepare inputs for model (224×224)
        img_tensor = sample['image'].unsqueeze(0).to(device)
        
        # Use dataset's image (already transformed/cropped) for visualization
        # This ensures alignment between model input and visualization
        original_img_np = denormalize(sample['image'].to(device))
        orig_h, orig_w = original_img_np.shape[:2]
        
        # Inference (model expects 224×224)
        with torch.no_grad():
            features = model.backbone.forward_features(img_tensor)
            pyramid = model.neck(features)
            
            # Detection
            det_out = model.det_head(pyramid)
            # Use sensible threshold (0.3) for real usage. 
            thresh = 0.2 if 'vis_thresh' not in cfg else cfg.vis_thresh
            
            # Postprocess returns a list (one per batch item)
            processed_dets_list = postprocess_detection(det_out, score_thresh=thresh)
            processed_dets = processed_dets_list[0]
            
            # Semantic (argmax at native resolution, then upsample)
            sem_out = model.sem_head(pyramid)
            sem_pred_native = torch.argmax(sem_out, dim=1)[0]  
            sem_pred = F.interpolate(
                sem_pred_native.unsqueeze(0).unsqueeze(0).float(),
                size=(orig_h, orig_w),
                mode='nearest' 
            )[0, 0].long()  # (orig_h, orig_w)
            
            # Instance Segmentation
            inst_results = None
            if len(processed_dets['boxes']) > 0:
                # Run instance head with detected boxes
                pred_boxes_list = [processed_dets['boxes']]
                pred_labels_list = [processed_dets['labels']]
                
                inst_out = model.inst_head(pyramid, pred_boxes_list, pred_labels_list)
                
                # Post-process to full-image masks (at 224×224)
                inst_results = postprocess_instance_segmentation(
                    inst_out,
                    [processed_dets],
                    image_size=(orig_h, orig_w),
                    mask_threshold=0.5
                )[0]
        
        # Predictions are already in 224x224 coords (same as visualization image)
        pred_boxes = processed_dets['boxes']
        gt_boxes = sample['boxes']
        
        # --- Upscale for Visualization (Target 800px) ---
        VIS_SIZE = 800
        scale_factor = VIS_SIZE / orig_h
        
        vis_img_np = cv2.resize(original_img_np, (VIS_SIZE, VIS_SIZE), interpolation=cv2.INTER_LINEAR)
        img_tensor_vis = torch.from_numpy(vis_img_np).permute(2, 0, 1).float() / 255.0
        
        # Scale Boxes
        def scale_boxes(boxes, scale):
            if boxes is None or len(boxes) == 0: return boxes
            b = boxes.clone()
            b *= scale
            return b

        gt_boxes_vis = scale_boxes(gt_boxes, scale_factor)
        pred_boxes_vis = scale_boxes(pred_boxes, scale_factor)
        

def draw_boxes_on_image(img, boxes, labels, scores=None, color_override=None):
    """Draw boxes directly on BGR numpy image."""
    img = img.copy()
    if boxes is None or len(boxes) == 0:
        return img
    
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = box.int().tolist()
        label = int(labels[i])
        
        # Color
        np.random.seed(label)
        color = color_override if color_override is not None else np.random.randint(0, 255, (3,), dtype=np.uint8).tolist()
        
        # Draw box
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        
        # Label
        cls_name = PascalUnifiedDataset.VOC_CLASSES[label] if label < len(PascalUnifiedDataset.VOC_CLASSES) else str(label)
        text = f"{cls_name}"
        if scores is not None:
            text += f" {scores[i]:.2f}"
            
        (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img, (x1, y1-20), (x1+w, y1), color, -1)
        cv2.putText(img, text, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
    return img

def draw_mask_on_image(img, mask, alpha=0.5, random_colors=False):
    """Overlay mask on BGR numpy image."""
    img = img.copy()
    if mask is None: return img
    
    mask = mask.cpu().numpy() if torch.is_tensor(mask) else mask
    if mask.ndim == 3: mask = mask.squeeze()
    
    # Create color layer
    h, w = img.shape[:2]
    color_mask = np.zeros_like(img)
    
    unique_ids = np.unique(mask)
    for uid in unique_ids:
        if uid == 0 or uid == 255: continue
        
        if random_colors:
             np.random.seed(int(uid) + 100) # Offset for instances
             color = np.random.randint(0, 255, (3,), dtype=np.uint8).tolist()
        else:
             np.random.seed(int(uid))
             color = np.random.randint(0, 255, (3,), dtype=np.uint8).tolist()
             
        color_mask[mask == uid] = color
        
    mask_bool = (mask > 0) & (mask != 255)
    if mask_bool.any():
        img[mask_bool] = cv2.addWeighted(img[mask_bool], 1-alpha, color_mask[mask_bool], alpha, 0)
    return img

@hydra.main(version_base=None, config_path="../configs", config_name="joint_training")
def main(cfg: DictConfig):
    device = torch.device(cfg.training.device if torch.cuda.is_available() else 'cpu')
    output_dir = Path("outputs/visualization")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load Model
    print("Initializing model...")
    model = JointModel(cfg).to(device)
    
    checkpoint_path = cfg.training.resume or os.path.join(cfg.training.checkpoint_dir, 'checkpoint_epoch_20.pth')
    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt, strict=False)
    model.eval()
    
    # Dataset
    dataset = PascalUnifiedDataset(data_root=cfg.data.root, split='val', subset='segmentation', limit=cfg.data.limit or 20)
    print(f"Visualizing {len(dataset)} samples...")
    
    for i in tqdm(range(len(dataset))):
        sample = dataset[i]
        img_id = sample['image_id']
        
        # 1. Load Original High-Res Image
        img_path = os.path.join(cfg.data.root, 'VOC2012_train_val', 'JPEGImages', img_id + '.jpg')
        if not os.path.exists(img_path):
            img_path = os.path.join(cfg.data.root, 'VOC2012_test', 'JPEGImages', img_id + '.jpg')
            
        orig_img_bgr = cv2.imread(img_path)
        if orig_img_bgr is None: continue
        
        # 2. Use Original Dimensions (No Resize)
        h, w = orig_img_bgr.shape[:2]
        vis_img_bgr = orig_img_bgr.copy()
        
        # 3. Scale Factors (Visualization / Model Input 224)
        scale_x = w / 224.0
        scale_y = h / 224.0
        
        # 4. Inference
        img_tensor = sample['image'].unsqueeze(0).to(device)
        with torch.no_grad():
            features = model.backbone.forward_features(img_tensor)
            pyramid = model.neck(features)
            
            # Det
            det_out = model.det_head(pyramid)
            processed_dets = postprocess_detection(det_out, score_thresh=0.3)[0]
            
            # Sem
            sem_out = model.sem_head(pyramid)
            sem_pred = torch.argmax(sem_out, dim=1)[0]
            
            # Inst
            inst_results = None
            if len(processed_dets['boxes']) > 0:
                inst_out = model.inst_head(pyramid, [processed_dets['boxes']], [processed_dets['labels']])
                inst_results = postprocess_instance_segmentation(inst_out, [processed_dets], image_size=(224,224))[0]

        # 5. Scale Predictions
        def scale_box(b): return b * torch.tensor([scale_x, scale_y, scale_x, scale_y], device=b.device)
        
        pred_boxes_vis = scale_box(processed_dets['boxes'])
        gt_boxes_vis = scale_box(sample['boxes'])
        sem_pred_vis = F.interpolate(sem_pred.float()[None,None], size=(h, w), mode='nearest')[0,0].long()
        
        # gt_sem_vis: sample['semantic_mask'] is (1, 224, 224) -> unsqueeze(0) makes (1, 1, 224, 224) -> OK
        if sample.get('semantic_mask') is not None:
             gt_mask = sample['semantic_mask'].float()
             if gt_mask.dim() == 2: # (H, W)
                 gt_mask = gt_mask[None, None]
             elif gt_mask.dim() == 3: # (C, H, W)
                 gt_mask = gt_mask.unsqueeze(0)
             gt_sem_vis = F.interpolate(gt_mask, size=(h, w), mode='nearest')[0,0].long()
        else:
             gt_sem_vis = None
        
        # 6. Draw
        
        # Detection (Keep Overlay)
        img_gt_det = draw_boxes_on_image(vis_img_bgr, gt_boxes_vis, sample['labels'], color_override=(0,255,0))
        img_pred_det = draw_boxes_on_image(vis_img_bgr, pred_boxes_vis, processed_dets['labels'], scores=processed_dets['scores'], color_override=(0,0,255))
        
        # Semantic (Mask Only - Black Background)
        black_bg = np.zeros_like(vis_img_bgr)
        img_gt_sem = draw_mask_on_image(black_bg, gt_sem_vis, alpha=1.0)
        img_pred_sem = draw_mask_on_image(black_bg, sem_pred_vis, alpha=1.0)
        
        # Instance (Mask Only - Black Background)
        img_gt_inst = black_bg.copy()
        
        # Load GT Instance PNG (High Res)
        inst_path = os.path.join(cfg.data.root, 'VOC2012_train_val', 'SegmentationObject', img_id + '.png')
        if not os.path.exists(inst_path):
             inst_path = os.path.join(cfg.data.root, 'VOC2012_test', 'SegmentationObject', img_id + '.png')
             
        if os.path.exists(inst_path):
            # Load as indexed image (PIL is safer for palette)
            gt_inst_pil = Image.open(inst_path)
            if gt_inst_pil.size != (w, h):
                gt_inst_pil = gt_inst_pil.resize((w, h), Image.NEAREST)
            gt_inst_np = np.array(gt_inst_pil) # (H, W) with instance IDs
            
            # Draw
            img_gt_inst = draw_mask_on_image(black_bg, gt_inst_np, alpha=1.0, random_colors=True)
        
        img_pred_inst = black_bg.copy()
        if inst_results:
             # Resize inst masks
             inst_masks_vis = F.interpolate(inst_results['masks'][None], size=(h, w), mode='bilinear')[0] > 0.5
             img_pred_inst = draw_mask_on_image(black_bg, (inst_masks_vis * (torch.arange(len(inst_masks_vis), device=device)[:,None,None]+1)).sum(0), alpha=1.0, random_colors=True)

        # Stack Rows
        def add_sep(i1, i2): return np.hstack([i1, np.ones((h, 10, 3), dtype=np.uint8)*255, i2])
        
        row1 = add_sep(img_gt_det, img_pred_det)
        row2 = add_sep(img_gt_sem, img_pred_sem)
        row3 = add_sep(img_gt_inst, img_pred_inst)
        
        final = np.vstack([row1, row2, row3])
        cv2.imwrite(str(output_dir / f"vis_{img_id}.png"), final)

if __name__ == '__main__':
    main()

