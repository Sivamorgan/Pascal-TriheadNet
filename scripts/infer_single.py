import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
import cv2
import hydra
from omegaconf import DictConfig
from pathlib import Path
from PIL import Image
from torchvision.transforms import v2 as transforms
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from models.architectures import JointModel
from utils.postprocess import postprocess_detection, postprocess_instance_segmentation
from utils.visualization import draw_boxes, visualize_segmentation, visualize_mask_only, visualize_instance_masks, visualize_instance_masks_only
from data.Dataset import PascalUnifiedDataset # For class names

class InferencePipeline:
    """Reusable inference pipeline for easy integration with Streamlit/FastAPI."""
    
    def __init__(self, checkpoint_path, cfg=None, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # Load config (dummy or minimal if not provided, but hydra usually provides)
        # Ideally we load the config used during training
        if cfg is None:
             # Fallback to loading from hydra compose if used strictly as library
             # For now assume cfg is passed or we load defaults
             pass
             
        self.model = JointModel(cfg).to(self.device)
        self.model.eval()
        
        # Check if loading a Quantized model
        is_quantized = "quantized" in str(checkpoint_path)
        if is_quantized:
            print("Detected Quantized Checkpoint. Applying quantization structure...")
            # We must quantize the model skeletal structure FIRST to match the weights
            
            self.model = torch.ao.quantization.quantize_dynamic(
                self.model, 
                {torch.nn.Linear,torch.nn.Conv2d}, 
                dtype=torch.qint8
            )
        
        print(f"Loading checkpoint: {checkpoint_path}")
        if os.path.exists(checkpoint_path):
            ckpt = torch.load(checkpoint_path, map_location=self.device)
            state_dict = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
            self.model.load_state_dict(state_dict, strict=False)
            
            # MEMORY CLEANUP: Free the checkpoint dict immediately
            del ckpt
            del state_dict
            import gc
            gc.collect()
        else:
            raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

        # Standard transform
        self.transform = transforms.Compose([
            transforms.ToImage(),
            transforms.Resize((224, 224)), # Force resize to model input
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def predict(self, image_input):
        """
        Run inference on a single image.
        Args:
            image_input: str (path) or PIL.Image or numpy array
        Returns:
            vis_img: numpy array (H, W, 3) visualization (at original resolution)
            predictions: dict of raw results
        """
        # Load Image
        if isinstance(image_input, str):
            image_pil = Image.open(image_input).convert("RGB")
        elif isinstance(image_input, np.ndarray):
            image_pil = Image.fromarray(image_input).convert("RGB")
        else:
            image_pil = image_input.convert("RGB")
            
        # 1. Store Original Dimensions
        orig_w, orig_h = image_pil.size
        
        # MEMORY SAFEGUARD: Resize HUGE images for visualization/processing
        # This prevents OOM on massive phone photos (e.g. 50MP)
        MAX_DIM = 1200
        if max(orig_w, orig_h) > MAX_DIM:
            scale = MAX_DIM / max(orig_w, orig_h)
            new_w = int(orig_w * scale)
            new_h = int(orig_h * scale)
            print(f"Resizing input image from {orig_w}x{orig_h} to {new_w}x{new_h} to save memory.")
            image_pil = image_pil.resize((new_w, new_h), Image.BICUBIC)
            
        # Update dimensions to the (potentially resized) working size
        vis_w, vis_h = image_pil.size
        original_img_np = np.array(image_pil)
        
        # 2. Preprocess (Resize to 224x224 for Model)
        # Use transforms.Resize((224, 224)) which is anti-aliased and rigorous
        prep_transform = transforms.Compose([
            transforms.ToImage(),
            transforms.Resize((224, 224)),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        img_tensor = prep_transform(image_pil).unsqueeze(0).to(self.device) # (1, 3, 224, 224)
        
        # Calculate Scale Factors (Visualization / 224)
        scale_x = vis_w / 224.0
        scale_y = vis_h / 224.0
        
        # Inference
        with torch.no_grad():
            features = self.model.backbone.forward_features(img_tensor)
            pyramid = self.model.neck(features)
            
            # --- Detection ---
            det_out = self.model.det_head(pyramid)
            processed_dets_list = postprocess_detection(det_out, score_thresh=0.2)
            processed_dets = processed_dets_list[0] # Boxes are in 0..224
            
            # Save 224-scale boxes for Instance Head later
            boxes_224 = processed_dets['boxes']
            
            # Scale Boxes to Visualization Resolution
            if len(boxes_224) > 0:
                boxes_vis = boxes_224.clone()
                boxes_vis[:, 0] *= scale_x
                boxes_vis[:, 2] *= scale_x
                boxes_vis[:, 1] *= scale_y
                boxes_vis[:, 3] *= scale_y
                processed_dets['boxes'] = boxes_vis # Update to vis res
            
            # --- Semantic ---
            sem_out = self.model.sem_head(pyramid)
            # Interpolate to Visualization Resolution directly
            sem_pred = F.interpolate(
                sem_out, 
                size=(vis_h, vis_w), 
                mode='nearest' # Class indices
            )
            sem_pred = torch.argmax(sem_pred, dim=1)[0].cpu() # (H, W)
            
            # --- Instance ---
            inst_results = None
            if len(boxes_224) > 0: # Check original detection count
                # Note: Instance Head needs 224-compatible boxes to align with features
                inst_out = self.model.inst_head(pyramid, [boxes_224], [processed_dets['labels']])
                
                # But Post-Processing (Pasting) should happen on Visualization Resolution
                inst_results = postprocess_instance_segmentation(
                    inst_out,
                    [processed_dets], # Contains 'boxes' which we updated to (vis)
                    image_size=(vis_h, vis_w),
                    mask_threshold=0.5
                )[0]
                
        # Visualization (Now on full-res original_img_np)
        return self._visualize_overlay(original_img_np, processed_dets, sem_pred, inst_results)

    def _visualize_overlay(self, img_np, dets, sem, inst):
        """
        Visualizes everything on a SINGLE image overlay (YOLO-style).
        Directly uses OpenCV on the numpy image for maximum speed and sharpness.
        """
        vis_img = img_np.copy()
        
        # --- 1. Semantic Segmentation Overlay ---
        # Fixed Pascal VOC Colors (Simplified for display)
        # We reuse the logic but implemented efficiently for Numpy
        if sem is not None:
             # Create colored mask
             sem_np = sem.numpy()
             h, w = sem_np.shape
             color_mask = np.zeros_like(vis_img)
             
             # Fast color mapping
             unique_labels = np.unique(sem_np)
             for l in unique_labels:
                 if l == 0 or l == 255: continue
                 # Generate a color based on label ID (Golden logic: (l * 50) % 255...)
                 # Or just use a fixed palette if we imported it, but simple hash is fast
                 np.random.seed(l)
                 color = np.random.randint(0, 255, (3,), dtype=np.uint8).tolist()
                 color_mask[sem_np == l] = color
             
             # Blend: img = img * 0.6 + mask * 0.4 where mask > 0
             mask_bool = (sem_np > 0) & (sem_np != 255)
             if mask_bool.any():
                 vis_img[mask_bool] = cv2.addWeighted(vis_img[mask_bool], 0.6, color_mask[mask_bool], 0.4, 0)

        # --- 2. Instance Segmentation Overlay ---
        if inst and len(inst['masks']) > 0:
            for i, mask in enumerate(inst['masks']):
                 mask_np = mask.squeeze().numpy()
                 if mask_np.max() < 1: continue # Skip empty
                 
                 # Random Color for instance
                 np.random.seed(i + 100) # Consistent color per instance index
                 color = np.random.randint(0, 255, (3,), dtype=np.uint8).tolist()
                 
                 # 1. Fill (Alpha blend)
                 mask_bool = mask_np > 0
                 # Simplified blend for speed
                 vis_img[mask_bool] = (vis_img[mask_bool] * 0.5 + np.array(color) * 0.5).astype(np.uint8)
                 
                 # 2. Contour (Sharp edge)
                 contours, _ = cv2.findContours(mask_np.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                 cv2.drawContours(vis_img, contours, -1, color, 2)

        # --- 3. Detections ---
        boxes = dets['boxes']
        labels = dets['labels']
        scores = dets['scores']
        
        # Load class names
        class_names = PascalUnifiedDataset.VOC_CLASSES
        
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = box.int().tolist()
            cls_id = int(labels[i])
            score = float(scores[i])
            
            # Robust Box Color
            np.random.seed(cls_id)
            color = np.random.randint(0, 255, (3,), dtype=np.uint8).tolist() # BGR
            
            # Draw Box
            cv2.rectangle(vis_img, (x1, y1), (x2, y2), color, 2)
            
            # Label
            # Safety check for index
            cls_name = class_names[cls_id] if 0 <= cls_id < len(class_names) else str(cls_id)
            label_text = f"{cls_name} {score:.2f}"
            t_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            c2 = x1 + t_size[0], y1 - t_size[1] - 3
            cv2.rectangle(vis_img, (x1, y1), c2, color, -1, cv2.LINE_AA) # Filled header
            cv2.putText(vis_img, label_text, (x1, y1 - 2), 0, 0.5, [255, 255, 255], thickness=1, lineType=cv2.LINE_AA)
        vis_img = cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR)
        return vis_img

    def _visualize_debug(self, img_np, dets, sem, inst):
        """Old 3-panel side-by-side visualization."""
        img_tensor_vis = torch.from_numpy(img_np).permute(2, 0, 1).float() / 255.0
        
        # Helper titles
        def add_title(img, text):
            border = 30
            img_border = cv2.copyMakeBorder(img, border, 0, 0, 0, cv2.BORDER_CONSTANT, value=(0,0,0))
            cv2.putText(img_border, text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            return img_border
            
        # 1. Detection
        vis_det = draw_boxes(img_tensor_vis, dets['boxes'], dets['labels'], dets['scores'])
        vis_det = add_title(vis_det, f"Detection (N={len(dets['boxes'])})")
        
        # 2. Semantic
        vis_sem = visualize_segmentation(img_tensor_vis, sem.to(img_tensor_vis.device))
        vis_sem = add_title(vis_sem, "Semantic Segmentation")
        
        # 3. Instance
        if inst and len(inst['masks']) > 0:
            vis_inst = visualize_instance_masks(img_tensor_vis, inst['masks'], inst['labels'])
            vis_inst_mask = visualize_instance_masks_only(inst['masks'])
            vis_inst = add_title(vis_inst, f"Instance (N={len(inst['masks'])})")
        else:
            vis_inst = visualize_instance_masks(img_tensor_vis, None)
            vis_inst = add_title(vis_inst, "Instance (None)")
            
        # Stack horizontal
        row1 = np.hstack([vis_det, vis_sem, vis_inst])
        return row1

@hydra.main(version_base=None, config_path="../configs", config_name="joint_training")
def main(cfg: DictConfig):
    # Parse args via hydra overrides or just standard argparse wrapper?
    # Hydra takes over argv, so we rely on config or overrides.
    # Usage: python scripts/infer_single.py +image_path="test.jpg"
    
    image_path = cfg.get('image_path', None)
    if not image_path:
        print("Error: Please provide +image_path='/path/to/image.jpg'")
        return

    checkpoint = cfg.training.resume or os.path.join(cfg.training.checkpoint_dir, 'checkpoint_epoch_48.pth')
    
    pipeline = InferencePipeline(checkpoint, cfg, device=cfg.training.device)
    vis_result = pipeline.predict(image_path)
    
    # Save
    out_name =  f"vis_{Path(image_path).stem}.png"
    cv2.imwrite(out_name, vis_result)
    print(f"Saved visualization to {out_name}")

if __name__ == '__main__':
    main()
