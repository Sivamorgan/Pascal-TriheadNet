import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import torch
import torch.nn.functional as F
import hydra
from tqdm import tqdm
import json
from pathlib import Path
from data.Dataset import PascalUnifiedDataset,joint_collate_fn
from torch.utils.data import DataLoader
from models.architectures import JointModel
from utils.postprocess import postprocess_detection, postprocess_instance_segmentation
from utils.metrics import MetricLogger

@hydra.main(version_base=None, config_path="../configs", config_name="joint_training")
def main(cfg):
    """Evaluate trained model on val or test split.
    """
    # Configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint_path = cfg.get('checkpoint', cfg.training.resume)
    split = cfg.get('split', 'val')
    output_dir = Path(f"outputs/evaluation/{split}")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print(f"🔍 Evaluation Script")
    print("="*80)
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Split: {split}")
    print(f"Device: {device}")
    print("="*80)
    
    # Load model
    print("\n📦 Loading model...")
    model = JointModel(cfg).to(device)
    
    if os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=device)
        state_dict = ckpt.get('model_state_dict', ckpt)
        model.load_state_dict(state_dict, strict=False)
        #epoch = ckpt.get('epoch', 'unknown')
        #print(f"✓ Loaded checkpoint from {checkpoint_path}")
    else:
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        return
    
    model.eval()
    
    # Load dataset
    print(f"\n📁 Loading {split} dataset...")
    dataset = PascalUnifiedDataset(
        data_root=cfg.data.root,
        split=split,
        subset='all',  # All images
        task='all',
        limit=cfg.data.get('limit', None),
        use_trainval_split=False  # Use actual split files
    )
    
    # Check GT availability
    has_detection_gt = check_detection_gt(split, cfg.data.root)
    has_semantic_gt = check_semantic_gt(split, cfg.data.root)
    has_instance_gt = check_instance_gt(split, cfg.data.root)
    
    print(f"\n📊 Ground Truth Availability:")
    print(f"  Detection: {'✓' if has_detection_gt else '✗'}")
    print(f"  Semantic:  {'✓' if has_semantic_gt else '✗'}")
    print(f"  Instance:  {'✓' if has_instance_gt else '✗'}")
    
    if not (has_detection_gt or has_semantic_gt or has_instance_gt):
        print(f"\n⚠️  No ground truth found for {split} split. Running inference only...")
    
    # DataLoader
    loader = DataLoader(
        dataset,
        batch_size=cfg.data.batch_size,
        shuffle=False,
        num_workers=cfg.data.num_workers,
        collate_fn=joint_collate_fn
    )
    
    # Initialize metrics
    metrics = MetricLogger(device=device, num_classes=21)
    
    # Enable class metrics if requested in config
    if cfg.get('class_metrics', False):
        print("ℹ️  Enabling per-class metrics reporting") # Original line
        if metrics.enabled:
            metrics.det_map.class_metrics = True
            metrics.inst_map.class_metrics = True
    
    # Evaluation loop
    print(f"\n🚀 Running evaluation on {len(dataset)} images...")
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(loader, desc="Evaluating")):
            images = batch['images'].to(device)
            
            # Forward pass
            features = model.backbone.forward_features(images)
            pyramid = model.neck(features)
            
            # Detection
            det_out = model.det_head(pyramid)
            det_results = postprocess_detection(det_out, score_thresh=0.25, nms_thresh=0.05)
            
            # Semantic segmentation
            sem_out = model.sem_head(pyramid)
            sem_pred = torch.argmax(sem_out, dim=1)
            
            # Instance segmentation (if detections exist)
            inst_predictions_batch = []
            
            # Prepare batch inputs for instance head
            all_boxes = [res['boxes'] for res in det_results]
            all_labels = [res['labels'] for res in det_results]
            
            # Check if any boxes exist in the batch
            total_boxes = sum(len(b) for b in all_boxes)
            
            if total_boxes > 0:
                inst_all_masks = model.inst_head(pyramid, all_boxes, all_labels) # (Total_N, 1, 28, 28)
                inst_predictions_batch = postprocess_instance_segmentation(
                    inst_all_masks,
                    det_results,
                    image_size=(cfg.data.image_size, cfg.data.image_size),
                    mask_threshold=0.5
                )
            else:
                # No detections in entire batch
                inst_predictions_batch = [
                    {
                        'boxes': torch.zeros((0, 4), device=device),
                        'labels': torch.zeros((0,), dtype=torch.long, device=device),
                        'scores': torch.zeros((0,), device=device),
                        'masks': torch.zeros((0, cfg.data.image_size, cfg.data.image_size), device=device, dtype=torch.uint8)
                    }
                    for _ in range(len(images))
                ]
            
            
            # Update metrics
            if has_detection_gt:
                gt_boxes = batch['boxes']
                gt_labels = batch['labels']  
                targets = []
                for i in range(len(images)):
                    targets.append({
                        'boxes': gt_boxes[i].to(device),
                        'labels': gt_labels[i].to(device)
                    })
                
                metrics.update_detection(det_results, targets)
            
            if has_semantic_gt:
                semantic_masks = batch.get('semantic_masks')  # List[(1, H, W) or None]
                if semantic_masks is not None:
                    valid_sem_indices = [i for i, m in enumerate(semantic_masks) if m is not None]
                    if valid_sem_indices:
                        sem_target = torch.stack([semantic_masks[i].squeeze(0) for i in valid_sem_indices]).to(device)
                        sem_pred_valid = sem_pred[valid_sem_indices] # (N, H, W)
                        sem_pred_resized = F.interpolate(
                            sem_pred_valid.unsqueeze(1).float(), 
                            size=sem_target.shape[-2:],
                            mode='nearest',
                        ).squeeze(1).long()
                        
                        metrics.update_semantic(sem_pred_resized, sem_target)
            
            # Instance segmentation metrics
            if has_instance_gt:
                gt_boxes = batch['boxes']           # List of tensors
                gt_labels = batch['labels']         # List of tensors
                instance_masks_gt = batch.get('instance_masks_28')
                inst_targets_batch = []
                
                from utils.postprocess import paste_masks_in_image

                for i in range(len(images)):
                    # Get GT if available
                    has_gt = (instance_masks_gt is not None) and (i < len(instance_masks_gt)) and (instance_masks_gt[i] is not None)
                    
                    if has_gt and len(gt_boxes[i]) > 0:  
                        gt_masks_28 = instance_masks_gt[i].to(device) # (N, 1, 28, 28)
                        
                        if len(gt_masks_28) > 0:
                            # Upsample 28×28 back to 224×224
                            gt_masks_full = paste_masks_in_image(
                                gt_masks_28,           # (N, 1, 28, 28)
                                gt_boxes[i].to(device), # (N, 4)
                                image_shape=(224, 224),
                                threshold=0.5
                            ).squeeze(1).to(torch.uint8)  # (N, 224, 224)
                            
                            inst_targets_batch.append({
                                'boxes': gt_boxes[i].to(device),
                                'labels': gt_labels[i].to(device),
                                'masks': gt_masks_full  # (N, 224, 224)
                            })
                        else:
                            # Empty masks list
                            inst_targets_batch.append({
                                'boxes': torch.zeros((0, 4), device=device),
                                'labels': torch.zeros((0,), dtype=torch.long, device=device),
                                'masks': torch.zeros((0, 224, 224), device=device, dtype=torch.uint8)
                            })
                    else:
                        # No instance GT for this image
                        inst_targets_batch.append({
                            'boxes': torch.zeros((0, 4), device=device),
                            'labels': torch.zeros((0,), dtype=torch.long, device=device),
                            'masks': torch.zeros((0, 224, 224), device=device, dtype=torch.uint8)
                        })
                metrics.update_instance(inst_predictions_batch, inst_targets_batch)
    
    # Extract per-class metrics manually if enabled (MetricLogger.compute() drops unknown keys via return dict construction)
    det_per_class = None
    inst_per_class = None
    if cfg.get('class_metrics', False) and metrics.enabled:
        try:
            # Detection
            d_res = metrics.det_map.compute()
            if 'map_per_class' in d_res:
                det_per_class = d_res['map_per_class']
            
            # Instance
            i_res = metrics.inst_map.compute()
            if 'map_per_class' in i_res:
                inst_per_class = i_res['map_per_class']
        except Exception as e:
            print(f"Warning: Could not extract per-class metrics: {e}")

    # Compute all metrics at once from MetricLogger (this also resets them)
    all_metrics = metrics.compute()
    results = {}
    
    print("\n" + "="*80)
    print("📊 EVALUATION RESULTS")
    print("="*80)
    
    if has_detection_gt:
        results['detection'] = {
            'mAP': all_metrics['det_map'],
            'mAP_50': all_metrics['det_map_50'],
            'mAP_75': all_metrics['det_map_75']
        }
        print(f"\n🎯 Detection:")
        print(f"  mAP:      {all_metrics['det_map']:.4f}")
        print(f"  mAP@50:   {all_metrics['det_map_50']:.4f}")
        print(f"  mAP@75:   {all_metrics['det_map_75']:.4f}")
        
        if det_per_class is not None:
            print("\n  🎯 Detection per class:")
            for i, score in enumerate(det_per_class):
                if i + 1 < len(PascalUnifiedDataset.VOC_CLASSES):
                    cls_name = PascalUnifiedDataset.VOC_CLASSES[i + 1] # Skip background (index 0)
                    print(f"    {cls_name:<15}: {score.item():.4f}")
            results['detection']['per_class'] = det_per_class.tolist()
    
    if has_semantic_gt:
        results['semantic'] = {
            'mIoU': all_metrics['sem_miou'],
            'pixel_acc': all_metrics['sem_acc']
        }
        print(f"\n🎨 Semantic Segmentation:")
        print(f"  mIoU:     {all_metrics['sem_miou']:.4f}")
        print(f"  Accuracy: {all_metrics['sem_acc']:.4f}")
    
    # Instance segmentation results
    if has_instance_gt and all_metrics.get('inst_map', 0.0) > 0:
        results['instance'] = {
            'mask_mAP': all_metrics['inst_map'],
            'mask_mAP_50': all_metrics['inst_map_50']
        }
        print(f"\n🎭 Instance Segmentation:")
        print(f"  mask mAP:    {all_metrics['inst_map']:.4f}")
        print(f"  mask mAP@50: {all_metrics['inst_map_50']:.4f}")
        
        if inst_per_class is not None:
            print("\n  🎭 Instance Segmentation per class:")
            for i, score in enumerate(inst_per_class):
                if i + 1 < len(PascalUnifiedDataset.VOC_CLASSES):
                    cls_name = PascalUnifiedDataset.VOC_CLASSES[i + 1]
                    print(f"    {cls_name:<15}: {score.item():.4f}")
            results['instance']['per_class'] = inst_per_class.tolist()
    elif has_instance_gt:
        print(f"\n⚠️  Instance metrics: 0.0000 (no instances detected/matched)")
    
    # Save results
    results['checkpoint'] = checkpoint_path
    results['split'] = split
    results['num_images'] = len(dataset)
    
    output_file = output_dir / f"results_{Path(checkpoint_path).stem}.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Results saved to: {output_file}")
    print("="*80)


def check_detection_gt(split, data_root):
    """Check if detection GT annotations exist."""
    import os
    # Check for Annotations directory (XML files with boxes)
    anno_dir = os.path.join(data_root, f"VOC2012_{split if split != 'val' else 'train_val'}", 
                            "Annotations")
    return os.path.exists(anno_dir) and len(os.listdir(anno_dir)) > 0


def check_semantic_gt(split, data_root):
    """Check if semantic segmentation GT exists."""
    import os
    seg_dir = os.path.join(data_root, f"VOC2012_{split if split != 'val' else 'train_val'}", 
                            "SegmentationClass")
    return os.path.exists(seg_dir) and len(os.listdir(seg_dir)) > 0


def check_instance_gt(split, data_root):
    """Check if instance segmentation GT exists."""
    import os
    inst_dir = os.path.join(data_root, f"VOC2012_{split if split != 'val' else 'train_val'}", 
                             "SegmentationObject")
    return os.path.exists(inst_dir) and len(os.listdir(inst_dir)) > 0


if __name__ == '__main__':
    main()
