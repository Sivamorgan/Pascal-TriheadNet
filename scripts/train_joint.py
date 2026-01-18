import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import time
from pathlib import Path
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import hydra
from omegaconf import DictConfig, OmegaConf
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
from models.architectures import JointModel
from training_utils import create_optimizer, create_scheduler

class CheckpointManager:
    """Manages Top-K checkpoints based on a configurable metric.
    
    Args:
        save_dir: Directory to save checkpoints
        k: Number of top checkpoints to keep
        metric_name: Name of metric to track (e.g., 'val_total', 'det_map', 'sem_miou')
        minimize: If True, lower is better (for losses). If False, higher is better (for mAP, mIoU)
    """
    def __init__(self, save_dir, k=2, metric_name='val_total', minimize=True):
        self.save_dir = Path(save_dir)
        self.k = k
        self.metric_name = metric_name
        self.minimize = minimize
        self.top_k = []  # List of (score, epoch)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.best_score = float('inf') if minimize else float('-inf')
        print(f"📁 CheckpointManager: Tracking '{metric_name}' ({'lower' if minimize else 'higher'} is better)")

    def get_ckpt_path(self, epoch):
        return self.save_dir / f"checkpoint_epoch_{epoch}.pth"
    
    def get_score(self, val_losses: dict, val_metrics: dict) -> float:
        """Extract the tracked metric score from validation results.
        
        Args:
            val_losses: Dict with 'val_total', 'val_loss_det', 'val_loss_sem', 'val_loss_inst'
            val_metrics: Dict with 'det_map', 'sem_miou', 'sem_acc', 'inst_map', etc.
        
        Returns:
            The score for the tracked metric
        """
        # Check losses first, then metrics
        if self.metric_name in val_losses:
            return val_losses[self.metric_name]
        elif self.metric_name in val_metrics:
            return val_metrics[self.metric_name]
        else:
            raise KeyError(f"Metric '{self.metric_name}' not found in val_losses or val_metrics. "
                          f"Available: {list(val_losses.keys())} + {list(val_metrics.keys())}")

    def update(self, model, optimizer, scheduler, scaler, epoch, score, losses, cfg):
        path = self.get_ckpt_path(epoch)
        
        # Check if this is a 'best' model
        is_better = score < self.best_score if self.minimize else score > self.best_score
        if is_better:
            self.best_score = score
            print(f"⭐ New best model ({self.metric_name}): {score:.4f} (Epoch {epoch})")
        
        # Save current checkpoint
        self._save(path, model, optimizer, scheduler, scaler, epoch, losses, cfg)
        
        # Manage Top-K
        self.top_k.append((score, epoch))
        # Sort based on score
        self.top_k.sort(key=lambda x: x[0], reverse=not self.minimize)
        
        # Keep only top K
        if len(self.top_k) > self.k:
            to_remove = self.top_k.pop()  # Remove worst (score, epoch)
            remove_path = self.get_ckpt_path(to_remove[1])
            try:
                if remove_path.exists():
                    os.remove(remove_path)
                    print(f"🗑️  Removed checkpoint: Epoch {to_remove[1]} ({self.metric_name}: {to_remove[0]:.4f})")
            except OSError as e:
                print(f"Error removing checkpoint {remove_path}: {e}")
        print(f"📊 Top-{self.k} checkpoints ({self.metric_name}):")
        for i, (s, e) in enumerate(self.top_k, 1):
            marker = "⭐" if s == self.best_score else "  "
            print(f"   {marker} {i}. Epoch {e}: {s:.4f}")
        
    def _save(self, path, model, optimizer, scheduler, scaler, epoch, losses, cfg):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'scaler_state_dict': scaler.state_dict(),
            'losses': losses,
            'config': OmegaConf.to_container(cfg, resolve=True),
            'top_k_state': [(s, e) for s, e in self.top_k],
            'best_score': self.best_score,
            'metric_name': self.metric_name  # Save which metric was tracked
        }
        torch.save(checkpoint, path)
        print(f"✓ Saved checkpoint to {path}")

    def load_state(self, checkpoint):
        """Restore internal state from loaded checkpoint."""
        if 'top_k_state' in checkpoint:
            # Handle both old format (s, e, p) and new format (s, e)
            try:
                raw_state = checkpoint['top_k_state']
                if len(raw_state) > 0 and len(raw_state[0]) == 3:
                     self.top_k = [(s, e) for s, e, _ in raw_state]  # Drop path
                else:
                     self.top_k = [(s, e) for s, e in raw_state]
            except ValueError:
                print("Warning: Could not parse top_k_state, starting fresh.")
                self.top_k = []
        
        # Check if metric changed
        saved_metric = checkpoint.get('metric_name', 'val_total')
        if saved_metric != self.metric_name:
            print(f"⚠️  Metric changed from '{saved_metric}' to '{self.metric_name}'. Resetting best_score.")
            self.top_k = []
        else:
            print(f"Resuming checkpoint comparison for '{self.metric_name}'")


@hydra.main(version_base=None, config_path="../configs", config_name="joint_training")
def main(cfg: DictConfig):
    #print(OmegaConf.to_yaml(cfg))
    device = torch.device(cfg.training.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    from utils.reproducibility import set_seed, worker_init_fn, get_generator
    if 'seed' in cfg.training:
        set_seed(cfg.training.seed, cfg.training.get('deterministic', False))
        generator = get_generator(cfg.training.seed)
        worker_init = worker_init_fn
    else:
        generator = None
        worker_init = None
    
    Path(cfg.training.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    
    if cfg.wandb.enabled and WANDB_AVAILABLE:
        wandb.init(project=cfg.wandb.project, name=cfg.wandb.name, config=OmegaConf.to_container(cfg))
    
    from data.Dataset import PascalUnifiedDataset, joint_collate_fn
    from data.samplers import UnifiedTaskSampler
    
    train_dataset = PascalUnifiedDataset(
        data_root=cfg.data.root,
        split='train',
        subset='all',
        task='all',
        limit=cfg.data.limit,
        use_trainval_split=cfg.data.get('use_trainval_split', True)
    )
    
    batch_sampler = UnifiedTaskSampler(
        train_dataset, 
        batch_size=cfg.data.batch_size,
        segmented_ratio=cfg.data.get('segmented_ratio', 0.3), 
        shuffle=True
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_sampler=batch_sampler,
        num_workers=cfg.data.num_workers,
        collate_fn=joint_collate_fn,
        pin_memory=True,
        worker_init_fn=worker_init,
        generator=generator
    )
    
    model = JointModel(cfg).to(device)
    unfreeze_n = cfg.model.get('unfreeze_n_layers',0)
    if unfreeze_n>0:
      model.backbone.unfreeze(unfreeze_n)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    optimizer = create_optimizer(model, cfg.training)
    scheduler = create_scheduler(optimizer, cfg.training)
    
    # Mixed precision scaler for faster training
    scaler = torch.amp.GradScaler(device=cfg.training.device,enabled=cfg.training.get('use_amp', True))
    
    from losses import JointTrainingLoss
    loss_fn = JointTrainingLoss(
        det_weight=cfg.loss.det_weight,
        sem_weight=cfg.loss.sem_weight,
        inst_weight=cfg.loss.inst_weight,
        boundary_weight=cfg.loss.get('boundary_weight', 2.0),
        ignore_index=255 
    ).to(device)
    
    # Validation Setup
    from utils.metrics import MetricLogger
    metric_logger = MetricLogger(device)
    
    # Validation Dataset 
    val_dataset = PascalUnifiedDataset(
        data_root=cfg.data.root,
        split='val', 
        subset='all', 
        task='all',
        limit=cfg.data.limit,
        use_trainval_split=cfg.data.get('use_trainval_split', True)
    )
    # Use standard sequential sampling for Validation (unbiased evaluation)
    val_loader = DataLoader(
        val_dataset, 
        batch_size=cfg.data.batch_size,
        shuffle=False, 
        num_workers=cfg.data.num_workers,
        collate_fn=joint_collate_fn,
        worker_init_fn=worker_init,
        generator=generator
    )
    
    # Checkpoint Manager - configurable metric
    ckpt_metric = cfg.training.get('checkpoint_metric', 'val_total')
    ckpt_minimize = cfg.training.get('checkpoint_minimize', True)
    ckpt_manager = CheckpointManager(
        save_dir=cfg.training.checkpoint_dir, 
        k=cfg.training.get('checkpoint_k', 2),
        metric_name=ckpt_metric,
        minimize=ckpt_minimize
    )
    
    start_epoch = 0
    if cfg.training.resume:
        print(f"Resuming from {cfg.training.resume}...")
        ckpt = torch.load(cfg.training.resume, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        try:
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            scheduler.load_state_dict(ckpt['scheduler_state_dict'])
            if 'scaler_state_dict' in ckpt:
                scaler.load_state_dict(ckpt['scaler_state_dict'])
            print("All states Loaded")
        except:
            print("Using fresh optimizer (no.of parameters changed)")
        start_epoch = ckpt['epoch'] # Resume from next epoch (ckpt['epoch'] is 1-based completed count)
        # Restore Top-K state and best score
        ckpt_manager.load_state(ckpt)
        print(f"Restored state. Best score so far: {ckpt_manager.best_score:.4f}")
    
    for epoch in range(start_epoch, cfg.training.num_epochs):
        epoch_start = time.time()
        
        avg_losses = train_epoch(model, train_loader, optimizer, loss_fn, scaler, epoch+1, cfg, device)
        
        # Capture LR before step
        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step()
        epoch_time = time.time() - epoch_start
        
        print(f"Epoch {epoch+1}: Loss {avg_losses['total']:.4f} LR {current_lr:.6f} Time {epoch_time:.1f}s")
        
        # Validation
        val_total_loss = None
        
        if (epoch + 1) % cfg.training.val_freq == 0:
            val_metrics, val_losses = validate_epoch(model, val_loader, metric_logger, loss_fn, epoch+1, cfg, device)
            val_total_loss = val_losses['val_total']
            
            print(f"   Validation: Loss={val_total_loss:.4f}, mAP={val_metrics['det_map']:.3f}, mIoU={val_metrics['sem_miou']:.3f}, Acc={val_metrics['sem_acc']:.3f}, InstAP={val_metrics.get('inst_map', 0.0):.3f}")
            
            # Log to WandB
            if cfg.wandb.enabled and WANDB_AVAILABLE:
                wandb.log({**avg_losses, **val_losses, **val_metrics, 'epoch': epoch+1, 'lr': current_lr})
        else:
             if cfg.wandb.enabled and WANDB_AVAILABLE:
                wandb.log({**avg_losses, 'epoch': epoch+1, 'lr': current_lr})
        
        # Save Checkpoint (Top K based on configured metric)
        if (epoch + 1) % cfg.training.save_freq == 0:
            if val_total_loss is not None:
                score = ckpt_manager.get_score(val_losses, val_metrics)
                ckpt_manager.update(model, optimizer, scheduler, scaler, epoch + 1, score, avg_losses, cfg)
            else:
                # Skip saving if validation didn't run (to avoid mixing scores)
                print(f"Skipping checkpoint save (Epoch {epoch+1}): Validation was not run, so performance is unknown.")

def train_epoch(model, dataloader, optimizer, loss_fn, scaler, epoch, cfg, device):
    model.train()
    total_losses = {
        'total': 0.0, 
        'loss_det': 0.0,
        'loss_sem': 0.0, 
        'loss_inst': 0.0,
    }
    num_batches = len(dataloader)
    num_inst_batches = 0
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch}/{cfg.training.num_epochs}')
    
    for batch in pbar:
        images = batch['images'].to(device)
        boxes = [b.to(device) for b in batch['boxes']]
        labels = [l.to(device) for l in batch['labels']]
        semantic_masks = batch['semantic_masks']  # List[(1, H, W) or None]
        instance_masks_28 = batch['instance_masks_28']  # List[(N, 1, 28, 28) or None]
        with torch.amp.autocast(device_type=cfg.training.device,enabled=scaler.is_enabled()):
            outputs = model(images, boxes, labels)
        
        total_loss = 0.0
        loss_det=0.0
        loss_sem = 0.0
        loss_inst = 0.0
        losses = {}
        
        # Detection
        det_targets = {'gt_boxes': boxes, 'gt_labels': labels}
        det_losses = loss_fn.det_loss(outputs['detection'], det_targets)
        loss_det = cfg.loss.det_weight * det_losses['total']
        losses['loss_det'] = loss_det
        total_loss += loss_det
        
        # Sem + Inst - only for samples with segmentation GT
        sem_pred = outputs['semantic']
        valid_seg_indices = [i for i, m in enumerate(semantic_masks) if m is not None]
        
        if len(valid_seg_indices) > 0:
            # Semantic Loss
            valid_sem_masks = [semantic_masks[i].to(device) for i in valid_seg_indices]
            sem_gt = torch.cat(valid_sem_masks, dim=0)  # (N_valid, H, W)
            sem_pred_valid = sem_pred[valid_seg_indices]  # (N_valid, C, H, W)
            
            if sem_pred_valid.shape[-2:] != sem_gt.shape[-2:]:
                sem_pred_valid = F.interpolate(sem_pred_valid, size=sem_gt.shape[-2:], mode='bilinear', align_corners=False)
            
            sem_losses = loss_fn.sem_loss(sem_pred_valid, sem_gt.squeeze(1))
            loss_sem = cfg.loss.sem_weight * sem_losses['total']
            total_loss += loss_sem
            
            # Instance 
            if outputs['instance'] is not None:
                inst_pred = outputs['instance']  # (N_total_boxes, 1, 28, 28)
                
                gt_per_box_list = []
                pred_indices = []
                box_offset = 0
                
                for i in range(len(boxes)):
                    num_boxes = len(boxes[i])
                    
                    if i in valid_seg_indices and num_boxes > 0:
                        gt_per_box_list.append(instance_masks_28[i].to(device))
                        pred_indices.extend(range(box_offset, box_offset + num_boxes))
                    
                    box_offset += num_boxes
                
                if len(gt_per_box_list) > 0:
                    gt_all = torch.cat(gt_per_box_list, dim=0)
                    pred_indices_tensor = torch.tensor(pred_indices, device=device, dtype=torch.long)
                    inst_pred_filtered = inst_pred[pred_indices_tensor]
                    
                    inst_losses = loss_fn.inst_loss(inst_pred_filtered, gt_all)
                    loss_inst = cfg.loss.inst_weight * inst_losses['total']
                    total_loss += loss_inst
                    num_inst_batches += 1
        losses['loss_sem'] = loss_sem
        losses['loss_inst'] = loss_inst
        if isinstance(total_loss, torch.Tensor) and total_loss.requires_grad:
            losses['total'] = total_loss # For logging
            optimizer.zero_grad()
            scaler.scale(total_loss).backward()
            scaler.unscale_(optimizer)  # Unscale before clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            losses['total'] = total_loss if isinstance(total_loss, torch.Tensor) else torch.tensor(total_loss, device=device)
            optimizer.zero_grad() # Just in case
        
        for k in losses:
            if k in total_losses: 
                if isinstance(losses[k], torch.Tensor):
                    total_losses[k] += losses[k].item()
                else:
                    total_losses[k] += losses[k]
        
        pbar.set_postfix({'loss': f"{losses['total'].item():.4f}"})
        
    avg = {k: v / num_batches for k, v in total_losses.items()}
    return avg



@torch.no_grad()
def validate_epoch(model, dataloader, metric_logger, loss_fn, epoch, cfg, device):
    """Run validation and compute metrics."""
    from utils.postprocess import postprocess_detection
    
    model.eval()
    metric_logger.reset() 
    
    total_losses = {
        'val_total': 0.0, 
        'val_loss_det': 0.0,
        'val_loss_sem': 0.0, 
        'val_loss_inst': 0.0
    }
    num_batches = len(dataloader)
    
    pbar = tqdm(dataloader, desc=f'Val Epoch {epoch}')
    
    for batch in pbar:
        images = batch['images'].to(device)
        boxes = [b.to(device) for b in batch['boxes']]
        labels = [l.to(device) for l in batch['labels']]
        semantic_masks = batch['semantic_masks']  # List[(1, H, W) or None]
        instance_masks_28 = batch['instance_masks_28']  # List[(N, 1, 28, 28) or None]
        features = model.backbone.forward_features(images)
        pyramid = model.neck(features)
        
        det_out = model.det_head(pyramid)
        sem_out = model.sem_head(pyramid)
        
        # --- Loss Calculation ---
        val_loss_total = 0.0
        val_loss_det = 0.0
        val_loss_sem = 0.0
        val_loss_inst = 0.0
        
        # Detection
        det_targets_loss = {'gt_boxes': boxes, 'gt_labels': labels}
        det_losses = loss_fn.det_loss(det_out, det_targets_loss)
        val_loss_det = cfg.loss.det_weight * det_losses['total']
        total_losses['val_loss_det'] += val_loss_det.item()
        val_loss_total += val_loss_det.item()
        valid_seg_indices = [i for i, m in enumerate(semantic_masks) if m is not None]
        
        if len(valid_seg_indices) > 0:
            # Semantic 
            valid_sem_masks = [semantic_masks[i].to(device) for i in valid_seg_indices]
            sem_gt = torch.cat(valid_sem_masks, dim=0)
            sem_pred_valid = sem_out[valid_seg_indices]
            
            if sem_pred_valid.shape[-2:] != sem_gt.shape[-2:]:
                sem_pred_valid = F.interpolate(sem_pred_valid, size=sem_gt.shape[-2:], mode='bilinear', align_corners=False)
            
            sem_losses = loss_fn.sem_loss(sem_pred_valid, sem_gt.squeeze(1))
            val_loss_sem = cfg.loss.sem_weight * sem_losses['total']
            total_losses['val_loss_sem'] += val_loss_sem.item()
            val_loss_total += val_loss_sem.item()
            
            # Instance 
            inst_out = model.inst_head(pyramid, boxes, labels)
            if inst_out is not None:
                gt_per_box_list = []
                pred_indices = []
                box_offset = 0
                
                for i in range(len(boxes)):
                    num_boxes = len(boxes[i])
                    
                    if i in valid_seg_indices and num_boxes > 0:
                        gt_per_box_list.append(instance_masks_28[i].to(device))
                        pred_indices.extend(range(box_offset, box_offset + num_boxes))
                    
                    box_offset += num_boxes
                
                if len(gt_per_box_list) > 0:
                    gt_all = torch.cat(gt_per_box_list, dim=0)
                    pred_indices_tensor = torch.tensor(pred_indices, device=device, dtype=torch.long)
                    inst_out_filtered = inst_out[pred_indices_tensor]
                    
                    inst_losses = loss_fn.inst_loss(inst_out_filtered, gt_all)
                    val_loss_inst = cfg.loss.inst_weight * inst_losses['total']
                    total_losses['val_loss_inst'] += val_loss_inst.item()
                    val_loss_total += val_loss_inst.item()
        
        total_losses['val_total'] += val_loss_total
        final_dets = postprocess_detection(det_out) 
        det_targets_metric = [
            {'boxes': b, 'labels': l} 
            for b, l in zip(boxes, labels)
        ]
        metric_logger.update_detection(final_dets, det_targets_metric)
        if len(valid_seg_indices) > 0:
            valid_sem_masks = [semantic_masks[i].to(device) for i in valid_seg_indices]
            sem_gt = torch.cat(valid_sem_masks, dim=0)
            sem_pred_valid = sem_out[valid_seg_indices]
            
            if sem_pred_valid.shape[-2:] != sem_gt.shape[-2:]:
                sem_pred_valid = F.interpolate(sem_pred_valid, size=sem_gt.shape[-2:], mode='bilinear', align_corners=False)
            
            metric_logger.update_semantic(sem_pred_valid, sem_gt.squeeze(1))
        if inst_out is not None and len(valid_seg_indices) > 0:
            H, W = images.shape[-2:]
            
            inst_preds_list = []
            inst_targets_list = []
            
            box_offset = 0
            for i in range(len(boxes)):
                num_boxes = len(boxes[i])
                
                if i in valid_seg_indices and num_boxes > 0:
                    pred_masks_28 = inst_out[box_offset:box_offset + num_boxes]  # (N, 1, 28, 28)
                    # Resize to full resolution and threshold
                    pred_masks_full = F.interpolate(pred_masks_28, size=(H, W), mode='bilinear', align_corners=False)
                    pred_masks_bool = (pred_masks_full.squeeze(1).sigmoid() > 0.5)  # (N, H, W)
                    gt_masks_28 = instance_masks_28[i].to(device)  # (N, 1, 28, 28)
                    gt_masks_full = F.interpolate(gt_masks_28, size=(H, W), mode='bilinear', align_corners=False)
                    gt_masks_bool = (gt_masks_full.squeeze(1) > 0.5)  # (N, H, W)
                    inst_preds_list.append({
                        'masks': pred_masks_bool,
                        'scores': torch.ones(num_boxes, device=device), 
                        'labels': labels[i]
                    })
                    inst_targets_list.append({
                        'masks': gt_masks_bool,
                        'labels': labels[i]
                    })
                
                box_offset += num_boxes
            
            if len(inst_preds_list) > 0:
                metric_logger.update_instance(inst_preds_list, inst_targets_list)

    metrics = metric_logger.compute()
    avg_losses = {k: v / num_batches for k, v in total_losses.items()}
    return metrics, avg_losses

if __name__ == '__main__':
    main()
