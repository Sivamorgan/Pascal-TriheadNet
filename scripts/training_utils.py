import torch
import math
from omegaconf import OmegaConf
import os 
from pathlib import Path

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

def create_optimizer(model, cfg):
    """Create optimizer with different LR for backbone."""
    # Filter backbone params
    backbone_params = [p for p in model.backbone.parameters() if p.requires_grad]
    
    other_params = (
        list(model.neck.parameters()) +
        list(model.det_head.parameters()) +
        list(model.sem_head.parameters()) +
        list(model.inst_head.parameters())
    )
    # Filter other params
    other_params = [p for p in other_params if p.requires_grad]
    
    param_groups = []
    if backbone_params:
        param_groups.append({'params': backbone_params, 'lr': cfg.base_lr * cfg.backbone_lr_mult})
        
    if other_params:
        param_groups.append({'params': other_params, 'lr': cfg.base_lr})
    
    optimizer = torch.optim.AdamW(param_groups, weight_decay=cfg.weight_decay)
    return optimizer


def create_scheduler(optimizer, cfg):
    """Create learning rate scheduler."""
    def lr_lambda(epoch):
        if epoch < cfg.warmup_epochs:
            return (epoch + 1) / cfg.warmup_epochs
        else:
            progress = (epoch - cfg.warmup_epochs) / (cfg.num_epochs - cfg.warmup_epochs)
            return 0.5 * (1.0 + math.cos(math.pi * progress))
    
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

def save_checkpoint(model, optimizer, scheduler, epoch, losses, cfg):
    """Save training checkpoint."""
    if cfg.training.get('dry_run', False):
        print("Dry run: Skipping checkpoint save.")
        return

    Path(cfg.training.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'losses': losses,
        'config': OmegaConf.to_container(cfg, resolve=True),
    }
    path = os.path.join(cfg.training.checkpoint_dir, f'checkpoint_epoch_{epoch}.pth')
    torch.save(checkpoint, path)
    print(f"✓ Saved checkpoint to {path}")