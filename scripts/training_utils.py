import torch
import math
from omegaconf import OmegaConf
import os 
from pathlib import Path

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