import torch
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torchmetrics.classification import MulticlassJaccardIndex, Accuracy

class MetricLogger:
    """Wrapper for torchmetrics to handle detection and segmentation evaluation.
    
    Handles device movement and metric accumulation/computation.
    """
    
    def __init__(self, device, num_classes: int = 21):
        self.device = device
        self.num_classes = num_classes
        self.enabled = MeanAveragePrecision is not None
        
        if not self.enabled:
            return

        # 1. Detection Metrics (mAP)
        self.det_map = MeanAveragePrecision(box_format='xyxy', class_metrics=False).to(device)
        
        # 2. Semantic Segmentation Metrics
        # num_classes = 21 (0=background, 1..20=objects)
        self.sem_miou = MulticlassJaccardIndex(num_classes=num_classes, ignore_index=255).to(device)
        self.sem_acc = Accuracy(task="multiclass", num_classes=num_classes, ignore_index=255).to(device)
        
        # 3. Instance Segmentation Metrics
        self.inst_map = MeanAveragePrecision(box_format='xyxy', iou_type="segm", class_metrics=False).to(device)

    def update_detection(self, preds, targets):
        """Update detection metrics.
        
        Args:
            preds: List of dicts with 'boxes', 'scores', 'labels'
            targets: List of dicts with 'boxes', 'labels'
        """
        if not self.enabled: return
        self.det_map.update(preds, targets)

    def update_semantic(self, preds, target):
        """Update semantic segmentation metrics.
        
        Args:
            preds: (N, C, H, W) logits or (N, H, W) predictions
            target: (N, H, W) gt labels
        """
        if not self.enabled: return
        if preds.dim() == 4: # Is Logits (N, C, H, W)
            preds = torch.argmax(preds, dim=1)
        
        self.sem_miou.update(preds, target)
        self.sem_acc.update(preds, target)

    def update_instance(self, preds, targets):
        """Update instance segmentation metrics.
        
        Args:
            preds: List of dicts with 'masks', 'scores', 'labels'
            targets: List of dicts with 'masks', 'labels'
        """
        if not self.enabled: return
        self.inst_map.update(preds, targets)

    def compute(self):
        """Compute all metrics and reset."""
        if not self.enabled: 
            return {
                'det_map': 0.0, 'det_map_50': 0.0, 'det_map_75': 0.0,
                'sem_miou': 0.0, 'sem_acc': 0.0,
                'inst_map': 0.0, 'inst_map_50': 0.0
            }
        
        results = {}
        
        # Detection
        try:
            det_res = self.det_map.compute()
            results['det_map'] = det_res['map'].item()
            results['det_map_50'] = det_res['map_50'].item()
            results['det_map_75'] = det_res['map_75'].item()
            self.det_map.reset()
        except Exception as e:
            print(f"Warning: Failed to compute detection metrics: {e}")
            results['det_map'] = 0.0

        # Semantic
        try:
            results['sem_miou'] = self.sem_miou.compute().item()
            results['sem_acc'] = self.sem_acc.compute().item()
            self.sem_miou.reset()
            self.sem_acc.reset()
        except Exception as e:
            print(f"Warning: Failed to compute semantic metrics: {e}")
            results['sem_miou'] = 0.0

        # Instance
        try:
            # Instance mAP might fail if no masks present
            inst_res = self.inst_map.compute()
            results['inst_map'] = inst_res['map'].item()
            results['inst_map_50'] = inst_res['map_50'].item()
            self.inst_map.reset()
        except Exception as e:
            # Common error if no predictions made
            print(f"Warning: Failed to compute instance metrics: {e}")
            results['inst_map'] = 0.0
            
        return results

    def reset(self):
        """Reset all metrics."""
        if not self.enabled: return
        self.det_map.reset()
        self.sem_miou.reset()
        self.sem_acc.reset()
        self.inst_map.reset()
