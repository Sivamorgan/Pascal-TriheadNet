import torch.nn as nn
from models.backbone import Encoder
from models.neck import ViTDetNeck
from models.head import DetectionHead, SemanticHead, InstanceHead

class JointModel(nn.Module):
    """Joint model with shared backbone/neck and task-specific heads."""
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.backbone = Encoder(
            model_name=cfg.model.backbone, 
            pretrained=cfg.model.pretrained,
            freeze_backbone=cfg.model.freeze_backbone
        )
        self.neck = ViTDetNeck(in_channels=self.backbone.feature_dim, out_channels=cfg.model.neck_channels)
        self.det_head = DetectionHead(in_channels=cfg.model.neck_channels, num_classes=20)
        self.sem_head = SemanticHead(in_channels=cfg.model.neck_channels, num_classes=21)
        self.inst_head = InstanceHead(in_channels=cfg.model.neck_channels, num_classes=20)
    
    def forward(self, images, boxes=None, labels=None):
        features = self.backbone.forward_features(images)
        pyramid = self.neck(features)
        det_out = self.det_head(pyramid)
        sem_out = self.sem_head(pyramid)
        inst_out = self.inst_head(pyramid, boxes, labels) if boxes is not None and labels is not None else None
        return {'detection': det_out, 'semantic': sem_out, 'instance': inst_out, 'pyramid': pyramid}

class DetectionModel(nn.Module):
    """Detection-only model."""
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.backbone = Encoder(
            model_name=cfg.model.backbone, 
            pretrained=cfg.model.pretrained,
            freeze_backbone=cfg.model.freeze_backbone
        )
        self.neck = ViTDetNeck(in_channels=self.backbone.feature_dim, out_channels=cfg.model.neck_channels)
        self.det_head = DetectionHead(in_channels=cfg.model.neck_channels, num_classes=20)
    
    def forward(self, images):
        features = self.backbone.forward_features(images)
        pyramid = self.neck(features)
        return self.det_head(pyramid)

class SemanticModel(nn.Module):
    """Semantic Segmentation-only model."""
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.backbone = Encoder(
            model_name=cfg.model.backbone, 
            pretrained=cfg.model.pretrained,
            freeze_backbone=cfg.model.freeze_backbone
        )
        self.neck = ViTDetNeck(in_channels=self.backbone.feature_dim, out_channels=cfg.model.neck_channels)
        self.sem_head = SemanticHead(in_channels=cfg.model.neck_channels, num_classes=21)
    
    def forward(self, images):
        features = self.backbone.forward_features(images)
        pyramid = self.neck(features)
        return self.sem_head(pyramid)

class InstanceModel(nn.Module):
    """Instance Segmentation-only model."""
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.backbone = Encoder(
            model_name=cfg.model.backbone, 
            pretrained=cfg.model.pretrained,
            freeze_backbone=cfg.model.freeze_backbone
        )
        self.neck = ViTDetNeck(in_channels=self.backbone.feature_dim, out_channels=cfg.model.neck_channels)
        self.inst_head = InstanceHead(in_channels=cfg.model.neck_channels, num_classes=20)
    
    def forward(self, images, boxes=None, labels=None):
        features = self.backbone.forward_features(images)
        pyramid = self.neck(features)
        if boxes is not None and labels is not None:
            return self.inst_head(pyramid, boxes, labels)
        return None
