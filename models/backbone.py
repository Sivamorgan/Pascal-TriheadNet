import timm
import torch.nn as nn

class Encoder(nn.Module):
    """Vision Transformer encoder for detection and segmentation tasks.
    
    Args: model_name, pretrained, img_size, freeze_backbone
    Returns: Feature embeddings from ViT backbone
    """
    
    def __init__(self, 
                 model_name='vit_base_patch16_224',
                 pretrained=True,
                 img_size=224,
                 freeze_backbone=False):
        """
        Args: model_name, pretrained, img_size, freeze_backbone
        """
        super().__init__()
        
        # Create ViT backbone (num_classes=0 to exclude classification head)
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            img_size=img_size,
            num_classes=0, 
            dynamic_img_size=True
        )
        
        # Store feature dimension for downstream tasks
        self.feature_dim = self.backbone.num_features
        
        # Freeze backbone if requested (useful for initial training)
        if freeze_backbone:
            self._make_trainable(self.backbone, grad=False)
    
    def forward(self, x):
        """Extract feature embeddings from input images.
        
        Args: x (B, C, H, W)
        Returns: Feature embeddings (B, feature_dim)
        """
        return self.backbone(x)
    
    def forward_features(self, x):
        """Extract spatial patch features (for detection/segmentation).
        
        Args: x (B, C, H, W)
        Returns: Spatial features (B, feature_dim, H/16, W/16)
        """
        # Get patch embeddings before global pooling
        x = self.backbone.forward_features(x)
        # Remove CLS token and reshape to spatial
        B, N, D = x.shape
        num_patches = N - 1
        H = W = int(num_patches ** 0.5)  # Assuming square patches
        x = x[:, 1:, :]  # (B, num_patches, D)
        x = x.transpose(1, 2).reshape(B, D, H, W)  # (B, D, H, W)
        return x
    
    @staticmethod
    def _make_trainable(module, grad: bool):
        """Set the requires_grad attribute of all parameters in a module.
        
        Args: module, grad
        """
        for param in module.parameters():
            param.requires_grad = grad
    
    def unfreeze(self, num_layers=None):
        """Unfreeze backbone parameters for fine-tuning.
        
        Args: num_layers (int, optional) - Number of last transformer blocks to unfreeze.
                                          If None, unfreezes all parameters.
        """
        if num_layers is None:
            # Unfreeze everything
            self._make_trainable(self.backbone, grad=True)
        else:
            # Unfreeze only the last num_layers transformer blocks
            total_blocks = len(self.backbone.blocks)
            assert 0 <= num_layers <= total_blocks, f"num_layers must be between 0 and {total_blocks}"
            for block in self.backbone.blocks[-num_layers:]:
                self._make_trainable(block, grad=True)
            if hasattr(self.backbone, 'norm'):
                self._make_trainable(self.backbone.norm, grad=True)
