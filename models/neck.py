import torch.nn as nn

class ViTDetNeck(nn.Module):
    """ViTDet Simple Feature Pyramid.
    
    Creates multi-scale features in parallel from ViT's 1/16 resolution output.
    Standard scales: P2 (1/4), P3 (1/8), P4 (1/16), P5 (1/32)
    Uses bilinear upsampling + conv to avoid ConvTranspose2d artifacts.
    
    Args: in_channels (ViT hidden dim), out_channels (pyramid channels)
    Returns: Dict of multi-scale features {p2, p3, p4, p5}
    """
    
    def __init__(self,
                 in_channels: int = 768,
                 out_channels: int = 256):
        """Initialize ViTDet neck.
        
        Args: in_channels, out_channels
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # P5: 1/32 scale - downsample 2x from 1/16
        self.p5 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 
                     kernel_size=3, stride=2,padding=1),
            nn.GroupNorm(32, out_channels)
        )
        
        # P4: 1/16 scale - same as ViT output
        self.p4 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.GroupNorm(32, out_channels)
        )
        
        # P3: 1/8 scale - upsample 2x from 1/16
        self.p3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(in_channels, out_channels, kernel_size=3,padding=1),
            nn.GroupNorm(32, out_channels)
        )
        
        # P2: 1/4 scale - upsample 4x from 1/16
        self.p2 = nn.Sequential(
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False),
            nn.Conv2d(in_channels, out_channels, kernel_size=3,padding=1),
            nn.GroupNorm(32, out_channels)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.GroupNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """Generate multi-scale features in parallel.
        
        Args: x - (B, in_channels, H, W) at 1/16 resolution
        Returns: {p2: 1/4, p3: 1/8, p4: 1/16, p5: 1/32}
        """
        return {
            'p2': self.p2(x),  # 1/4  - finest
            'p3': self.p3(x),  # 1/8
            'p4': self.p4(x),  # 1/16 - base (ViT output)
            'p5': self.p5(x)   # 1/32 - coarsest
        }
