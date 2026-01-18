import torch
import torch.nn as nn

class DetectionHead(nn.Module):
    """FCOS-style anchor-free detection head.
    
    Predicts bounding boxes and classes directly from feature pyramid.
    Each pyramid level predicts objects at that scale.
    
    Args: in_channels, num_classes, num_shared_convs
    Returns: Dict with classification, bbox regression, and centerness predictions
    """
    
    def __init__(self,
                 in_channels: int = 256,
                 num_classes: int = 20,
                 num_shared_convs: int = 4):
        """Initialize FCOS detection head.
        
        Args: in_channels (from neck), num_classes (VOC=20), num_shared_convs
        """
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.num_shared_convs = num_shared_convs
        
        # Shared convolutions for all pyramid levels
        self.shared_convs = nn.ModuleList()
        for i in range(num_shared_convs):
            self.shared_convs.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
                    nn.GroupNorm(32, in_channels),
                    nn.ReLU(inplace=True)
                )
            )
        
        # Classification head (predicts class probabilities)
        self.cls_head = nn.Conv2d(in_channels, num_classes, kernel_size=3, padding=1)
        
        # Box regression head (predicts LTRB: left, top, right, bottom)
        self.bbox_head = nn.Conv2d(in_channels, 4, kernel_size=3, padding=1)
        
        # Centerness head (predicts quality/confidence score)
        self.centerness_head = nn.Conv2d(in_channels, 1, kernel_size=3, padding=1)
        
        # Scales for each pyramid level (learnable)
        self.scales = nn.ParameterList([nn.Parameter(torch.ones(1)) for _ in range(4)])
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with proper bias for classification."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean=0, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.GroupNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
        # Bias initialization for classification (helps with class imbalance)
        prior_prob = 0.01
        bias_value = -torch.log(torch.tensor((1 - prior_prob) / prior_prob))
        nn.init.constant_(self.cls_head.bias, bias_value)
    
    def forward(self, features):
        """Forward pass through detection head.
        
        Args: features - Dict from neck {p2, p3, p4, p5}
        Returns: Dict with 'cls', 'bbox', 'centerness' predictions per level
        """
        cls_scores = []
        bbox_preds = []
        centerness_preds = []
        
        for i, (level, feat) in enumerate(sorted(features.items())):
            x = feat
            for shared_conv in self.shared_convs:
                x = shared_conv(x)
            
            # Classification (sigmoid activation for multi-label)
            cls_score = self.cls_head(x)  # (B, 20, H, W)
            cls_scores.append(cls_score)
            
            # Bbox regression (scaled by learnable scale parameter)
            bbox_pred = self.bbox_head(x) * self.scales[i]  # (B, 4, H, W)
            bbox_preds.append(torch.exp(bbox_pred))  # Always positive
            
            # Centerness
            centerness = self.centerness_head(x)  # (B, 1, H, W)
            centerness_preds.append(centerness)
        
        return {
            'cls': cls_scores,        # List of (B, 20, H, W) per level
            'bbox': bbox_preds,       # List of (B, 4, H, W) per level
            'centerness': centerness_preds  # List of (B, 1, H, W) per level
        }


class SemanticHead(nn.Module):
    """Semantic segmentation head with progressive upsampling (Panoptic FPN).
    """
    
    def __init__(self,
                 in_channels: int = 256,
                 num_classes: int = 21):
        """Initialize semantic segmentation head.
        
        Args: in_channels (from neck), num_classes (VOC=21, including background)
        """
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        
        # Lateral convs - keep same channels for element-wise sum
        self.lateral_p5 = nn.Conv2d(in_channels, in_channels // 2, kernel_size=1)
        self.lateral_p4 = nn.Conv2d(in_channels, in_channels // 2, kernel_size=1)
        self.lateral_p3 = nn.Conv2d(in_channels, in_channels // 2, kernel_size=1)
        self.lateral_p2 = nn.Conv2d(in_channels, in_channels // 2, kernel_size=1)
    
        # P5: 3 stages (1/32 → 1/16 → 1/8 → 1/4)
        self.p5_upsample_1 = self._make_upsample_module(in_channels // 2)
        self.p5_upsample_2 = self._make_upsample_module(in_channels // 2)
        self.p5_upsample_3 = self._make_upsample_module(in_channels // 2)
        
        # P4: 2 stages (1/16 → 1/8 → 1/4)
        self.p4_upsample_1 = self._make_upsample_module(in_channels // 2)
        self.p4_upsample_2 = self._make_upsample_module(in_channels // 2)
        
        # P3: 1 stage (1/8 → 1/4)
        self.p3_upsample_1 = self._make_upsample_module(in_channels // 2)
        
        self.fusion = nn.Sequential(
            nn.Conv2d(in_channels // 2, in_channels // 2, kernel_size=3, padding=1),
            nn.GroupNorm(32, in_channels // 2),
            nn.ReLU(inplace=True)
        )
        
        # Final upsampling from P2 (1/4) to full resolution (1/1)
        self.final_upsample = nn.Sequential(
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False),
            nn.Conv2d(in_channels // 2, in_channels // 2, kernel_size=3, padding=1),
            nn.GroupNorm(32, in_channels // 2),
            nn.ReLU(inplace=True)
        )
        
        # Final classification layer
        self.classifier = nn.Conv2d(in_channels // 2, num_classes, kernel_size=1)
        
        self._init_weights()
    
    def _make_upsample_module(self, channels: int):
        """Create a 2x upsampling module.
        
        Args: channels
        Returns: Sequential module with Convolution + Group Norm + 
        """
        return nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.GroupNorm(32, channels),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        )
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.GroupNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, features):
        """
        Args: features - Dict from neck {p2, p3, p4, p5}
        Returns: Semantic logits (B, num_classes, H, W) at full resolution
        """
        # Process each level with lateral conv
        p5_lat = self.lateral_p5(features['p5'])  # (B, C//2, H/32, W/32)
        p4_lat = self.lateral_p4(features['p4'])  # (B, C//2, H/16, W/16)
        p3_lat = self.lateral_p3(features['p3'])  # (B, C//2, H/8, W/8)
        p2_lat = self.lateral_p2(features['p2'])  # (B, C//2, H/4, W/4)
        
        # P5: 3 stages (1/32 → 1/16 → 1/8 → 1/4)
        p5_to_p2 = self.p5_upsample_1(p5_lat)   # 1/32 → 1/16
        p5_to_p2 = self.p5_upsample_2(p5_to_p2)  # 1/16 → 1/8
        p5_to_p2 = self.p5_upsample_3(p5_to_p2)  # 1/8 → 1/4
        
        # P4: 2 stages (1/16 → 1/8 → 1/4)
        p4_to_p2 = self.p4_upsample_1(p4_lat)   # 1/16 → 1/8
        p4_to_p2 = self.p4_upsample_2(p4_to_p2)  # 1/8 → 1/4
        
        # P3: 1 stage (1/8 → 1/4)
        p3_to_p2 = self.p3_upsample_1(p3_lat)   # 1/8 → 1/4
        
        # P2: Already at 1/4, no upsampling needed
        p2_to_p2 = p2_lat
        
        # Element-wise SUM
        fused = p5_to_p2 + p4_to_p2 + p3_to_p2 + p2_to_p2
        
        x = self.fusion(fused)
        x = self.final_upsample(x)
        
        # Classification
        semantic_logits = self.classifier(x)
        
        return semantic_logits


class InstanceHead(nn.Module):
    """
    Instance segmentation head (Mask R-CNN style).
    
    """
    
    def __init__(self,
                 in_channels: int = 256,
                 num_classes: int = 20,
                 mask_resolution: int = 28,
                 num_convs: int = 4,
                 roi_size: int = 14):
        """
        Args:
            in_channels: Input channels from neck
            num_classes: Number of classes (VOC=20)
            mask_resolution: Output mask size (default 28x28)
            num_convs: Number of conv layers in mask head
            roi_size: RoI Align output size (default 14x14)
        """
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.mask_resolution = mask_resolution
        self.roi_size = roi_size
        
        # Multi-level feature pyramid
        self.feature_levels = ['p2', 'p3', 'p4', 'p5']
        self.spatial_scales = {
            'p2': 1 / 4,   # 1/4 resolution
            'p3': 1 / 8,   # 1/8 resolution
            'p4': 1 / 16,  # 1/16 resolution
            'p5': 1 / 32   # 1/32 resolution
        }
        from torchvision.ops import RoIAlign
        self.roi_aligns = nn.ModuleDict({
            level: RoIAlign(
                output_size=(roi_size, roi_size),
                spatial_scale=self.spatial_scales[level],
                sampling_ratio=2,
                aligned=True
            )
            for level in self.feature_levels
        })
        mask_layers = []
        for i in range(num_convs):
            mask_layers.extend([
                nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
                nn.GroupNorm(32, in_channels),
                nn.ReLU(inplace=True)
            ])
        
        self.mask_fcn = nn.Sequential(*mask_layers)
        
        # Upsample from roi_size (14x14) to mask_resolution (28x28)
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(in_channels, in_channels, kernel_size=2, stride=2),
            nn.ReLU(inplace=True)
        )
        
        # Final mask prediction - class-specific (20 channels for VOC)
        self.mask_predictor = nn.Conv2d(in_channels, num_classes, kernel_size=1)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights."""
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def _assign_boxes_to_levels(self, boxes):
        """Assign boxes to FPN levels based on box area.
        
        Args: 
            boxes - List of boxes per image (N_i, 4) in [x1, y1, x2, y2] format
        
        Returns: 
            assigned_boxes: Dict mapping level names to list of (batch_idx, boxes) tuples
            box_to_global_idx: Dict mapping level names to list of global indices
        """
        k_min, k_max = 2, 5   #p2,p5
        k0 = 4                #p4 is actual feature dim
        
        assigned_boxes = {level: [] for level in self.feature_levels}
        box_to_global_idx = {level: [] for level in self.feature_levels}
        
        global_idx = 0 
        
        for batch_idx, boxes_per_image in enumerate(boxes):
            if len(boxes_per_image) == 0:
                continue
            
            # Compute box areas
            widths = boxes_per_image[:, 2] - boxes_per_image[:, 0]
            heights = boxes_per_image[:, 3] - boxes_per_image[:, 1]
            areas = widths * heights
            
            # Assign to levels
            target_levels = torch.floor(k0 + torch.log2(torch.sqrt(areas) / 224))
            target_levels = torch.clamp(target_levels, k_min, k_max).long()
            
            # Create global indices for this batch
            batch_global_indices = torch.arange(
                global_idx, 
                global_idx + len(boxes_per_image),
                dtype=torch.long,
                device=boxes_per_image.device
            )
            
            for level_idx in range(k_min, k_max + 1):
                level_name = f'p{level_idx}'
                mask = target_levels == level_idx
                if mask.any():
                    # Store boxes and their global indices
                    assigned_boxes[level_name].append((batch_idx, boxes_per_image[mask]))
                    box_to_global_idx[level_name].append(batch_global_indices[mask])
            
            global_idx += len(boxes_per_image)
        
        return assigned_boxes, box_to_global_idx
    
    def forward(self, 
                features,
                boxes,
                labels= None):
        """Forward pass with multi-level RoI Align and class-specific masks.
        
        Args:
            features: Dict from neck {p2, p3, p4, p5}
            boxes: List of detected boxes per image, each (N_i, 4) 
                   in [x1, y1, x2, y2] format (absolute image coordinates, NOT normalized)
            labels: Optional list of class labels per box (N_i,) for class-specific mask selection
        
        Returns:
            masks: (N_total, num_classes, mask_resolution, mask_resolution)
                   or (N_total, 1, mask_resolution, mask_resolution) if labels provided
        """
        # Validate inputs
        if boxes is None or len(boxes) == 0:
            device = features['p2'].device
            return torch.zeros((0, self.num_classes, self.mask_resolution, self.mask_resolution), device=device)
        
        total_boxes = sum(len(b) for b in boxes)
        if total_boxes == 0:
            device = features['p2'].device
            return torch.zeros((0, self.num_classes, self.mask_resolution, self.mask_resolution), device=device)
        
        # Validate and clean box coordinates
        import warnings
        cleaned_boxes = []
        for batch_idx, boxes_per_image in enumerate(boxes):
            if len(boxes_per_image) == 0:
                cleaned_boxes.append(boxes_per_image)
                continue
            
            # Clip negative coordinates to 0
            if boxes_per_image.min() < 0:
                warnings.warn(f"Batch {batch_idx}: Found negative box coordinates, clipping to 0")
                boxes_per_image = boxes_per_image.clamp(min=0)
            
            # Fix invalid boxes where x2 <= x1 or y2 <= y1
            x1, y1, x2, y2 = boxes_per_image[:, 0], boxes_per_image[:, 1], boxes_per_image[:, 2], boxes_per_image[:, 3]
            invalid_x = x2 <= x1
            invalid_y = y2 <= y1
            
            if invalid_x.any() or invalid_y.any():
                warnings.warn(f"Batch {batch_idx}: Found {invalid_x.sum() + invalid_y.sum()} invalid boxes, fixing")
                # Add small epsilon to ensure x2 > x1 and y2 > y1
                boxes_per_image[:, 2] = torch.maximum(x2, x1 + 1.0)
                boxes_per_image[:, 3] = torch.maximum(y2, y1 + 1.0)
            
            cleaned_boxes.append(boxes_per_image)
        
        boxes = cleaned_boxes
        
        # Assign boxes to FPN levels based on box area
        assigned_boxes, box_to_global_idx = self._assign_boxes_to_levels(boxes)
        
        # Total boxes for output tensor
        total_boxes = sum(len(b) for b in boxes)
        
        # Process RoIs from each level, tracking global indices
        all_roi_features = []
        all_global_indices = []
        
        for level_name in self.feature_levels:
            level_box_list = assigned_boxes[level_name]
            level_idx_list = box_to_global_idx[level_name]
            
            if len(level_box_list) == 0:
                continue
            
            # Prepare boxes for RoI Align: (N, 5) [batch_idx, x1, y1, x2, y2]
            rois = []
            for (batch_idx, boxes_for_level), global_indices in zip(level_box_list, level_idx_list):
                all_global_indices.append(global_indices)
                
                # Prepare RoI tensor
                batch_indices = torch.full(
                    (len(boxes_for_level), 1),
                    batch_idx,
                    dtype=boxes_for_level.dtype,
                    device=boxes_for_level.device
                )
                rois.append(torch.cat([batch_indices, boxes_for_level], dim=1))
            
            if len(rois) == 0:
                continue
            
            rois_tensor = torch.cat(rois, dim=0)  # (N_level, 5)
            
            # Extract RoI features using level-specific RoIAlign
            feature_map = features[level_name]
            roi_features = self.roi_aligns[level_name](feature_map, rois_tensor)
            all_roi_features.append(roi_features)
        
        if len(all_roi_features) == 0:
            device = features['p2'].device
            return torch.zeros((0, self.num_classes, self.mask_resolution, self.mask_resolution), device=device)
        
        # Concatenate RoI features and global indices from all levels
        all_roi_features = torch.cat(all_roi_features, dim=0)  # (N_total, C, roi_size, roi_size)
        all_global_indices = torch.cat(all_global_indices, dim=0)  # (N_total,)
        
        # Process through mask FCN
        x = self.mask_fcn(all_roi_features)
        
        # Upsample to mask resolution
        x = self.deconv(x)  # (N_total, C, mask_resolution, mask_resolution)
        
        # Predict class-specific masks
        masks = self.mask_predictor(x)  # (N_total, num_classes, mask_resolution, mask_resolution)
        
        # Reorder masks to match original input box order
        reordered_masks = torch.zeros(
            (total_boxes, self.num_classes, self.mask_resolution, self.mask_resolution),
            device=masks.device,
            dtype=masks.dtype
        )
        reordered_masks[all_global_indices] = masks
        masks = reordered_masks
        
        # If labels provided, select only the relevant class mask for each instance
        if labels is not None:
            labels_flat = torch.cat(labels, dim=0)  # (N_total,)
            # VOC labels are 1-20 (object classes), but mask predictor expects 0-19
            labels_flat = (labels_flat - 1).clamp(0, self.num_classes - 1)
            N = masks.shape[0]
            masks = masks[torch.arange(N, device=masks.device), labels_flat].unsqueeze(1)
        
        return masks