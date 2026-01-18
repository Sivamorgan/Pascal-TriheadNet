from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision.transforms import v2
from torchvision import tv_tensors
import xml.etree.ElementTree as ET
import os
import hashlib
import numpy as np

class Pascal_VOCDataset(Dataset):
    """
    Args: data_root, split (train/val/trainval/test)
    Returns: Dictionary with image, boxes, labels, and masks (based on task GT's gets loaded)
    """
    
    VOC_CLASSES = (
        "background", "aeroplane", "bicycle", "bird", "boat", "bottle",
        "bus", "car", "cat", "chair", "cow", "diningtable", "dog",
        "horse", "motorbike", "person", "pottedplant", "sheep", "sofa",
        "train", "tvmonitor"
    )

    def __init__(self, data_root, split='train', image_set_subdir='Main', use_trainval_split=True):
        """
        Args: data_root, split (train/val/test), image_set_subdir (Main/Segmentation), use_trainval_split (bool)
        """
        super().__init__()
        self.split = split
        self.task='all'
        self.class_to_ind = dict(zip(self.VOC_CLASSES, range(len(self.VOC_CLASSES))))
        
        if split == 'test':
            self.root = os.path.join(data_root, 'VOC2012_test')
        else:
            self.root = os.path.join(data_root, 'VOC2012_train_val')
        
        # Create transforms for train and test
        if self.split == 'test' or self.split == 'val':
            self.transforms = self.test_transforms()
        else:
            self.transforms = self.train_transforms()
        
        # Standard splits are train, val, trainval, test
        # If use_trainval_split is True, we load 'trainval' and manually split it 80/20 to get more training data.
        load_split = split
        if use_trainval_split and split in ['train', 'val']:
            load_split = 'trainval'
            
        split_file = os.path.join(self.root, 'ImageSets', image_set_subdir, load_split + '.txt')
        self.image_set_subdir = image_set_subdir
        
        if not os.path.exists(split_file):
             if image_set_subdir != 'Main':
                 print(f"Warning: {image_set_subdir} split {load_split}.txt not found, falling back to Main.")
                 self.image_set_subdir = 'Main'
                 split_file = os.path.join(self.root, 'ImageSets', 'Main', load_split + '.txt')
                 
        if not os.path.exists(split_file):
             raise FileNotFoundError(f"Split file not found: {split_file}")
             
        with open(split_file, 'r') as file:
            all_ids = file.read().splitlines()
    
        if use_trainval_split:
            if split == 'train':
                self.ids = self._get_train_ids(all_ids)
            elif split == 'val':
                self.ids = self._get_val_ids(all_ids)
            else:
                self.ids = all_ids
        else:
            self.ids = all_ids
            
        print(f"Loaded {len(self.ids)} images from {self.image_set_subdir}/{split} (Source: {load_split})")
        
    def _get_train_ids(self, all_ids):
        """Get training IDs from trainval split (80% of data).
        Args: all_ids (list of image IDs)
        Returns: List of training image IDs
        """
        return [img_id for img_id in all_ids 
                if int(hashlib.md5(img_id.encode()).hexdigest(), 16) % 10 < 8]
    
    def _get_val_ids(self, all_ids):
        """Get validation IDs from trainval split (20% of data).
        Args: all_ids (list of image IDs)
        Returns: List of validation image IDs
        """
        return [img_id for img_id in all_ids 
                if int(hashlib.md5(img_id.encode()).hexdigest(), 16) % 10 >= 8]

    def test_transforms(self):
        """
        Returns standard test/validation transforms.
        """
        return v2.Compose([
                v2.ToImage(),
                v2.Resize((224, 224)),
                #v2.CenterCrop(224),
                #v2.SanitizeBoundingBoxes(),
                v2.ToDtype({
                    tv_tensors.Image: torch.float32,
                    tv_tensors.Mask: torch.int64,
                    "others": None
                }, scale=True),
                v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            
    def train_transforms(self):
        """
        Returns training transforms.
        """
        
        return v2.Compose([
                v2.ToImage(),
                v2.RandomResizedCrop(size=(224, 224), scale=(0.4, 1.0)),
                v2.RandomPhotometricDistort(p=0.6),
                v2.RandomHorizontalFlip(p=0.6),
                v2.RandomAffine(degrees=10, translate=(0.05, 0.05)),
                #v2.SanitizeBoundingBoxes(),
                v2.RandomApply([v2.ColorJitter(0.4,0.4,0.2,0.1)],p=0.8),
                v2.RandomGrayscale(p=0.2),
                v2.ToDtype({
                    tv_tensors.Image: torch.float32,
                    tv_tensors.Mask: torch.int64,
                    "others": None
                }, scale=True),
                v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
    
    def __len__(self):
        """Returns the length of the loader"""
        
        return len(self.ids)
    
    def __getitem__(self, index):
        """Load and transform image with annotations at given index
        
        Args: index
        Returns: Dictionary with image_id, image, and available annotations
        """
        img_id = self.ids[index]
        img_path = os.path.join(self.root, "JPEGImages", img_id + '.jpg')
        img = Image.open(img_path).convert("RGB")
        boxes = None
        labels = None
        sem_mask = None
        inst_mask = None
        is_segmented = False
    
        if self.task in ['detection', 'all', 'instance']:
            xml_path = os.path.join(self.root, "Annotations", img_id + '.xml')
            if os.path.exists(xml_path):
                boxes, labels, is_segmented_xml = self.parse_xml(xml_path)
                is_segmented = bool(is_segmented_xml)
        
        if self.task in ['semantic', 'instance', 'all']:
             if self.task in ['semantic', 'all']:
                sem_path = os.path.join(self.root, "SegmentationClass", img_id + '.png')
                if os.path.exists(sem_path):
                    sem_mask = Image.open(sem_path)
                    is_segmented = True
                else:
                    is_segmented=False
             
             if self.task in ['instance', 'all']:
                inst_path = os.path.join(self.root, "SegmentationObject", img_id + '.png')
                if os.path.exists(inst_path):
                    inst_mask = Image.open(inst_path)
                    is_segmented = True
                    inst_np = np.asarray(inst_mask)
                    unq_ids = np.unique(inst_np)
                    unq_ids = unq_ids[(unq_ids>0) & (unq_ids<255)]
                    candidate_masks = (inst_np ==unq_ids[:,None,None]).astype(np.uint8)
                    if boxes is not None and len(boxes) != len(candidate_masks):
                        boxes, labels, inst_mask = self._align_boxes_and_masks(boxes, labels, candidate_masks, inst_np)
                    else:
                        inst_mask = torch.from_numpy(candidate_masks)
                else:
                    is_segmented=False

        img, target = self.transform_image(img, boxes, sem_mask, inst_mask, labels)
        ret = {
            'image_id': img_id,
            'image': img,
            'is_segmented': is_segmented,
        }
        if "boxes" in target:
            ret['boxes'] = target['boxes']
            ret['labels'] = target['labels']
        if "sem_masks" in target:
            ret['semantic_mask'] = target['sem_masks']
        
        # Extract per-box 28x28 instance masks (crop to box, then resize)
        if is_segmented and "masks" in target and "boxes" in target and len(target['boxes']) > 0:
            from data.mask_utils import crop_and_resize_instance_masks
            instance_masks_28 = crop_and_resize_instance_masks(
                target['masks'],
                target['boxes'],
                mask_size=28
            )
            ret['instance_masks_28'] = instance_masks_28
            
        return ret
        
    def _align_boxes_and_masks(self, boxes, labels, candidate_masks, inst_np):
        """Match boxes to instance masks based on spatial overlap (IoU).
        Args:
                    boxes: torch.Tensor (N, 4) in [x1, y1, x2, y2] format
                    labels: torch.Tensor (N,) class labels
                    candidate_masks: torch.Tensor (M, H, W) binary masks from instance image
                    inst_np: np.ndarray (H, W) original instance ID image
            
                Returns:
                    boxes: torch.Tensor (K, 4) filtered boxes with matches
                    labels: torch.Tensor (K,) filtered labels
                    aligned_masks: torch.Tensor (K, H, W) aligned masks
        """
        num_boxes = len(boxes)
        num_masks = len(candidate_masks)
        H, W = inst_np.shape
        if num_boxes == 0 or num_masks == 0:
            return (
                torch.zeros(0, 4, dtype=boxes.dtype),
                torch.zeros(0, dtype=labels.dtype),
                torch.zeros(0, H, W, dtype=torch.uint8)
            )
    
        # Convert to numpy for processing
        boxes_np = boxes.cpu().numpy() if isinstance(boxes, torch.Tensor) else boxes
        labels_np = labels.cpu().numpy() if isinstance(labels, torch.Tensor) else labels
        masks_np = candidate_masks.cpu().numpy() if isinstance(candidate_masks, torch.Tensor) else candidate_masks
    
        aligned_masks = np.zeros((num_boxes, H, W), dtype=np.uint8)
        matched = np.zeros(num_masks, dtype=bool)
    
        for box_idx in range(num_boxes):
            box = boxes_np[box_idx]
            x1 = max(0, int(box[0]))
            y1 = max(0, int(box[1]))
            x2 = min(W, int(box[2]))
            y2 = min(H, int(box[3]))
        
            # Skip degenerate boxes
            if x2 <= x1 or y2 <= y1:
                continue
        
            best_iou = 0
            best_mask_idx = -1
            for mask_idx in range(num_masks):
                if matched[mask_idx]:
                    continue
                # ✅ Compute IoU: intersection over union
                mask = masks_np[mask_idx]
                mask_crop = mask[y1:y2, x1:x2]
                intersection = np.sum(mask_crop)
                mask_area = np.sum(mask)
                box_area = (x2 - x1) * (y2 - y1)
                union = mask_area + box_area - intersection
                iou = intersection / union if union > 0 else 0
            
                # ✅ Use lower threshold (0.3) or intersection over mask area
                if iou > best_iou and iou > 0.3:
                    best_iou = iou
                    best_mask_idx = mask_idx
            if best_mask_idx != -1:
                aligned_masks[box_idx] = masks_np[best_mask_idx]
                matched[best_mask_idx] = True
        keep = np.any(aligned_masks, axis=(1, 2))
    
        if not np.any(keep):
            return (
                torch.zeros(0, 4, dtype=boxes.dtype),
                torch.zeros(0, dtype=labels.dtype),
                torch.zeros(0, H, W, dtype=torch.uint8)
            )
    
        # Filter boxes, labels, masks
        boxes_filtered = torch.as_tensor(boxes_np[keep], dtype=torch.float32)
        labels_filtered = torch.as_tensor(labels_np[keep], dtype=torch.int64)
        aligned_masks = torch.from_numpy(aligned_masks[keep])
    
        return boxes_filtered, labels_filtered, aligned_masks
    
    def transform_image(self, image, boxes=None, sem_mask=None, inst_mask=None, labels=None):
        """Apply corresponding transformation to image and targets.
        
        Args: image, boxes, sem_masks, inst_masks, labels
        Returns: Transformed image tensor and target dictionary
        """
        #image = tv_tensors.Image(image)
        h,w = image.size[1],image.size[0]
        target = {}
        if boxes is not None and len(boxes)>0:
            target["boxes"] = tv_tensors.BoundingBoxes(boxes, format="XYXY", canvas_size=(h,w))
            if labels is not None:
                target["labels"] = torch.as_tensor(labels,dtype=torch.int64)
        if sem_mask is not None:
            target["sem_masks"] = tv_tensors.Mask(sem_mask)
        if inst_mask is not None:
            target["masks"] = tv_tensors.Mask(inst_mask)
        if target:
            out_image, out_target = self.transforms(image, target)
        else:
            out_image=self.transforms(image)
            out_target={}
        
        if "boxes" in out_target:
            
            #Sanitize Bounding boxes, labels & Instance masks if any
            sanitized_boxes, pack = v2.functional.sanitize_bounding_boxes(
                out_target["boxes"]
                )
            out_target["boxes"] = sanitized_boxes
            
            if "labels" in out_target:
                out_target["labels"] = out_target["labels"][pack]
            
            if "masks" in out_target:
                out_target["masks"] = out_target["masks"][pack]
        
        return out_image,out_target
            
            
            

    def parse_xml(self, xml_path):
        """Parse Pascal VOC XML annotation file and convert labels to tensors.
        
        Args: xml_path
        Returns: Tensors for boxes, labels and segmented flag (0 or 1)
        """
        tree = ET.parse(xml_path)
        root = tree.getroot()
        boxes = []
        labels = []
        # <segmented>1</segmented> means segmentation is available (as per my understanding)
        segmented_elem = root.find('segmented')
        is_segmented = int(segmented_elem.text) if segmented_elem is not None else 0
        
        for obj in root.findall('object'):
            label = obj.find('name').text
            bbox = obj.find('bndbox')
            xmin = int(bbox.find('xmin').text)
            xmax = int(bbox.find('xmax').text)
            ymin = int(bbox.find('ymin').text)
            ymax = int(bbox.find('ymax').text)
            
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(self.class_to_ind[label])
        
        return (
            torch.tensor(boxes, dtype=torch.float32) if boxes else torch.zeros(0, 4),
            torch.tensor(labels, dtype=torch.int64) if labels else torch.zeros(0, dtype=torch.int64),
            is_segmented
        )





class PascalUnifiedDataset(Pascal_VOCDataset):
    """
    Unified dataset for detection and segmentation.
    Loads both detection (boxes, labels) and segmentation (semantic, instance) 
    annotations. 
    """
    
    def __init__(self, data_root, split='train', subset='all', task='all', limit=None, use_trainval_split=False):
        """
        Args: 
            data_root: Path to VOC dataset
            split: train/val/test
            subset: 'all' (default) or 'segmentation' (only images with segmentation)
            task: 'all' (default), 'detection', 'semantic', 'instance'
            limit: Limit number of samples (for debugging)
            use_trainval_split: If True, uses 80/20 split of trainval for train/val.
        """
        
        use_seg_split = (subset == 'segmentation') or (task in ['semantic', 'instance'])
        subdir = 'Segmentation' if use_seg_split else 'Main'
        
        super().__init__(data_root, split, image_set_subdir=subdir, use_trainval_split=use_trainval_split)
        self.task = task
        
        if limit is not None:
            self.ids = self.ids[:limit]
            print(f"PascalUnifiedDataset: Limited to {len(self.ids)} images")
        
        self.segmented_indices = []
        self.unsegmented_indices = []
        
        if self.image_set_subdir == 'Segmentation':
            self.segmented_indices = list(range(len(self.ids)))
            self.unsegmented_indices = []
        else:
            for idx, img_id in enumerate(self.ids):
                sem_path = os.path.join(self.root, "SegmentationClass", img_id + '.png')
                is_seg = os.path.exists(sem_path)
                
                if is_seg:
                    self.segmented_indices.append(idx)
                else:
                    self.unsegmented_indices.append(idx)
    def __getitem__(self, index):
        """Get sample with all available annotations.
        
        Args: index
        Returns: Dictionary with image, detection, and segmentation data
        """
        sample = super().__getitem__(index)
        
        # Ensure all keys exist with defaults
        if 'boxes' not in sample:
            sample['boxes'] = torch.zeros(0, 4)
            sample['labels'] = torch.zeros(0, dtype=torch.long)
        if 'semantic_mask' not in sample:
            sample['semantic_mask'] = None
        if 'instance_masks_28' not in sample:
            sample['instance_masks_28'] = None
            
        return sample



class Dummy_Dataset(PascalUnifiedDataset):
    """Lightweight dataset limited to first 10 samples for testing and debugging.
    
    Args: data_root, split
    """
    
    def __init__(self, data_root, split='trainval', subset='all', task='all', limit=10):
        """Initialize and limit to 10 samples.
        
        Args: data_root, split
        """
        super().__init__(data_root, split=split, subset=subset, task=task, limit=limit)


def joint_collate_fn(batch):
    """Custom collate function for joint training.
    
    Returns lists with None for samples without segmentation ground truth.
    
    Args: batch - List of sample dicts
    Returns: Batched dictionary
    """
    
    images = torch.stack([sample['image'] for sample in batch])
    boxes = [sample['boxes'] for sample in batch]
    labels = [sample['labels'] for sample in batch]
    semantic_masks = []
    instance_masks_28 = []
    for sample in batch:
        sem_mask = sample.get('semantic_mask', None)
        if sem_mask is not None:
            if sem_mask.ndim == 2:
                sem_mask = sem_mask.unsqueeze(0)
        semantic_masks.append(sem_mask)
        instance_masks_28.append(sample.get('instance_masks_28', None))
    return {
        'images': images,
        'boxes': boxes,
        'labels': labels,
        'semantic_masks': semantic_masks,      
        'instance_masks_28': instance_masks_28, 
        'image_ids': [sample['image_id'] for sample in batch],
    }
