import random
from torch.utils.data import Sampler

class UnifiedTaskSampler(Sampler):
    """Sampler that ensures a fixed ratio of segmented samples in each batch for stable training on all tasks.
    
    Args:
        dataset: Dataset 
        batch_size: Total batch size
        segmented_ratio: Fraction of batch that should be segmented (default 0.3)
        shuffle: Whether to shuffle indices (default True)
        drop_last: Whether to drop the last incomplete batch (default True to ensure ratio)
    """
    
    def __init__(self, dataset, batch_size, segmented_ratio=0.3, shuffle=True, drop_last=True):
        self.batch_size = batch_size
        self.segmented_ratio = segmented_ratio
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.num_seg = int(round(batch_size * segmented_ratio))
        self.num_unseg = batch_size - self.num_seg
    
        if hasattr(dataset, 'segmented_indices') and hasattr(dataset, 'unsegmented_indices'):
            self.seg_indices = dataset.segmented_indices
            self.unseg_indices = dataset.unsegmented_indices
        else:
            raise ValueError("Dataset must have 'segmented_indices' and 'unsegmented_indices' attributes")

        # Adjust ratios if one group is empty(Just for Debugging with less limits)
        # if len(self.seg_indices) == 0:
        #     print("UnifiedTaskSampler: No segmented samples found. Adjusting to 0'%' segmented.")
        #     self.num_seg = 0
        #     self.num_unseg = batch_size
        # elif len(self.unseg_indices) == 0:
        #     print("UnifiedTaskSampler: No unsegmented samples found. Adjusting to 100'%' segmented.")
        #     self.num_seg = batch_size
        #     self.num_unseg = 0
            
        print(f"StratifiedBatchSampler: {len(self.seg_indices)} segmented, {len(self.unseg_indices)} unsegmented")
        print(f"Batch composition: {self.num_seg} seg + {self.num_unseg} unseg")

    def __len__(self):
        # Calculates Epoch Length
        num_batches_seg = len(self.seg_indices) // self.num_seg if self.num_seg > 0 else 0
        num_batches_unseg = len(self.unseg_indices) // self.num_unseg if self.num_unseg > 0 else 0
        
        if self.num_seg == 0:
            return num_batches_unseg
        if self.num_unseg == 0:
            return num_batches_seg    
        return max(num_batches_seg, num_batches_unseg)

    def __iter__(self):
        n_batches = len(self)
        
        # Helper to generate randomized sequence of required length
        @staticmethod
        def get_random_indices(source_indices, count_needed):
            result = []
            while len(result) < count_needed:
                # Shuffling to make Segmentation samples order random to prevent overfitting
                shuffled = source_indices[:]
                if self.shuffle:  
                    random.shuffle(shuffled)
                result.extend(shuffled)
            return result[:count_needed]

        # Generate full Epoch lists (with independent shuffling for repeats)
        total_seg = n_batches * self.num_seg
        total_unseg = n_batches * self.num_unseg
        
        seg_epoch_indices = get_random_indices(self.seg_indices, total_seg)
        unseg_epoch_indices = get_random_indices(self.unseg_indices, total_unseg)
        
        for i in range(n_batches):
            seg_slice = seg_epoch_indices[i*self.num_seg : (i+1)*self.num_seg]
            unseg_slice = unseg_epoch_indices[i*self.num_unseg : (i+1)*self.num_unseg]
            batch = seg_slice + unseg_slice
            # Shuffle within batch (Essential for BN/Training stability)
            if self.shuffle:
                random.shuffle(batch)
            yield batch
