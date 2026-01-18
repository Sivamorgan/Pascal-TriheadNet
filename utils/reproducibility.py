import random
import numpy as np
import torch
import os

def set_seed(seed: int = 42, deterministic: bool = False):
    """Set random seeds for reproducibility.
    
    Args:
        seed (int): The seed value.
        deterministic (bool): If True, sets CuDNN to deterministic mode. 
                              Note that this might slow down training.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        print(f"Random Seed: {seed} (Deterministic: True)")
    else:
        print(f"Random Seed: {seed} (Deterministic: False)")

def worker_init_fn(worker_id):
    """Worker initialization function for DataLoader."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def get_generator(seed: int):
    """Get PyTorch generator for DataLoader."""
    g = torch.Generator()
    g.manual_seed(seed)
    return g
