import torch

def compute_locations(features, stride):
    """Compute (x, y) locations for a feature map.
    
    Args:
        features,
        stride: stride of the feature map
    Returns:
        locations: (H*W, 2) locations (x, y)
    """
    h, w = features.shape[-2:]
    shifts_x = torch.arange(0, w * stride, step=stride, dtype=torch.float32, device=features.device)
    shifts_y = torch.arange(0, h * stride, step=stride, dtype=torch.float32, device=features.device)
    shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x, indexing='ij')
    shift_x = shift_x.reshape(-1)
    shift_y = shift_y.reshape(-1)
    locations = torch.stack((shift_x, shift_y), dim=1) + stride // 2.0
    return locations

