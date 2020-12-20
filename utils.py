import torch
from torch import Tensor


def sharpen(p: Tensor, T: float):
    "Sharpen a probability distribution"
    p1 = p**(1/T)
    return p1 / p1.sum(dim=-1, keepdim=True)


def softmax_with_mask(x: Tensor, mask: Tensor, dim: int=-1):
    """
    Perform softmax over x's dim factoring boolean `mask` of the same shape
    """
    
    assert x.shape == mask.shape, f"Input's shape {x.shape} and mask's shape {mask.shape} need to be equal"
    scores = F.softmax(x, dim)
    masked_scores = scores * mask.float()
    return masked_scores / (masked_scores.sum(dim, keepdim=True) + 1e-10)


def sequence_mask(lengths: Tensor, max_len: int=None):
    """
    Creates a boolean mask from sequence lengths
    - lengths: 1D tensor
    """
    batch_size = lengths.numel()
    max_len = max_len or lengths.max()
    return (torch.arange(0, max_len, device=lengths.device).type_as(lengths)
            .unsqueeze(0).expand(batch_size, max_len).lt(lengths.unsqueeze(1)))
