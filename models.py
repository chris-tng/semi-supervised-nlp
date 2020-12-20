import torch
import torch.nn as nn
import torch.nn.functional as F

from layers import ManifoldMixup


class GenericMixupModel(nn.Module):
    """
    Generic model consists of an embedding layer, an encoder, a pooler and a classification head 
    """
    
    def __init__(self, embed: nn.Module, encoder: nn.Module, 
                pooler: nn.Module, cls_head: nn.Module, use_mixup: bool = True):
        super().__init__()
        self.embed = embed
        self.encoder = encoder
        self.pooler = pooler
        self.cls_head = cls_head
        if use_mixup:
            self.mixup = nn.Sequential(
                ManifoldMixup(), nn.ReLU()
            )
        
    def forward(self, x1, x1_lens=None, x2=None, x2_lens=None, mixup_factor: float=1.):
        """
        - x2: example to mixup with x1
        - mixup_factor: 1 no mixup
        """
        x1_embed = self.embed(x1)
        x1_encoded, _ = self.encoder(x1_embed, x1_lens)
        
        if x2 is not None:
            x2_embed = self.embed(x2)
            x2_encoded, _ = self.encoder(x2_embed, x2_lens)
            x_encoded = self.mixup(x1_encoded, x2_encoded, mixup_factor)
        else:
            x_encoded = x1_encoded
            
        x_pooled = self.pooler(x_encoded)
        logits = self.cls_head(x_pooled)
        return logits
