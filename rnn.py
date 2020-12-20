# +
__all__ = ["RNNDropout", "WeightDropout", "EmbeddingDropout", 
           "LSTMWeightDrop", "HANAttention"]

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
# -

from typing import Tuple


# ### RNNDropout

def dropout_mask(x: Tensor, size: Tuple, p: float):
    """Return a dropout mask of the same type as `x`, 
    size `size`, with probability `p` to nullify an element."""
    return x.new(*size).bernoulli_(1-p).div_(1-p)


class RNNDropout(nn.Module):
    "Dropout with probability `p` that is consistent on the seq_len dimension."
    def __init__(self, p: float=0.5): 
        super().__init__()
        self.p = p

    def forward(self, x: Tensor):
        """batch-major x of shape (batch_size, seq_len, feature_size)"""
        assert x.ndim == 3, f"Expect x of dimension 3, whereas dim x is {x.ndim}"
        if not self.training or self.p == 0.: return x
        return x * dropout_mask(x.data, (x.size(0), 1, x.size(2)), self.p)


# ### WeightDropout

# +
import warnings

class WeightDropout(nn.Module):
    """
    Wrapper around another layer in which some weights 
    will be replaced by 0 during training.
    Args:
    - module {nn.Module}: the module being wrapped
    - weight_p {float}: probability of dropout
    - layer_names {List[str]}: names of weights of `module` being dropped out. 
        By default: it drops hidden to hidden connection of LSTM
    """

    def __init__(self, module: nn.Module, weight_p: float, 
                 layer_names: List[str]=['weight_hh_l0']):
        super().__init__()
        self.module, self.weight_p = module, weight_p
        self.layer_names = layer_names
        for layer in self.layer_names:
            # Makes a copy of the weights of the selected layers.
            w = getattr(self.module, layer)
            self.register_parameter(f'{layer}_raw', nn.Parameter(w.data))
            self.module._parameters[layer] = F.dropout(w, p=self.weight_p, training=False)

    def _setweights(self):
        "Apply dropout to the raw weights."
        for layer in self.layer_names:
            raw_w = getattr(self, f'{layer}_raw')
            self.module._parameters[layer] = F.dropout(raw_w, p=self.weight_p, training=self.training)

    def forward(self, *args):
        self._setweights()
        with warnings.catch_warnings():
            #To avoid the warning that comes because the weights aren't flattened.
            warnings.simplefilter("ignore")
            return self.module.forward(*args)

    def reset(self):
        for layer in self.layer_names:
            raw_w = getattr(self, f'{layer}_raw')
            self.module._parameters[layer] = F.dropout(raw_w, p=self.weight_p, training=False)
        if hasattr(self.module, 'reset'): self.module.reset()


# -

# ### EmbeddingDropout

class EmbeddingDropout(nn.Module):
    "Apply dropout with probabily `embed_p` to an embedding layer `emb`."

    def __init__(self, emb: nn.Module, embed_p: float):
        super().__init__()
        self.emb, self.embed_p = emb, embed_p

    def forward(self, x: Tensor, scale: float=None):
        if self.training and self.embed_p != 0:
            size = (self.emb.weight.size(0),1)
            mask = dropout_mask(self.emb.weight.data, size, self.embed_p)
            masked_embed = self.emb.weight * mask
        else: 
            masked_embed = self.emb.weight
        
        if scale: 
            masked_embed.mul_(scale)
        return F.embedding(x, masked_embed, 
            self.emb.padding_idx or -1, self.emb.max_norm,
            self.emb.norm_type, self.emb.scale_grad_by_freq, 
            self.emb.sparse)


# ### LSTMWeightDrop

# +
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

def one_param(m):
    "First parameter in `m`"
    return first(m.parameters())

def first(x): return next(iter(x))

def _to_detach(b):  
    return [(o[0].detach(), o[1].detach()) if isinstance(o, tuple) else o.detach() 
            for o in b]


# +
from typing import Tuple

class LSTMWeightDrop(nn.Module):
    """
    LSTM with dropouts
    
    Args:
    - input_p : float - RNNDropout applied to input after embedding
    - weight_p : float - WeightDropout applied to hidden-hidden connection of LSTM
    - hidden_p : float - RNNDropout applied to two of the inner LSTMs
    - hidden_sz : int - total hidden size including bidir
    
    Outputs:
    - raw outputs : List[torch.Tensor] - activation for each layer without dropout in reverse order, last at index 0
    - outputs : List[torch.Tensor] - activation for each layer with dropout in reverse order, last at index 0
    """
    
    def __init__(self, input_size, hidden_size, num_layers=1, 
                 bidirectional=False, hidden_p=0.2, input_p=0.6,
                 weight_p=0.5, pack_pad_seq=False):
        super().__init__()
        self.input_size, self.hidden_size = input_size, hidden_size 
        self.num_layers = num_layers
        self.pack_pad_seq = pack_pad_seq
        self.batch_sz = 1
        self.n_dir = 2 if bidirectional else 1
        self.rnns = nn.ModuleList([
            self._one_rnn(
                n_in = input_size if layer_idx == 0 else hidden_size*self.n_dir, 
                n_out = hidden_size, 
                bidir = bidirectional, weight_p = weight_p) 
            for layer_idx in range(num_layers)]
        )
        self.input_dropout = RNNDropout(input_p)
        self.hidden_dropouts = nn.ModuleList(
            [RNNDropout(hidden_p) for l in range(num_layers)]
        )

    def forward(self, x: Tensor, x_lens=None):
        """
        Args:
        - x : Tensor - batch-major input of shape `(batch, seq_len, emb_sz)`
        - x_lens : Tensor - 1D tensor containing sequence length
        
        Outputs:
        - raw outputs : List[Tensor] - activation for each layer without dropout
        - outputs : List[Tensor] - activation for each layer with dropout
        """
        batch_sz, seq_len = x.shape[:2]
        if batch_sz != self.batch_sz: 
            self.batch_sz = batch_sz
            # self.reset()
        
        all_h = self.input_dropout(x)
        last_hiddens, raw_outputs, outputs = [], [], []
        for layer_idx, (rnn, hid_dropout) in enumerate(zip(self.rnns, self.hidden_dropouts)):
            if self.pack_pad_seq: 
                if x_lens is not None: 
                    all_h = pack_padded_sequence(
                        all_h, x_lens, batch_first=True, enforce_sorted=False)
                else: 
                    raise ValueError("Please supply `x_lens` when pack_pad_seq=True")
            # all_h, last_hidden = rnn(all_h, self.last_hiddens[layer_idx])
            all_h, last_hidden = rnn(all_h)
            if self.pack_pad_seq:
                all_h = pad_packed_sequence(all_h, batch_first=True)[0]
            
            last_hiddens.append(last_hidden)
            raw_outputs.append(all_h)
            # apply dropout to hidden states except last layer
            if layer_idx != self.num_layers - 1: 
                all_h = hid_dropout(all_h)
            outputs.append(all_h)
        # self.last_hiddens = _to_detach(last_hiddens)
        self.raw_ouputs = raw_outputs
        self.outputs = outputs
        return all_h, last_hidden

    def _one_rnn(self, n_in, n_out, bidir, weight_p):
        "Return one of the inner rnn wrapped by WeightDropout"
        rnn = nn.LSTM(n_in, n_out, 1, batch_first=True, bidirectional=bidir)
        return WeightDropout(rnn, weight_p)

    def _init_h0(self, layer_idx: int) -> Tuple:
        "Init (h0, c0) as zero tensors for layer i"
        h0 = one_param(self).new_zeros(self.n_dir, 
                                       self.batch_sz, self.hidden_size)
        c0 = one_param(self).new_zeros(self.n_dir, 
                                       self.batch_sz, self.hidden_size)
        return (h0, c0)
    
    def reset(self):
        "Reset the hidden states - (for weightdrop)"
        [r.reset() for r in self.rnns if hasattr(r, 'reset')]
        self.last_hiddens = [self._init_h0(l) for l in range(self.num_layers)]


# -
def init_weight_bias(model, init_range=0.1):    
    for name_w, w in model.named_parameters():
        if "weight" in name_w:
            w.data.uniform_(-init_range, init_range)
        elif "bias" in name_w:
            w.bias.data.fill_(0.)


# ### HANAttention

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


class HANAttention(nn.Module):
    """
    HAN Attention described in [Hierarchial Attention Networks - ACL16]
    with multi-head mechanism and diversity penalization specified in 
    [A Structure Self-Attentive Sentence Embedding - ICLR17]
    sometimes referred as SelfAttention
    
    Attrs:
    - input_size: num of features / embedding sz
    - n_heads: num of subspaces to project input
    - pool_mode : flatten to return summary vectors, 
    otherwise sum across features
    """
    
    def __init__(self, input_size: int, attention_size: int, n_heads: int, 
                 pool_mode: str="flatten"):
        super().__init__()
        self.n_heads, self.pool_mode = n_heads, pool_mode
        self.proj = nn.Linear(input_size, attention_size, bias=False)
        self.queries = nn.Linear(attention_size, n_heads, False)
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_normal_(self.proj.weight)
        torch.nn.init.xavier_normal_(self.queries.weight)
        
    def forward(self, x: Tensor, x_lens: Tensor=None):
        """
        Args:
            x : input of shape `(batch_sz, seq_len, n_features)`
            x_lens : lengths of x of shape `(batch_sz)`
        """
        x_proj = torch.tanh(self.proj(x))
        x_queries_sim = self.queries(x_proj)
        if x_lens is not None:
            masks = sequence_mask(x_lens).unsqueeze(-1)
            # attn_w: (batch_sz, seq_len, n_head)
            attn_w = softmax_with_mask(x_queries_sim, 
                                       masks.expand_as(x_queries_sim), dim=1)
        else:
            attn_w = F.softmax(x_queries_sim, dim=1)
        # x_attended: (batch_sz, n_head, n_features)
        x_attended = attn_w.transpose(2, 1) @ x
        self.attn_w = attn_w
        return self.pool(x_attended), attn_w
    
    def pool(self, x): 
        return x.flatten(1, 2) if self.pool_mode=="flatten" else x.mean(dim=1)
    
    def diversity(self):
        "Don't seem to work at all"
        # cov: (batch_sz, n_head, n_head)
        cov = self.attn_w.transpose(2, 1).bmm(self.attn_w) - torch.eye(self.n_head, device=self.attn_w.device).unsqueeze(0)
        return (cov**2).sum(dim=[1, 2])



