import torch
import json
from transformers.cache_utils import Cache
from transformers.utils import deprecate_kwarg
from typing import List, Optional

class DynamicCache(Cache):
    @deprecate_kwarg("num_hidden_layers", version="4.47.0")
    def __init__(self, num_hidden_layers: Optional[int] = None) -> None:
        super().__init__()
        self._seen_tokens = 0
        self.key_cache: List[torch.Tensor] = []
        self.value_cacle: List[torch.Tensor] = []
    def update_last_token(self, value_states: torch.Tensor, layer_idx:int, ratio: float, group_size: int):
        """
        Update the last token of the cache by adding the attn_output
        
        Args:
            value_states: The new value states to cache.
            layer_idx: The index of the layer to cache the states for.
            ratio: The ratio of the last token to update.
            group_size: The group size for GQA.
        """
        if len(self.value_cache) <= layer_idx:
            raise ValueError(f"Layer index {layer_idx} is out of bounds for cache with {len(self.value_cache)} layers")
        bsz, num_heads, seq_len, head_dim = value_states.shape
        if seq_len > 1:
            pass
        else:
            if group_size > 1:
                value_states = value_states.view(bsz, num_heads // group_size, group_size, seq_len, head_dim).mean(dim=2)
            
            self.value_cache[layer_idx][:, :, -1:] += ratio * value_states
    
    
    