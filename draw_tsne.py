import torch
from sklearn.manifold import TSNE
from umap import UMAP
import os
import matplotlib.pyplot as plt
from tqdm import tqdm



def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

def tsne(past_key_values, layer_idx, label):
    """
    value_state: [batch_size, num_heads, seq_len, head_dim]
    """
    value_state = past_key_values[layer_idx][1].squeeze(0)
    _, seq_len, _ = value_state.shape

    flattened_value_state = value_state.permute(1, 0, 2).reshape(seq_len, -1).to(torch.float).cpu().numpy()
    tsne = TSNE(n_components=2, random_state=42)
    # tsne = UMAP(n_components=2, random_state=42)
    tsne_result = tsne.fit_transform(flattened_value_state)

    plt.figure(figsize=(10, 8))
    plt.scatter(tsne_result[:, 0], tsne_result[:, 1], c=range(seq_len), cmap='viridis')
    # plt.colorbar(label='Token Index', orientation='horizontal')
    # plt.title(f'Layer {layer_idx}')
    ax = plt.gca()
    for spine in ax.spines.values():
        spine.set_linewidth(2)

    caption_text = f'Layer: {layer_idx}'
    plt.text(0.95, 0.05, caption_text, transform=ax.transAxes, fontsize=22, fontweight='bold', color='white', 
         verticalalignment='bottom', horizontalalignment='right', 
         bbox=dict(facecolor='gray', edgecolor='black', boxstyle='round,pad=0.5'))
    ax.tick_params(axis='both', labelsize=14, width=2)
    ax.tick_params(axis='x', length=10)
    ax.tick_params(axis='y', length=10)
    if not os.path.exists(f'tsne'):
        os.makedirs(f'tsne')
    plt.savefig(f'tsne/{layer_idx}_tsne_{label}.png', dpi=120, bbox_inches='tight')

if __name__ == "__main__":
    # 32, 2, [1, 32, 275, 128] 


    past_key_values = torch.load("/home/sjx/kv/LongBench/exps/tsne_exps/llama2-7b-0.35.pt")
    # past_key_values_ours = torch.load("past_key_values_035.pt")
    
    for layer_idx in tqdm(range(32), desc="Processing layers"):
        # tsne(past_key_values_ours, layer_idx, 'ours_0.35')
        tsne(past_key_values, layer_idx, 'origin')