import torch.nn as nn

class CrossAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.0, batch_first=False, device=None, dtype=None):
        super(CrossAttention, self).__init__()
        self.multihead_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=batch_first, device=device, dtype=dtype)
    
    def forward(self, query, key, value, key_padding_mask=None, need_weights=True, attn_mask=None, average_attn_weights=True):
        return self.multihead_attn(query, key, value, key_padding_mask=key_padding_mask, need_weights=need_weights, attn_mask=attn_mask, average_attn_weights=average_attn_weights)
