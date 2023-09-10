import torch
import torch.nn as nn
from torch import einsum
from einops import rearrange, repeat


class CrossAttn(nn.Module):
    def __init__(self, query_dim, cross_dim=None, heads=8, dim_head=64, dropout=0.,device='cuda'):
        super(CrossAttn, self).__init__()
        inner_dim = dim_head * heads
        if cross_dim is None:
            cross_dim = query_dim

        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(cross_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(cross_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x1, x2=None):
        # [b, feature_len, embed_dim]
        h = self.heads
        if x2 is None:
            x2 = x1
        q = self.to_q(x1)
        k = self.to_k(x2)
        v = self.to_v(x2)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))

        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale

        del q, k

        # attention, what we cannot get enough of
        sim = sim.softmax(dim=-1)

        out = einsum('b i j, b j d -> b i d', sim, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
        return self.to_out(out)


class MultiAttn(nn.Module):
    def __init__(self, query_dim, cross_dim, heads=8, dim_head=64, dropout=0.,device='cuda'):
        super(MultiAttn, self).__init__()
        self.in_linear = nn.Linear(query_dim, query_dim)
        self.c_attn = CrossAttn(query_dim, cross_dim, heads=heads, dim_head=dim_head, dropout=dropout).to(device)
        self.s_attn = CrossAttn(query_dim, heads=heads, dim_head=dim_head, dropout=dropout).to(device)

    def forward(self, x1, x2):
        x_in = x1
        x1 = self.c_attn(x1, x2)
        x1 = self.s_attn(x1)
        return x1 + x_in
