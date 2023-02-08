import torch
import torch.nn.functional as F
from torch import nn, einsum
from einops import rearrange, repeat
from typing import Any, Optional
from .util import DummyModule

xformers_loaded = True
try:
    import xformers.ops
except:
    xformers_loaded = False

from .util import exists, default, zero_module

# feedforward
class GEGLU(nn.Module):
    def __init__(self, dim_in, dim_out, dtype, device):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2, dtype=dtype, device=device)

    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * F.gelu(gate)


class FeedForward(nn.Module):
    def __init__(self, dim, dim_out=None, mult=4, glu=False, dtype=torch.float32, device="cuda"):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = default(dim_out, dim)
        project_in = nn.Sequential(
            nn.Linear(dim, inner_dim),
            nn.GELU()
        ) if not glu else GEGLU(dim, inner_dim, dtype, device=device)

        self.net = nn.Sequential(
            project_in,
            DummyModule(),
            nn.Linear(inner_dim, dim_out, dtype=dtype, device=device)
        )

    def forward(self, x):
        return self.net(x)


def Normalize(in_channels, dtype, device):
    return torch.nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True, dtype=dtype, device=device)


class CrossAttention(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dtype=torch.float32, device="cuda"):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False, dtype=dtype, device=device)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False, dtype=dtype, device=device)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False, dtype=dtype, device=device)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim, dtype=dtype, device=device),
        )

    def forward(self, x, context=None, mask=None):
        h = self.heads

        q = self.to_q(x)
        context = default(context, x)
        k = self.to_k(context)
        v = self.to_v(context)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))

        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale

        if exists(mask):
            mask = rearrange(mask, 'b ... -> b (...)')
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, 'b j -> (b h) () j', h=h)
            sim.masked_fill_(~mask, max_neg_value)

        # attention, what we cannot get enough of
        attn = sim.softmax(dim=-1)

        out = einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
        return self.to_out(out)


class MemoryEfficientCrossAttention(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dtype=torch.float32, device="cuda"):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.heads = heads
        self.dim_head = dim_head

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False, dtype=dtype, device=device)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False, dtype=dtype, device=device)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False, dtype=dtype, device=device)

        self.to_out = nn.Sequential(nn.Linear(inner_dim, query_dim, dtype=dtype, device=device))
        self.attention_op: Optional[Any] = None

    def forward(self, x, context=None, mask=None):
        q = self.to_q(x)
        context = default(context, x)
        k = self.to_k(context)
        v = self.to_v(context)

        b, _, _ = q.shape
        q, k, v = map(
            lambda t: t.unsqueeze(3)
            .reshape(b, t.shape[1], self.heads, self.dim_head)
            .permute(0, 2, 1, 3)
            .reshape(b * self.heads, t.shape[1], self.dim_head)
            .contiguous(),
            (q, k, v),
        )

        # actually compute the attention, what we cannot get enough of
        out = xformers.ops.memory_efficient_attention(q, k, v, attn_bias=None, op=self.attention_op)

        # TODO: Use this directly in the attention operation, as a bias
        if exists(mask):
            raise NotImplementedError
        out = (
            out.unsqueeze(0)
            .reshape(b, self.heads, out.shape[1], self.dim_head)
            .permute(0, 2, 1, 3)
            .reshape(b, out.shape[1], self.heads * self.dim_head)
        )
        return self.to_out(out)

class BasicTransformerBlock(nn.Module):
    def __init__(self, dim, n_heads, d_head, context_dim=None, gated_ff=True, checkpoint=True, dtype=torch.float32, device="cuda"):
        super().__init__()
        cross_attention_class = MemoryEfficientCrossAttention if xformers_loaded else CrossAttention
        self.attn1 = cross_attention_class(query_dim=dim, heads=n_heads, dim_head=d_head, dtype=dtype, device=device)  # is a self-attention
        self.ff = FeedForward(dim, glu=gated_ff, dtype=dtype, device=device)
        self.attn2 = cross_attention_class(query_dim=dim, context_dim=context_dim,
                                    heads=n_heads, dim_head=d_head, dtype=dtype, device=device)  # is self-attn if context is none
        self.norm1 = nn.LayerNorm(dim, dtype=dtype, device=device)
        self.norm2 = nn.LayerNorm(dim, dtype=dtype, device=device)
        self.norm3 = nn.LayerNorm(dim, dtype=dtype, device=device)
        self.checkpoint = checkpoint

    def forward(self, x, context=None):
        x = self.attn1(self.norm1(x)) + x
        x = self.attn2(self.norm2(x), context=context) + x
        x = self.ff(self.norm3(x)) + x
        return x


class SpatialTransformer(nn.Module):
    """
    Transformer block for image-like data.
    First, project the input (aka embedding)
    and reshape to b, t, d.
    Then apply standard transformer action.
    Finally, reshape to image
    """
    def __init__(self, in_channels, n_heads, d_head, padding_mode, 
                 depth=1, context_dim=None, use_linear=False, dtype=torch.float32, device="cuda"):
        super().__init__()
        self.in_channels = in_channels
        inner_dim = n_heads * d_head
        self.norm = Normalize(in_channels, dtype, device=device)

        if use_linear:
            self.proj_in = nn.Linear(
                in_channels,
                inner_dim,
                dtype=dtype,
                device=device
            )
        else:
            self.proj_in = nn.Conv2d(
                in_channels,
                inner_dim,
                padding_mode=padding_mode,
                kernel_size=1,
                stride=1,
                padding=0,
                dtype=dtype,
                device=device
            )

        self.transformer_blocks = nn.ModuleList(
            [BasicTransformerBlock(inner_dim, n_heads, d_head, context_dim=context_dim, dtype=dtype, device=device)
                for d in range(depth)]
        )

        if use_linear:
            self.proj_out = zero_module(nn.Linear(
                in_channels, 
                inner_dim,
                dtype=dtype,
                device=device
            ))
        else:
            self.proj_out = zero_module(nn.Conv2d(
                inner_dim,
                in_channels,
                padding_mode=padding_mode,
                kernel_size=1,
                stride=1,
                padding=0,
                dtype=dtype,
                device=device
            ))
        self.use_linear = use_linear

    def forward(self, x, context=None):
        # note: if no context is given, cross-attention defaults to self-attention
        if not isinstance(context, list):
            context = [context]
        b, c, h, w = x.shape
        x_in = x
        x = self.norm(x)
        if not self.use_linear:
            x = self.proj_in(x)
        x = rearrange(x, 'b c h w -> b (h w) c').contiguous()
        if self.use_linear:
            x = self.proj_in(x)
        for i, block in enumerate(self.transformer_blocks):
            x = block(x, context=context[i])
        if self.use_linear:
            x = self.proj_out(x)
        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w).contiguous()
        if not self.use_linear:
            x = self.proj_out(x)
        return x + x_in