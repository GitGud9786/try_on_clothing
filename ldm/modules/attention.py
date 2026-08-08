from inspect import isfunction
import math
import torch
import torch.nn.functional as F
from torch import nn, einsum
from einops import rearrange, repeat
from typing import Optional, Any
import os

from ldm.modules.diffusionmodules.util import checkpoint

try:
    import xformers
    import xformers.ops
    XFORMERS_IS_AVAILBLE = True
except:
    XFORMERS_IS_AVAILBLE = False

# CrossAttn precision handling
import os
_ATTN_PRECISION = os.environ.get("ATTN_PRECISION", "fp32")

def exists(val):
    return val is not None


def uniq(arr):
    return{el: True for el in arr}.keys()


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


def max_neg_value(t):
    return -torch.finfo(t.dtype).max
def sim_minmax(x):
    assert x.ndim == 3, f"sim matrix shape : {x.shape} mush be [b HW hw]"
    return (x - x.min(dim=-1, keepdim=True)[0]) / (x.max(dim=-1, keepdim=True)[0] - x.min(dim=-1, keepdim=True)[0])

def init_(tensor):
    dim = tensor.shape[-1]
    std = 1 / math.sqrt(dim)
    tensor.uniform_(-std, std)
    return tensor

def get_tvloss(coords, mask, ch, cw):
    b, n, _ = coords.shape
    coords = coords.reshape(b,ch,cw,2)
    mask = mask.unsqueeze(-1)
    y_mask = mask[:,1:] * mask[:,:-1]
    x_mask = mask[:,:,1:] * mask[:,:,:-1]
    
    y_tvloss = torch.abs(coords[:,1:] - coords[:,:-1]) * y_mask
    x_tvloss = torch.abs(coords[:,:,1:] - coords[:,:,:-1]) * x_mask
    tv_loss = y_tvloss.sum() / y_mask.sum() + x_tvloss.sum() / x_mask.sum()
    return tv_loss

# feedforward
class GEGLU(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * F.gelu(gate)


class FeedForward(nn.Module):
    def __init__(self, dim, dim_out=None, mult=4, glu=False, dropout=0.):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = default(dim_out, dim)
        project_in = nn.Sequential(
            nn.Linear(dim, inner_dim),
            nn.GELU()
        ) if not glu else GEGLU(dim, inner_dim)

        self.net = nn.Sequential(
            project_in,
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim_out)
        )

    def forward(self, x):
        return self.net(x)


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


def Normalize(in_channels):
    return torch.nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)


class SpatialSelfAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.norm = Normalize(in_channels)
        self.q = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.k = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.v = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.proj_out = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=1,
                                        stride=1,
                                        padding=0)

    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        b,c,h,w = q.shape
        q = rearrange(q, 'b c h w -> b (h w) c')
        k = rearrange(k, 'b c h w -> b c (h w)')
        w_ = torch.einsum('bij,bjk->bik', q, k)

        w_ = w_ * (int(c)**(-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=2)

        # attend to values
        v = rearrange(v, 'b c h w -> b c (h w)')
        w_ = rearrange(w_, 'b i j -> b j i')
        h_ = torch.einsum('bij,bjk->bik', v, w_)
        h_ = rearrange(h_, 'b c (h w) -> b c h w', h=h)
        h_ = self.proj_out(h_)

        return x+h_
@torch.no_grad()
def attn_mask_resize(m,h,w):
    """
    m : [BS x 1 x mask_h x mask_w] => downsample, reshape and bool, [BS x h x w]
    """  
    m = F.interpolate(m, (h, w)).squeeze(1).contiguous()
    m = torch.where(m>=0.5, True, False)
    return m

def init_ip_projections(module, inner_dim, ip_context_dim, source_k=None, source_v=None, zero_init=True):
    """
    Build the decoupled key/value projections used by IP-Adapter style conditioning.

    `to_k_ip` is warm-started from the pretrained `to_k` whenever the context dims
    agree, so the semantic branch starts out reading its tokens the same way the
    pretrained branch does. `to_v_ip` is zero initialised, which makes the whole
    branch contribute exactly 0 at step 0 (out = softmax(q k^T) v = 0). The model
    therefore starts bit-identical to the baseline and learns the semantic anchor
    in gradually. Note we zero only `to_v_ip`: zeroing `to_k_ip` as well would also
    zero its gradient (d out/d k is proportional to v) and the branch would never
    train.
    """
    to_k_ip = nn.Linear(ip_context_dim, inner_dim, bias=False)
    to_v_ip = nn.Linear(ip_context_dim, inner_dim, bias=False)
    if source_k is not None and source_k.weight.shape == to_k_ip.weight.shape:
        with torch.no_grad():
            to_k_ip.weight.copy_(source_k.weight)
    if zero_init:
        to_v_ip = zero_module(to_v_ip)
    elif source_v is not None and source_v.weight.shape == to_v_ip.weight.shape:
        with torch.no_grad():
            to_v_ip.weight.copy_(source_v.weight)
    module.to_k_ip = to_k_ip
    module.to_v_ip = to_v_ip


def split_ip_context(context, ip_context, ip_num_tokens):
    """
    Resolve the (base_context, ip_context) pair for a decoupled attention layer.

    Two ways the semantic tokens arrive:
      1. explicit `ip_context` tensor - used by the warp / zero cross-attention
         blocks, where the base context is a spatial feature map whose channel
         count differs from the semantic embedding dim so they cannot be concatenated.
      2. concatenated onto the tail of `context` - used everywhere the dims match,
         matching the reference IP-Adapter attention processor.
    """
    if ip_context is not None:
        return context, ip_context
    if ip_num_tokens and context is not None and context.shape[1] > ip_num_tokens:
        end_pos = context.shape[1] - ip_num_tokens
        return context[:, :end_pos, :], context[:, end_pos:, :]
    return context, None


class CrossAttention(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.,
                 ip_context_dim=None, ip_num_tokens=0, ip_scale=1.0, ip_zero_init=True, **kwargs):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            nn.Dropout(dropout)
        )

        self.ip_num_tokens = ip_num_tokens
        self.ip_scale = ip_scale
        if ip_context_dim is not None:
            init_ip_projections(self, inner_dim, ip_context_dim, self.to_k, self.to_v, ip_zero_init)
        else:
            self.to_k_ip = None
            self.to_v_ip = None

    # Replace the forward method in CrossAttention class (around line 184-243)

    def forward(self, x, context=None, mask=None, hint=None, mask1=None, mask2=None, use_attention_tv_loss=False,
                ip_context=None, **kwargs):
        # **kwargs absorbs tv_loss_type / use_loss, which CustomBasicTransformerBlock
        # passes unconditionally. MemoryEfficientCrossAttention already swallowed them;
        # this path would raise TypeError under --use_atv_loss without xformers.
        h = self.heads
        is_self_attn = context is None
        q = self.to_q(x)
        if isinstance(context, (list, tuple)):
            contexts = [default(single_context, x) for single_context in context if single_context is not None]
        else:
            contexts = [default(context, x)]

        if self.to_k_ip is not None and not is_self_attn:
            base_context, ip_context = split_ip_context(contexts[0], ip_context, self.ip_num_tokens)
            contexts = [base_context]
        else:
            ip_context = None

        key_token_length = sum(single_context.shape[1] for single_context in contexts)

        def attend(single_context, to_k=None, to_v=None, use_masks=True):
            to_k = default(to_k, self.to_k)
            to_v = default(to_v, self.to_v)
            k = to_k(single_context)
            v = to_v(single_context)

            # The attention-mask / TV-loss geometry assumes the key axis is a
            # flattened spatial grid. The semantic branch carries 77 text tokens,
            # which has no such geometry, so it always takes the plain path.
            use_sdpa = (not (use_masks and exists(mask1)) and not (use_masks and exists(mask2)) and
                        not exists(mask) and not (use_masks and use_attention_tv_loss))

            if use_sdpa and hasattr(F, 'scaled_dot_product_attention'):
                b, n, _ = q.shape
                q_ = rearrange(q, 'b n (h d) -> b h n d', h=h)
                k_ = rearrange(k, 'b n (h d) -> b h n d', h=h)
                v_ = rearrange(v, 'b n (h d) -> b h n d', h=h)

                if _ATTN_PRECISION == "fp32":
                    with torch.autocast(enabled=False, device_type='cuda'):
                        qf, kf, vf = q_.float(), k_.float(), v_.float()
                        out = F.scaled_dot_product_attention(
                            qf, kf, vf, attn_mask=None, dropout_p=0.0, scale=self.scale
                        )
                else:
                    out = F.scaled_dot_product_attention(
                        q_, k_, v_, attn_mask=None, dropout_p=0.0, scale=self.scale
                    )

                return rearrange(out, 'b h n d -> b n (h d)')

            q_, k_, v_ = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))

            if _ATTN_PRECISION == "fp32":
                with torch.autocast(enabled=False, device_type='cuda'):
                    qf, kf = q_.float(), k_.float()
                    sim = einsum('b i d, b j d -> b i j', qf, kf) * self.scale
            else:
                sim = einsum('b i d, b j d -> b i j', q_, k_) * self.scale

            del q_, k_
            if use_masks and (exists(mask1) or exists(mask2)):
                if mask1.ndim == 4 and mask2.ndim == 4:
                    _, HW, hw = sim.shape
                    bs = mask1.shape[0]
                    dx = int((HW // 12) ** 0.5)
                    mH = int(4 * dx)
                    mW = int(3 * dx)
                    dx = int((hw // 12) ** 0.5)
                    mh = int(4 * dx)
                    mw = int(3 * dx)
                    if mH != 8:
                        mask1_resized = attn_mask_resize(mask1, mH, mW)
                        mask2_resized = attn_mask_resize(mask2, mh, mw)

                        attn_mask = mask1_resized.reshape(bs, -1).unsqueeze(-1) * mask2_resized.reshape(bs, -1).unsqueeze(1)
                        attn_mask = repeat(attn_mask, "b HW hw -> (b h) HW hw", h=h)

                        assert attn_mask.shape == sim.shape, f"mask : {attn_mask.shape}, attn map : {sim.shape}"

                        if not use_attention_tv_loss:
                            max_neg_value = -torch.finfo(sim.dtype).max
                            sim.masked_fill_(attn_mask, max_neg_value)
                else:
                    raise NotImplementedError
            if exists(mask):
                mask_ = rearrange(mask, 'b ... -> b (...)')
                max_neg_value = -torch.finfo(sim.dtype).max
                mask_ = repeat(mask_, 'b j -> (b h) () j', h=h)
                sim.masked_fill_(~mask_, max_neg_value)

            sim = sim.softmax(dim=-1)
            out = einsum('b i j, b j d -> b i d', sim, v_)
            out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
            return out

        out = torch.zeros((q.shape[0], q.shape[1], q.shape[2]), dtype=q.dtype, device=x.device)
        for single_context in contexts:
            out = out + attend(single_context)

        # Decoupled cross-attention: Attention(Q, K_c, V_c) + Attention(Q, K_i, V_i).
        # The two branches share Q but own separate K/V projections, so the semantic
        # tokens are never concatenated into the pretrained branch's key space.
        if ip_context is not None and self.to_k_ip is not None:
            out = out + self.ip_scale * attend(ip_context, self.to_k_ip, self.to_v_ip, use_masks=False)

        if not use_attention_tv_loss:
            return self.to_out(out)

        attn_loss = torch.tensor(0, dtype=x.dtype, device=x.device)
        return self.to_out(out), attn_loss

class MemoryEfficientCrossAttention(nn.Module):
    # https://github.com/MatthieuTPHR/diffusers/blob/d80b531ff8060ec1ea982b65a1b8df70f73aa67c/src/diffusers/models/attention.py#L223
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.0, zero_init=False,
                 ip_context_dim=None, ip_num_tokens=0, ip_scale=1.0, ip_zero_init=True, **kwargs):
        super().__init__()
        print(f"Setting up {self.__class__.__name__}. Query dim is {query_dim}, context_dim is {context_dim} and using "
              f"{heads} heads.")
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.heads = heads
        self.dim_head = dim_head
        if not zero_init:
            self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
            self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
            self.to_v = nn.Linear(context_dim, inner_dim, bias=False)
        else:
            self.to_q = zero_module(nn.Linear(query_dim, inner_dim, bias=False))
            self.to_k = zero_module(nn.Linear(context_dim, inner_dim, bias=False))
            self.to_v = zero_module(nn.Linear(context_dim, inner_dim, bias=False))

        self.to_out = nn.Sequential(nn.Linear(inner_dim, query_dim), nn.Dropout(dropout))
        self.attention_op: Optional[Any] = None

        self.ip_num_tokens = ip_num_tokens
        self.ip_scale = ip_scale
        if ip_context_dim is not None:
            init_ip_projections(self, inner_dim, ip_context_dim, self.to_k, self.to_v, ip_zero_init)
        else:
            self.to_k_ip = None
            self.to_v_ip = None

    def forward(
            self,
            x,
            context=None,
            mask=None,
            hint=None,
            mask1=None,
            mask2=None,
            use_attention_tv_loss=False,
            use_loss=True,
            ip_context=None,
            **kwargs
        ):
        q = self.to_q(x)
        is_self_attn = context is None
        if isinstance(context, (list, tuple)):
            contexts = [default(single_context, x) for single_context in context if single_context is not None]
        else:
            contexts = [default(context, x)]

        if self.to_k_ip is not None and not is_self_attn:
            base_context, ip_context = split_ip_context(contexts[0], ip_context, self.ip_num_tokens)
            contexts = [base_context]
        else:
            ip_context = None

        key_token_length = sum(single_context.shape[1] for single_context in contexts)
        b, _, _ = q.shape

        def attend(single_context, to_k=None, to_v=None, use_masks=True):
            to_k = default(to_k, self.to_k)
            to_v = default(to_v, self.to_v)
            k = to_k(single_context)
            v = to_v(single_context)
            q_, k_, v_ = map(
                lambda t: t.unsqueeze(3)
                .reshape(b, t.shape[1], self.heads, self.dim_head)
                .permute(0, 2, 1, 3)
                .reshape(b * self.heads, t.shape[1], self.dim_head)
                .contiguous(),
                (q, k, v),
            )

            attn_loss = torch.tensor(0, dtype=x.dtype, device=x.device)
            if use_masks and use_attention_tv_loss and key_token_length > 700 and (not is_self_attn) and key_token_length < 3000 and use_loss:
                sim = einsum('b i d, b j d -> b i j', q_, k_) * (self.dim_head ** -0.5)
                sim = sim.softmax(dim=-1)
                h = self.heads
                _, HW, hw = sim.shape
                dx = int((HW // 12) ** 0.5)
                mH = int(4 * dx)
                mW = int(3 * dx)
                dx = int((hw // 12) ** 0.5)
                mh = int(4 * dx)
                mw = int(3 * dx)

                mask1_resized = attn_mask_resize(mask1, mH, mW)  # [BS x H x W]
                reshaped_sim = sim.reshape(-1, h, mH * mW, mh, mw).mean(dim=1)
                mask1_repeat = mask1_resized
                h_linspace = torch.linspace(0, mh - 1, mh, device=sim.device)
                w_linspace = torch.linspace(0, mw - 1, mw, device=sim.device)
                grid_h, grid_w = torch.meshgrid(h_linspace, w_linspace)
                grid_hw = torch.stack([grid_h, grid_w])

                weighted_grid_hw = reshaped_sim.unsqueeze(2) * grid_hw.unsqueeze(0).unsqueeze(0)
                weighted_centered_grid_hw = weighted_grid_hw.sum((-2, -1))

                tv_loss = get_tvloss(weighted_centered_grid_hw, ~mask1_repeat, ch=mh, cw=mw)
                attn_loss = tv_loss * 0.001

            out = xformers.ops.memory_efficient_attention(q_, k_, v_, attn_bias=None, op=self.attention_op)

            if exists(mask):
                raise NotImplementedError
            out = (
                out.unsqueeze(0)
                .reshape(b, self.heads, out.shape[1], self.dim_head)
                .permute(0, 2, 1, 3)
                .reshape(b, out.shape[1], self.heads * self.dim_head)
            )
            return out, attn_loss

        out = torch.zeros((b, q.shape[1], self.heads * self.dim_head), dtype=q.dtype, device=x.device)
        attn_loss = torch.tensor(0, dtype=x.dtype, device=x.device)
        for single_context in contexts:
            partial_out, partial_loss = attend(single_context)
            out = out + partial_out
            attn_loss = attn_loss + partial_loss

        # Decoupled cross-attention: Attention(Q, K_c, V_c) + Attention(Q, K_i, V_i).
        if ip_context is not None and self.to_k_ip is not None:
            ip_out, _ = attend(ip_context, self.to_k_ip, self.to_v_ip, use_masks=False)
            out = out + self.ip_scale * ip_out

        if not use_attention_tv_loss:
            return self.to_out(out)
        return self.to_out(out), attn_loss
    
def enable_ip_adapter(root, ip_context_dim, ip_num_tokens=0, ip_scale=1.0, ip_zero_init=True,
                      attn_names=("attn2",), verbose_name=""):
    """
    Install decoupled IP-Adapter key/value projections on every cross-attention
    module under `root`.

    Done as a post-construction pass rather than by threading kwargs through
    UNetModel -> SpatialTransformer -> BasicTransformerBlock, which would touch a
    lot of baseline code for no behavioural gain.

    Only modules named in `attn_names` are touched - by default that is `attn2`,
    the cross-attention. `attn1` (self-attention) is deliberately left untouched:
    no self-attention surgery.

    `ip_num_tokens > 0` means the semantic tokens arrive concatenated on the tail
    of the normal context and get split off inside the layer. `ip_num_tokens == 0`
    means they arrive as an explicit `ip_context` argument instead, which is what
    the warp / zero cross-attention blocks use because their base context is a
    spatial feature map with an incompatible channel count.
    """
    targets = []
    for name, module in root.named_modules():
        if not isinstance(module, (CrossAttention, MemoryEfficientCrossAttention)):
            continue
        if name.rsplit(".", 1)[-1] not in attn_names:
            continue
        if getattr(module, "to_k_ip", None) is not None:
            continue
        targets.append(module)

    for module in targets:
        inner_dim = module.to_k.weight.shape[0]
        init_ip_projections(module, inner_dim, ip_context_dim, module.to_k, module.to_v, ip_zero_init)
        module.to_k_ip = module.to_k_ip.to(module.to_k.weight.device, dtype=module.to_k.weight.dtype)
        module.to_v_ip = module.to_v_ip.to(module.to_v.weight.device, dtype=module.to_v.weight.dtype)
        module.ip_num_tokens = ip_num_tokens
        module.ip_scale = ip_scale

    print(f"[IP-Adapter] installed decoupled K/V on {len(targets)} cross-attn layers "
          f"in {verbose_name or type(root).__name__} (ip_num_tokens={ip_num_tokens}, scale={ip_scale})")
    return len(targets)


def ip_adapter_parameters(root):
    """Every parameter belonging to the decoupled semantic branch, for the optimizer."""
    params = []
    for module in root.modules():
        if isinstance(module, (CrossAttention, MemoryEfficientCrossAttention)) and getattr(module, "to_k_ip", None) is not None:
            params += list(module.to_k_ip.parameters())
            params += list(module.to_v_ip.parameters())
    return params


class BasicTransformerBlock(nn.Module):
    ATTENTION_MODES = {
        "softmax": CrossAttention,  # vanilla attention
        "softmax-xformers": MemoryEfficientCrossAttention
    }
    def __init__(self, dim, n_heads, d_head, dropout=0., context_dim=None, gated_ff=True, checkpoint=True,
                 disable_self_attn=False, attn_drop=0.0,attn_res=None,use_learnable_temperature=False,is_lora=False,lora_context_dim=None):
        super().__init__()
        attn_mode = "softmax-xformers" if XFORMERS_IS_AVAILBLE else "softmax"
        assert attn_mode in self.ATTENTION_MODES
        attn_cls = self.ATTENTION_MODES[attn_mode]
        self.disable_self_attn = disable_self_attn
        self.attn1 = attn_cls(query_dim=dim, heads=n_heads, dim_head=d_head, dropout=dropout,
                              context_dim=context_dim if self.disable_self_attn else None, attn_drop=attn_drop,attn_res=attn_res,use_learnable_temperature=use_learnable_temperature,is_lora=is_lora,lora_context_dim=lora_context_dim)  # is a self-attention if not self.disable_self_attn
        self.ff = FeedForward(dim, dropout=dropout, glu=gated_ff)
        self.attn2 = attn_cls(query_dim=dim, context_dim=context_dim,
                              heads=n_heads, dim_head=d_head, dropout=dropout, attn_drop=attn_drop,attn_res=attn_res,use_learnable_temperature=use_learnable_temperature)  # is self-attn if context is none
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.checkpoint = checkpoint

    def forward(self, x, context=None,hint=None):
        if hint is None:
            return checkpoint(self._forward, (x, context), self.parameters(), self.checkpoint)
        else:
            return checkpoint(self._forward, (x, context, hint), self.parameters(), self.checkpoint)

    def _forward(self, x, context=None,hint=None):
        x = self.attn1(self.norm1(x), context=context if self.disable_self_attn else None,hint=hint) + x
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
    NEW: use_linear for more efficiency instead of the 1x1 convs
    """
    def __init__(self, in_channels, n_heads, d_head,
                 depth=1, dropout=0., context_dim=None,
                 disable_self_attn=False, use_linear=False,
                 use_checkpoint=True, attn_drop=0.0,attn_res=None,use_learnable_temperature=False, is_lora=False,lora_context_dim=None):
        super().__init__()
        if exists(context_dim) and not isinstance(context_dim, list):
            context_dim = [context_dim]
        self.in_channels = in_channels
        inner_dim = n_heads * d_head
        self.norm = Normalize(in_channels)
        if not use_linear:
            self.proj_in = nn.Conv2d(in_channels,
                                     inner_dim,
                                     kernel_size=1,
                                     stride=1,
                                     padding=0)
        else:
            self.proj_in = nn.Linear(in_channels, inner_dim)

        self.transformer_blocks = nn.ModuleList(
            [BasicTransformerBlock(inner_dim, n_heads, d_head, dropout=dropout, context_dim=context_dim[d],
                                   disable_self_attn=disable_self_attn, checkpoint=use_checkpoint, attn_drop=attn_drop,attn_res=attn_res,use_learnable_temperature=use_learnable_temperature,is_lora=is_lora,lora_context_dim=lora_context_dim)
                for d in range(depth)]
        )
        if not use_linear:
            self.proj_out = zero_module(nn.Conv2d(inner_dim,
                                                  in_channels,
                                                  kernel_size=1,
                                                  stride=1,
                                                  padding=0))
        else:
            self.proj_out = zero_module(nn.Linear(in_channels, inner_dim))
        self.use_linear = use_linear
        self.is_lora = is_lora

    def forward(self, x, context=None,hint=None):
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
            x = block(x, context=context[i],hint=hint)
        if self.use_linear:
            x = self.proj_out(x)
        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w).contiguous()
        if not self.use_linear:
            x = self.proj_out(x)
        return x + x_in