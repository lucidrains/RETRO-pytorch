import torch
import torch.nn.functional as F
from torch import nn, einsum

from einops import rearrange, repeat

# helper functions

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def divisible_by(val, divisor):
    return (val / divisor).is_integer()

def cast_tuple(val, num = 1):
    return val if isinstance(val, tuple) else ((val,) * num)

# normalization

class RMSNorm(nn.Module):
    def __init__(
        self,
        dim,
        *,
        eps = 1e-8
    ):
        super().__init__()
        self.eps = eps
        self.scale = dim ** -0.5
        self.gamma = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = x.norm(keepdim = True, dim = -1) * self.scale
        return (x / norm.clamp(min = self.eps)) * self.gamma

# positional embedding

class RotaryEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        inv_freq = 1. / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, max_seq_len, device):
        seq = torch.arange(max_seq_len, device = device)
        freqs = einsum('i , j -> i j', seq.type_as(self.inv_freq), self.inv_freq)
        emb = torch.cat((freqs, freqs), dim = -1)
        return rearrange(emb, 'n d -> 1 1 n d')

def rotate_half(x):
    x = rearrange(x, '... (j d) -> ... j d', j = 2)
    x1, x2 = x.unbind(dim = -2)
    return torch.cat((-x2, x1), dim = -1)

def apply_rotary_pos_emb(t, freqs):
    seq_len = t.shape[-2]
    freqs = freqs[:, :, -seq_len:]
    return (t * freqs.cos()) + (rotate_half(t) * freqs.sin())

# feedforward

def FeedForward(dim, mult = 4, dropout = 0.):
    inner_dim = int(mult * dim)

    return nn.Sequential(
        RMSNorm(dim),
        nn.Linear(dim, inner_dim),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(inner_dim, dim)
    )

# attention

class Attention(nn.Module):
    def __init__(
        self,
        dim,
        *,
        dim_head = 64,
        heads = 8,
        causal = False,
        dropout = 0.
    ):
        super().__init__()
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.causal = causal
        inner_dim = dim_head * heads

        self.norm = RMSNorm(dim)
        self.dropout = nn.Dropout(dropout)

        self.to_q = nn.Linear(dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias = False)
        self.to_out = nn.Linear(inner_dim, dim)

    def forward(self, x, context = None, pos_emb = None):
        device, h, scale = x.device, self.heads, self.scale
        kv_input = default(context, x)

        x = self.norm(x)

        q = self.to_q(x)
        k, v = self.to_kv(kv_input).chunk(2, dim = -1)

        # split heads

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (q, k, v))

        # scale

        q = q * scale

        # apply relative positional encoding (rotary embeddings)

        if exists(pos_emb):
            q_pos_emb, k_pos_emb = cast_tuple(pos_emb, num = 2)
            q = apply_rotary_pos_emb(q, q_pos_emb)
            k = apply_rotary_pos_emb(k, k_pos_emb)

        # derive query key similarities

        sim = einsum('b h i d, b h j d -> b h i j', q, k)

        # masking

        if self.causal:
            i, j = sim.shape[-2:]
            causal_mask = torch.ones(i, j, device = device, dtype = torch.bool).triu(j - i + 1)
            mask_value = -torch.finfo(sim.dtype).max
            sim = sim.masked_fill(causal_mask, mask_value)

        # attention

        attn = sim.softmax(dim = -1)

        attn = self.dropout(attn)

        # aggregate

        out = einsum('b h i j, b h j d -> b h i d', attn, v)

        # merge heads

        out = rearrange(out, 'b h n d -> b n (h d)')

        # combine heads linear out

        return self.to_out(out)


class ChunkedCrossAttention(nn.Module):
    def __init__(
        self,
        **kwargs
    ):
        super().__init__()
        self.cross_attn = Attention(**kwargs)

    def forward(self, x, *, context, pos_emb = None, **kwargs):
        # derive variables

        b, n, chunk_size = x.shape[0], x.shape[-2], context.shape[-2]
        causal_padding = chunk_size - 1

        # causal padding

        x = F.pad(x, (0, 0, -causal_padding, causal_padding), value = 0.)

        if exists(pos_emb):
            q_pos_emb, k_pos_emb = cast_tuple(pos_emb, num = 2)

            # make sure queries positions are properly shifted
            q_pos_emb = F.pad(q_pos_emb, (0, 0, -causal_padding, causal_padding), value = 0.)

            # properly chunk positional embeddings

            q_pos_emb = repeat(q_pos_emb, '1 h (k n) d -> (b k) h n d', n = chunk_size, b = b)
            k_pos_emb = repeat(k_pos_emb, '1 h (k n) d -> (b k) h n d', n = chunk_size, b = b)

            pos_emb = (q_pos_emb, k_pos_emb)

        # reshape so we have chunk to chunk attention, without breaking causality

        x = rearrange(x, 'b (k n) d -> (b k) n d', n = chunk_size)
        context = rearrange(context, 'b k n d -> (b k) n d')

        # cross attention

        out = self.cross_attn(x, context = context, pos_emb = pos_emb, **kwargs)

        # reshape back to original sequence

        out = rearrange(out, '(b k) n d -> b (k n) d', k = n // chunk_size)

        # pad back to original, with 0s at the beginning (which will be added to the residual and be fine)

        out = F.pad(out, (0, 0, causal_padding, -causal_padding), value = 0.)
        return out

class Transformer(nn.Module):
    def __init__(
        self,
        dim,
        *,
        depth,
        causal = False,
        heads = 8,
        dim_head = 64,
        attn_dropout = 0.,
        ff_mult = 4,
        ff_dropout = 0.,
        final_norm = True,
        cross_attn_layers = tuple(),
        chunked_cross_attn = False
    ):
        super().__init__()
        self.layers = nn.ModuleList([])

        rotary_emb_dim = max(default(dim_head, dim_head // 2), 32)
        self.rotary_pos_emb = RotaryEmbedding(rotary_emb_dim)

        for layer_num in range(1, depth + 1):
            has_cross_attn = layer_num in cross_attn_layers
            cross_attn_klass = Attention if not chunked_cross_attn else ChunkedCrossAttention

            self.layers.append(nn.ModuleList([
                Attention(dim = dim, dim_head = dim_head, heads = heads, dropout = attn_dropout, causal = causal),
                cross_attn_klass(dim = dim, dim_head = dim_head, heads = heads, dropout = attn_dropout) if has_cross_attn else None,
                FeedForward(dim = dim, mult = ff_mult, dropout = ff_dropout),
            ]))

        self.norm_out = RMSNorm(dim) if final_norm else nn.Identity()

    def forward(self, x, *, context):
        device, n = x.device, x.shape[-2]
        pos_emb = self.rotary_pos_emb(n, device = device)

        for attn, cross_attn, ff in self.layers:
            x = attn(x, pos_emb = pos_emb) + x

            if exists(cross_attn):
                x = cross_attn(x, context = context, pos_emb = pos_emb) + x

            x = ff(x) + x

        return self.norm_out(x)

# main class

class RETRO(nn.Module):
    def __init__(
        self,
        *,
        num_tokens,
        dim,
        max_seq_len = 2048,
        enc_depth = 12,
        enc_cross_attn_layers = (1, 3, 6, 9),
        dec_depth = 12,
        dec_cross_attn_layers = (1, 3, 6, 9),
        heads = 8,
        dim_head = 64,
        enc_attn_dropout = 0.,
        enc_ff_dropout = 0.,
        dec_attn_dropout = 0.,
        dec_ff_dropout = 0.
    ):
        super().__init__()
        self.token_emb = nn.Embedding(num_tokens, dim)
        self.pos_emb = nn.Embedding(max_seq_len, dim)

        self.encoder = Transformer(
            dim = dim,
            depth = enc_depth,
            attn_dropout = enc_attn_dropout,
            ff_dropout = enc_ff_dropout,
            cross_attn_layers = enc_cross_attn_layers
        )

        self.decoder = Transformer(
            dim = dim,
            depth = dec_depth,
            attn_dropout = dec_attn_dropout,
            ff_dropout = dec_ff_dropout,
            causal = True,
            chunked_cross_attn = True,
            cross_attn_layers = dec_cross_attn_layers
        )

        self.to_logits = nn.Linear(dim, num_tokens)

    def forward(
        self,
        seq,
        retrieved,
        return_loss = False
    ):
        assert not (return_loss and not self.training), 'must be training if returning loss'

        # if training, derive labels

        if return_loss:
            seq, labels = seq[:, :-1], seq[:, 1:]

        # variables

        n, chunk_size, device = seq.shape[-1], retrieved.shape[-1], seq.device

        assert divisible_by(n, chunk_size), 'sequence length must be divisible by chunk size'

        # embed both sequence and retrieved chunks

        embed = self.token_emb(seq)
        retrieved = self.token_emb(retrieved)

        # get absolute positional embedding

        pos_emb = self.pos_emb(torch.arange(n, device = device))
        pos_emb = rearrange(pos_emb, 'n d -> 1 n d')
        embed = embed + pos_emb

        # encode

        retrieved = rearrange(retrieved, 'b k n d -> (b k) n d')
        embed_as_context = rearrange(embed, 'b (k n) d -> (b k) n d', n = chunk_size)

        retrieved = self.encoder(retrieved, context = embed_as_context)
        retrieved = rearrange(retrieved, '(b k) n d -> b k n d', k = n // chunk_size)

        # decode

        embed = self.decoder(embed, context = retrieved)

        # project to logits

        logits = self.to_logits(embed)

        if not return_loss:
            return logits

        # cross entropy loss

        loss = F.cross_entropy(rearrange(logits, 'b n c -> b c n'), labels)
        return loss
