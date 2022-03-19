from functools import partial

import torch
import torch.nn.functional as F
from torch import nn, einsum

from retro_pytorch.retrieval import BERT_VOCAB_SIZE
from einops import rearrange, repeat

# constants

MIN_DIM_HEAD = 32

# helper functions

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def divisible_by(val, divisor):
    return (val / divisor).is_integer()

def cast_tuple(val, num = 1):
    return val if isinstance(val, tuple) else ((val,) * num)

# helper functions

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = RMSNorm(dim)

    def forward(self, x, *args, **kwargs):
        return self.fn(self.norm(x), *args, **kwargs) + x

class PostNorm(nn.Module):
    def __init__(self, dim, fn, scale_residual = 1):
        super().__init__()
        self.fn = fn
        self.scale_residual = scale_residual
        self.norm = RMSNorm(dim)

    def forward(self, x, *args, **kwargs):
        residual = x * self.scale_residual
        out = self.fn(x, *args, **kwargs) + residual
        return self.norm(out)

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

    def forward(self, max_seq_len, *, device, offset = 0):
        seq = torch.arange(max_seq_len, device = device) + offset
        freqs = einsum('i , j -> i j', seq.type_as(self.inv_freq), self.inv_freq)
        emb = torch.cat((freqs, freqs), dim = -1)
        return rearrange(emb, 'n d -> 1 1 n d')

def rotate_half(x):
    x = rearrange(x, '... (j d) -> ... j d', j = 2)
    x1, x2 = x.unbind(dim = -2)
    return torch.cat((-x2, x1), dim = -1)

def apply_rotary_pos_emb(t, freqs):
    seq_len, rot_dim = t.shape[-2], freqs.shape[-1]
    t, t_pass = t[..., :rot_dim], t[..., rot_dim:]
    t = (t * freqs.cos()) + (rotate_half(t) * freqs.sin())
    return torch.cat((t, t_pass), dim = -1)

# feedforward

def FeedForward(dim, mult = 4, dropout = 0.):
    inner_dim = int(mult * dim)

    return nn.Sequential(
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
        dropout = 0.,
        null_kv = False
    ):
        super().__init__()
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.causal = causal
        inner_dim = dim_head * heads

        self.dropout = nn.Dropout(dropout)

        self.to_q = nn.Linear(dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias = False)
        self.to_out = nn.Linear(inner_dim, dim)

        # allowing for attending to nothing (null function)
        # and to save attention from breaking if all retrieved chunks are padded out
        self.null_kv = nn.Parameter(torch.randn(2, inner_dim)) if null_kv else None

    def forward(self, x, mask = None, context = None, pos_emb = None):
        b, device, h, scale = x.shape[0], x.device, self.heads, self.scale

        kv_input = default(context, x)

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

        # add null key / values

        if exists(self.null_kv):
            nk, nv = self.null_kv.unbind(dim = 0)
            nk, nv = map(lambda t: repeat(t, '(h d) -> b h 1 d', b = b, h = h), (nk, nv))
            k = torch.cat((nk, k), dim = -2)
            v = torch.cat((nv, v), dim = -2)

        # derive query key similarities

        sim = einsum('b h i d, b h j d -> b h i j', q, k)

        # masking

        mask_value = -torch.finfo(sim.dtype).max

        if exists(mask):
            if exists(self.null_kv):
                mask = F.pad(mask, (1, 0), value = True)

            mask = rearrange(mask, 'b j -> b 1 1 j')
            sim = sim.masked_fill(~mask, mask_value)

        if self.causal:
            i, j = sim.shape[-2:]
            causal_mask = torch.ones(i, j, device = device, dtype = torch.bool).triu(j - i + 1)
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
        chunk_size,
        **kwargs
    ):
        super().__init__()
        self.chunk_size = chunk_size
        self.cross_attn = Attention(null_kv = True, **kwargs)

    def forward(self, x, *, context_mask = None, context, pos_emb = None):
        # derive variables
        chunk_size = self.chunk_size

        b, n, num_chunks, num_retrieved = x.shape[0], x.shape[-2], *context.shape[-4:-2]

        # if sequence length less than chunk size, do an early return

        if n < self.chunk_size:
            return torch.zeros_like(x)

        # causal padding

        causal_padding = chunk_size - 1

        x = F.pad(x, (0, 0, -causal_padding, causal_padding), value = 0.)

        # remove sequence which is ahead of the neighbors retrieved (during inference)

        seq_index = (n // chunk_size) * chunk_size
        x, x_remainder = x[:, :seq_index], x[:, seq_index:]

        seq_remain_len = x_remainder.shape[-2]

        # take care of rotary positional embedding
        # make sure queries positions are properly shifted to the future

        q_pos_emb, k_pos_emb = pos_emb
        q_pos_emb = F.pad(q_pos_emb, (0, 0, -causal_padding, causal_padding), value = 0.)

        k_pos_emb = repeat(k_pos_emb, 'b h n d -> b h (r n) d', r = num_retrieved)
        pos_emb = (q_pos_emb, k_pos_emb)

        # reshape so we have chunk to chunk attention, without breaking causality

        x = rearrange(x, 'b (k n) d -> (b k) n d', k = num_chunks)
        context = rearrange(context, 'b k r n d -> (b k) (r n) d')

        if exists(context_mask):
            context_mask = rearrange(context_mask, 'b k r n -> (b k) (r n)')

        # cross attention

        out = self.cross_attn(x, context = context, mask = context_mask, pos_emb = pos_emb)

        # reshape back to original sequence

        out = rearrange(out, '(b k) n d -> b (k n) d', b = b)

        # pad back to original, with 0s at the beginning (which will be added to the residual and be fine)

        out = F.pad(out, (0, 0, causal_padding, -causal_padding + seq_remain_len), value = 0.)
        return out

# encoder and decoder classes

class Encoder(nn.Module):
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
        cross_attn_layers = None,
        post_norm = False,
        scale_residual = 1.
    ):
        super().__init__()
        self.layers = nn.ModuleList([])

        rotary_emb_dim = max(dim_head // 2, MIN_DIM_HEAD)
        self.rotary_pos_emb = RotaryEmbedding(rotary_emb_dim)

        wrapper = partial(PreNorm, dim) if not post_norm else partial(PostNorm, dim, scale_residual = scale_residual)

        for layer_num in range(1, depth + 1):
            has_cross_attn = not exists(cross_attn_layers) or layer_num in cross_attn_layers

            self.layers.append(nn.ModuleList([
                wrapper(Attention(dim = dim, dim_head = dim_head, heads = heads, dropout = attn_dropout, causal = causal)),
                wrapper(Attention(dim = dim, dim_head = dim_head, heads = heads, dropout = attn_dropout)) if has_cross_attn else None,
                wrapper(FeedForward(dim = dim, mult = ff_mult, dropout = ff_dropout)),
            ]))

        self.norm_out = RMSNorm(dim) if final_norm and not post_norm else nn.Identity()

    def forward(self, x, *, mask = None, chunked_seq):
        device, chunk_size, seq_len = x.device, x.shape[-2], chunked_seq.shape[-2]

        q_pos_emb = self.rotary_pos_emb(chunk_size, device = device)
        k_pos_emb = self.rotary_pos_emb(seq_len, device = device)

        for attn, cross_attn, ff in self.layers:
            x = attn(x, mask = mask, pos_emb = q_pos_emb)

            if exists(cross_attn):
                x = cross_attn(x, context = chunked_seq, pos_emb = (q_pos_emb, k_pos_emb))

            x = ff(x)

        return self.norm_out(x)

class Decoder(nn.Module):
    def __init__(
        self,
        dim,
        *,
        depth,
        heads = 8,
        dim_head = 64,
        attn_dropout = 0.,
        ff_mult = 4,
        ff_dropout = 0.,
        final_norm = True,
        cross_attn_layers = None,
        chunk_size = 64,
        post_norm = False,
        scale_residual = 1.
    ):
        super().__init__()
        self.layers = nn.ModuleList([])

        rotary_emb_dim = max(dim_head // 2, MIN_DIM_HEAD)
        self.rotary_pos_emb = RotaryEmbedding(rotary_emb_dim)

        wrapper = partial(PreNorm, dim) if not post_norm else partial(PostNorm, dim, scale_residual = scale_residual)

        self.chunk_size = chunk_size

        for layer_num in range(1, depth + 1):
            has_cross_attn = not exists(cross_attn_layers) or layer_num in cross_attn_layers

            self.layers.append(nn.ModuleList([
                wrapper(Attention(dim = dim, dim_head = dim_head, heads = heads, dropout = attn_dropout, causal = True)),
                wrapper(ChunkedCrossAttention(chunk_size = chunk_size, dim = dim, dim_head = dim_head, heads = heads, dropout = attn_dropout)) if has_cross_attn else None,
                wrapper(FeedForward(dim = dim, mult = ff_mult, dropout = ff_dropout)),
            ]))

        self.norm_out = RMSNorm(dim) if final_norm and not post_norm else nn.Identity()

    def forward(self, x, *, context_mask = None, retrieved = None):
        device, seq_len = x.device, x.shape[-2]
        self_attn_pos_emb = self.rotary_pos_emb(seq_len, device = device)

        if exists(retrieved):
            num_chunks, num_neighbors, chunk_size = retrieved.shape[-4:-1]

            cross_attn_q_pos_emb = self.rotary_pos_emb(self.chunk_size, device = device, offset = self.chunk_size - 1)  # need to add extra chunk size, since it will be shifted
            cross_attn_k_pos_emb = self.rotary_pos_emb(chunk_size, device = device)

            cross_attn_pos_emb = (cross_attn_q_pos_emb, cross_attn_k_pos_emb)

        for attn, cross_attn, ff in self.layers:
            x = attn(x, pos_emb = self_attn_pos_emb)

            if exists(cross_attn) and exists(retrieved):
                x = cross_attn(
                    x,
                    context = retrieved,
                    context_mask = context_mask,
                    pos_emb = cross_attn_pos_emb
                )

            x = ff(x)

        return self.norm_out(x)

# main class

class RETRO(nn.Module):
    def __init__(
        self,
        *,
        num_tokens = BERT_VOCAB_SIZE,
        max_seq_len = 2048,
        enc_dim = 896,
        enc_depth = 2,
        enc_cross_attn_layers = None,
        dec_depth = 12,
        dec_cross_attn_layers = (1, 3, 6, 9),
        heads = 8,
        dec_dim = 768,
        dim_head = 64,
        enc_attn_dropout = 0.,
        enc_ff_dropout = 0.,
        dec_attn_dropout = 0.,
        dec_ff_dropout = 0.,
        chunk_size = 64,
        pad_id = 0,
        enc_scale_residual = 1.,
        dec_scale_residual = 1.,
        post_norm = False
    ):
        super().__init__()
        assert dim_head >= MIN_DIM_HEAD, f'dimension per head must be greater than {MIN_DIM_HEAD}'
        self.seq_len = max_seq_len
        self.pad_id = pad_id

        self.token_emb = nn.Embedding(num_tokens, enc_dim)
        self.pos_emb = nn.Embedding(max_seq_len, enc_dim)

        self.chunk_size = chunk_size

        self.to_decoder_model_dim = nn.Linear(enc_dim, dec_dim) if enc_dim != dec_dim else nn.Identity()
        self.encoder_output_to_decoder_dim = nn.Linear(enc_dim, dec_dim) if enc_dim != dec_dim else nn.Identity()

        self.encoder = Encoder(
            dim = enc_dim,
            depth = enc_depth,
            attn_dropout = enc_attn_dropout,
            ff_dropout = enc_ff_dropout,
            cross_attn_layers = enc_cross_attn_layers,
            post_norm = post_norm,
            scale_residual = enc_scale_residual
        )

        self.decoder = Decoder(
            dim = dec_dim,
            depth = dec_depth,
            attn_dropout = dec_attn_dropout,
            ff_dropout = dec_ff_dropout,
            cross_attn_layers = dec_cross_attn_layers,
            chunk_size = chunk_size,
            post_norm = post_norm,
            scale_residual = dec_scale_residual
        )

        self.to_logits = nn.Linear(dec_dim, num_tokens)

    def forward_without_retrieval(
        self,
        seq
    ):
        # embed sequence

        embed = self.token_emb(seq)
        embed = embed[:, :self.seq_len]

        # get absolute positional embedding

        pos_emb = self.pos_emb(torch.arange(embed.shape[1], device = embed.device))
        pos_emb = rearrange(pos_emb, 'n d -> 1 n d')
        embed = embed + pos_emb

        embed = self.to_decoder_model_dim(embed)
        embed = self.decoder(embed)

        # project to logits

        return self.to_logits(embed)

    def forward(
        self,
        seq,
        retrieved = None,
        return_loss = False
    ):
        """
        b - batch
        n - sequence length / chunk length
        k - number of chunks
        d - feature dimension
        r - num retrieved neighbors
        """

        if not exists(retrieved):
            return self.forward_without_retrieval(seq)

        assert not (return_loss and not self.training), 'must be training if returning loss'

        # assume padding token id (usually 0.) is to be masked out

        mask = retrieved != self.pad_id

        # handle some user inputs

        if retrieved.ndim == 3:
            retrieved = rearrange(retrieved, 'b k n -> b k 1 n') # 1 neighbor retrieved

        # if training, derive labels

        if return_loss:
            seq, labels = seq[:, :-1], seq[:, 1:]

        # variables

        n, num_chunks, num_neighbors, chunk_size, retrieved_shape, device = seq.shape[-1], *retrieved.shape[-3:], retrieved.shape, seq.device

        assert chunk_size >= self.chunk_size, 'chunk size of retrieval input must be greater or equal to the designated chunk_size on RETRO initialization'

        num_seq_chunks = n // self.chunk_size
        assert num_chunks == num_seq_chunks, f'sequence requires {num_seq_chunks} retrieved chunks, but only {num_chunks} passed in'

        # sequence index at which k-nearest neighbors have not been fetched yet after

        seq_index = num_seq_chunks * self.chunk_size

        # embed both sequence and retrieved chunks

        embed = self.token_emb(seq)
        retrieved = self.token_emb(retrieved)

        # get absolute positional embedding

        pos_emb = self.pos_emb(torch.arange(n, device = device))
        pos_emb = rearrange(pos_emb, 'n d -> 1 n d')
        embed = embed + pos_emb

        # handle masks for encoder and decoder, if needed

        encoder_retrieved_mask = decoder_retrieved_mask = None

        if exists(mask):
            assert mask.shape == retrieved_shape, 'retrieval mask must be of the same shape as the retrieval tokens'
            encoder_retrieved_mask = rearrange(mask, 'b k r n -> (b k r) n')
            decoder_retrieved_mask = mask

        # encode

        retrieved = rearrange(retrieved, 'b k r n d -> (b k r) n d')
        embed_as_context = repeat(embed[:, :seq_index], 'b (k n) d -> (b k r) n d', n = self.chunk_size, r = num_neighbors)

        retrieved = self.encoder(retrieved, mask = encoder_retrieved_mask, chunked_seq = embed_as_context)
        retrieved = rearrange(retrieved, '(b k r) n d -> b k r n d', k = num_chunks, r = num_neighbors)

        # project both sequence embedding and retrieved embedding to decoder dimension if necessary

        embed = self.to_decoder_model_dim(embed)
        retrieved = self.encoder_output_to_decoder_dim(retrieved)

        # decode

        embed = self.decoder(embed, context_mask = decoder_retrieved_mask, retrieved = retrieved)

        # project to logits

        logits = self.to_logits(embed)

        if not return_loss:
            return logits

        # cross entropy loss

        loss = F.cross_entropy(rearrange(logits, 'b n c -> b c n'), labels, ignore_index = self.pad_id)
        return loss
