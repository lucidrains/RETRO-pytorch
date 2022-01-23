<img src="./RETRO.png" width="500px"></img>

## RETRO - Pytorch (wip)

Implementation of <a href="https://arxiv.org/abs/2112.04426">RETRO</a>, Deepmind's Retrieval based Attention net, in Pytorch. This will deviate from the paper slightly, using rotary embeddings for relative positional encoding, as well as Faiss library instead of Scann.

This library leverages <a href="https://github.com/criteo/autofaiss">autofaiss</a> for building the index and calculating the k-nearest neighbors for all chunks.

If you are interested, please join <a href="https://discord.gg/3AvcJfbEBd">this Discord</a> for discussions

## Install

```bash
$ pip install retro-pytorch
````

## Usage

```python
import torch
from retro_pytorch import RETRO

retro = RETRO(
    chunk_size = 64,                         # the chunk size that is indexed and retrieved (needed for proper relative positions as well as causal chunked cross attention)
    max_seq_len = 2048,                      # max sequence length
    enc_dim = 896,                           # encoder model dim
    enc_depth = 2,                           # encoder depth
    dec_dim = 796,                           # decoder model dim
    dec_depth = 12,                          # decoder depth
    dec_cross_attn_layers = (3, 6, 9, 12),   # decoder cross attention layers (with causal chunk cross attention)
    heads = 8,                               # attention heads
    dim_head = 64,                           # dimension per head
    dec_attn_dropout = 0.25,                 # decoder attention dropout
    dec_ff_dropout = 0.25                    # decoder feedforward dropout
)

seq = torch.randint(0, 20000, (2, 2048 + 1))      # plus one since it is split into input and labels for training
retrieved = torch.randint(0, 20000, (2, 32, 2, 128)) # retrieved tokens - (batch, num chunks, num retrieved neighbors, retrieved chunk with continuation)

loss = retro(seq, retrieved, return_loss = True)
loss.backward()

# do above for many steps
```

## RETRO Datasets (wip)

The `RETRODataset` class accepts paths to a number of memmapped numpy arrays containing the chunks, masking, continuation, and pre-calculated neighbors. You can use this to easily assemble the data for `RETRO` training.


```python
import torch
from torch.utils.data import DataLoader
from retro_pytorch import RETRO, RETRODataset

# mock data constants

import numpy as np

NUM_CHUNKS = 1000
CHUNK_SIZE = 64
NUM_SEQS = 100
NUM_NEIGHBORS = 2

def save_memmap(path, tensor):
    f = np.memmap(path, dtype = tensor.dtype, mode = 'w+', shape = tensor.shape)
    f[:] = tensor
    del f

SHAPE = (NUM_CHUNKS, CHUNK_SIZE + 1)

# generate mock chunk data

save_memmap(
    './train.chunks.dat',
    np.int32(np.random.randint(0, 8192, size = SHAPE))
)

# generate nearest neighbors for each chunk

save_memmap(
    './train.chunks.knn.dat',
    np.int32(np.random.randint(0, 1000, size = (NUM_CHUNKS, NUM_NEIGHBORS)))
)

# generate seq data

save_memmap(
    './train.seq.dat',
    np.int32(np.random.randint(0, 128, size = (NUM_SEQS,)))
)

# instantiate dataset class
# which constructs the sequence and neighbors from memmapped chunk and neighbor information

ds = RETRODataset(
    num_sequences = NUM_SEQS,
    num_chunks = NUM_CHUNKS,
    num_neighbors = NUM_NEIGHBORS,
    chunk_size = CHUNK_SIZE,
    seq_len = 2048,
    chunk_memmap_path = './train.chunks.dat',
    chunk_nn_memmap_path = './train.chunks.knn.dat',
    seq_memmap_path = './train.seq.dat'
)

dl = iter(DataLoader(ds, batch_size = 2))

# one forwards and backwards

retro = RETRO(
    max_seq_len = 2048,                      # max sequence length
    enc_dim = 896,                           # encoder model dimension
    enc_depth = 3,                           # encoder depth
    dec_dim = 768,                           # decoder model dimensions
    dec_depth = 12,                          # decoder depth
    dec_cross_attn_layers = (1, 3, 6, 9),    # decoder cross attention layers (with causal chunk cross attention)
    heads = 8,                               # attention heads
    dim_head = 64,                           # dimension per head
    dec_attn_dropout = 0.25,                 # decoder attention dropout
    dec_ff_dropout = 0.25                    # decoder feedforward dropout
).cuda()

seq, retrieved = map(lambda t: t.cuda(), next(dl))

# seq       - (2, 2049)         - 1 extra token since split by seq[:, :-1], seq[:, 1:]
# retrieved - (2, 32, 2, 128)   - 128 since chunk + continuation, each 64 tokens

loss = retro(
    seq,
    retrieved,
    return_loss = True
)

loss.backward()

```

## Retrieval related tools (wip)

This repository will use the default tokenizer (sentencepiece) for the cased version of BERT. Embeddings will be fetched from the vanilla BERT, and can either be masked mean pooled representation, or the CLS token.

ex. masked mean pooled representation

```python
import torch
from retro_pytorch.retrieval import bert_embed, tokenize

ids, mask = tokenize([
    'hello world',
    'foo bar'
])

embeds = bert_embed(ids, mask) # (2, 768) - 768 is hidden dimension of BERT
```

ex. CLS token representation


```python
import torch
from retro_pytorch.retrieval import bert_embed, tokenize

ids, mask = tokenize([
    'hello world',
    'foo bar'
])

embeds = bert_embed(ids, mask, return_cls_repr = True) # (2, 768)
```

## Todo

- [x] handle partially filled chunks with mask
- [ ] handle indexing of corpus of text with faiss
- [ ] handle reindexing of all nearest neighbors
- [ ] function for getting frozen BERT embeddings for batch of chunks
- [ ] inference code, autoretrieving at chunk boundaries
- [ ] autohandle retrieved chunks for last chunk in sequence, whether it is given or not

## Citations

```bibtex
@misc{borgeaud2022improving,
    title   = {Improving language models by retrieving from trillions of tokens}, 
    author  = {Sebastian Borgeaud and Arthur Mensch and Jordan Hoffmann and Trevor Cai and Eliza Rutherford and Katie Millican and George van den Driessche and Jean-Baptiste Lespiau and Bogdan Damoc and Aidan Clark and Diego de Las Casas and Aurelia Guy and Jacob Menick and Roman Ring and Tom Hennigan and Saffron Huang and Loren Maggiore and Chris Jones and Albin Cassirer and Andy Brock and Michela Paganini and Geoffrey Irving and Oriol Vinyals and Simon Osindero and Karen Simonyan and Jack W. Rae and Erich Elsen and Laurent Sifre},
    year  = {2022},
    eprint = {2112.04426},
    archivePrefix = {arXiv},
    primaryClass = {cs.CL}
}
```

```bibtex
@misc{su2021roformer,
    title   = {RoFormer: Enhanced Transformer with Rotary Position Embedding},
    author  = {Jianlin Su and Yu Lu and Shengfeng Pan and Bo Wen and Yunfeng Liu},
    year    = {2021},
    eprint  = {2104.09864},
    archivePrefix = {arXiv},
    primaryClass = {cs.CL}
}
```

*I consider always the adult life to be the continuous retrieval of childhood.* - Umberto Eco
