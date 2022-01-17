<img src="./RETRO.png" width="500px"></img>

## RETRO - Pytorch (wip)

Implementation of <a href="https://arxiv.org/abs/2112.04426">RETRO</a>, Deepmind's Retrieval based Attention net, in Pytorch. This will deviate from the paper slightly, using rotary embeddings for relative positional encoding, as well as Faiss library instead of Scann.

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
    num_tokens = 20000,                      # number of tokens
    max_seq_len = 2048,                      # max sequence length
    enc_dim = 896,                           # encoder model dim
    enc_depth = 2,                           # encoder depth
    dec_depth = 12,                          # decoder depth
    dec_cross_attn_layers = (1, 3, 6, 9),    # decoder cross attention layers (with causal chunk cross attention)
    heads = 8,                               # attention heads
    dec_dim = 796,                           # decoder model dim
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

## Todo

- [ ] handle indexing of corpus of text with faiss
- [ ] handle reindexing of all nearest neighbors
- [ ] function for getting frozen BERT embeddings for batch of chunks
- [ ] check masking
- [ ] inference code, autoretrieving

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
