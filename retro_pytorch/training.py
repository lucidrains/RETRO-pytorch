import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from retro_pytorch import RETRO, RETRODataset
from retro_pytorch.data import knn_to_retrieved_chunks
from retro_pytorch.optimizer import get_optimizer
from retro_pytorch.retrieval import text_folder_to_chunks_, chunks_to_precalculated_knn_, bert_embed, SOS_ID, EOS_ID
from retro_pytorch.utils import memmap

from einops import rearrange

# helpers

def exists(val):
    return val is not None

def eval_decorator(fn):
    def inner(model, *args, **kwargs):
        was_training = model.training
        model.eval()
        out = fn(model, *args, **kwargs)
        model.train(was_training)
        return out
    return inner

def safe_cat(accum, t, dim = -1):
    if not exists(accum):
        return t
    return torch.cat((accum, t), dim = dim)

# sampling helpers

def log(t, eps = 1e-20):
    return torch.log(t.clamp(min = eps))

def gumbel_noise(t):
    noise = torch.zeros_like(t).uniform_(0, 1)
    return -log(-log(noise))

def gumbel_sample(t, temperature = 1., dim = -1):
    return ((t / temperature) + gumbel_noise(t)).argmax(dim = dim)

def top_k(logits, thres = 0.5):
    num_logits = logits.shape[-1]
    k = max(int((1 - thres) * num_logits), 1)
    val, ind = torch.topk(logits, k)
    probs = torch.full_like(logits, float('-inf'))
    probs.scatter_(1, ind, val)
    return probs

# training wrapper class

class TrainingWrapper(nn.Module):
    def __init__(
        self,
        *,
        retro,
        chunk_size,
        documents_path,
        knn,
        glob = '**/*.txt',
        chunks_memmap_path = './train.chunks.dat',
        seqs_memmap_path = './train.seq.dat',
        doc_ids_memmap_path = './train.doc_ids.dat',
        max_chunks = 1_000_000,
        max_seqs = 100_000,
        knn_extra_neighbors = 100,
        **index_kwargs
    ):
        super().__init__()
        assert isinstance(retro, RETRO), 'retro must be instance of RETRO'
        self.retro = retro

        self.stats = text_folder_to_chunks_(
            folder = documents_path,
            glob = glob,
            chunks_memmap_path = chunks_memmap_path,
            seqs_memmap_path = seqs_memmap_path,
            doc_ids_memmap_path = doc_ids_memmap_path,
            chunk_size = chunk_size,
            seq_len = retro.seq_len,
            max_chunks = max_chunks,
            max_seqs = max_seqs
        )

        num_chunks = self.stats['chunks']
        num_seqs = self.stats['seqs']

        knn_memmap_path, faiss_index = chunks_to_precalculated_knn_(
            num_chunks = num_chunks,
            chunk_size = chunk_size,
            chunk_memmap_path = chunks_memmap_path,
            doc_ids_memmap_path = doc_ids_memmap_path,
            num_nearest_neighbors = knn,
            num_extra_neighbors = knn_extra_neighbors,
            **index_kwargs
        )

        self.ds = RETRODataset(
            num_sequences = num_seqs,
            num_chunks = num_chunks,
            num_neighbors = knn,
            chunk_size = chunk_size,
            seq_len = retro.seq_len,
            chunk_memmap_path = chunks_memmap_path,
            chunk_nn_memmap_path = knn_memmap_path,
            seq_memmap_path = seqs_memmap_path
        )

        # params needed for generation

        self.knn = knn
        self.faiss_index = faiss_index
        self.max_seq_len = self.retro.seq_len
        self.num_chunks = num_chunks
        self.chunk_size = chunk_size
        self.chunks_memmap_path = chunks_memmap_path

    @torch.no_grad()
    @eval_decorator
    def generate(
        self,
        start = None,
        filter_thres = 0.9,
        temperature = 1.0
    ):
        device = next(self.retro.parameters()).device

        if not exists(start):
            start = torch.ones((1, 1), dtype = torch.bool, device = device) * SOS_ID

        b, start_seq_len = start.shape
        out = start

        # prepare retrieval related variables

        retrieved = None

        ones = torch.ones((b, 1), dtype = torch.bool, device = device)
        sos = ones * SOS_ID
        eos = ones * EOS_ID

        for i in range(start_seq_len - 1, self.max_seq_len):
            logits = self.retro(out, retrieved = retrieved)
            logits = logits[:, i]

            logits = top_k(logits, thres = filter_thres)
            sampled = gumbel_sample(logits, temperature = temperature, dim = -1)
            sampled = rearrange(sampled, 'b -> b 1')

            out = torch.cat((out, sampled), dim = 1)

            # early terminate if all EOS

            is_eos_tokens = (out == EOS_ID)

            if is_eos_tokens.any(dim = -1).all():

                # mask out everything after the eos tokens

                shifted_is_eos_tokens = F.pad(is_eos_tokens, (1, -1))
                mask = shifted_is_eos_tokens.float().cumsum(dim = -1) >= 1
                out = out.masked_fill(mask, self.retro.pad_id)
                break

            # when the sequence length is a multiple of the chunk size
            # retrieve the next set of knns

            curr_seq_len = out.shape[-1]

            if (curr_seq_len % self.chunk_size) == 0:
                last_chunk = rearrange(out, 'b (c n) -> b c n', n = self.chunk_size)[:, -1]

                # prepare last chunk with sos and eos tokens for BERT embed

                last_chunk = torch.cat((sos, last_chunk, eos), dim = 1)

                # embed with frozen BERT

                embeds = bert_embed(last_chunk.cpu()) # fetch embeds on CPU for now

                # retrieval of knn with faiss

                _, knn_indices = self.faiss_index.search(embeds.numpy(), k = self.knn)

                # numpy to torch

                with memmap(self.chunks_memmap_path, dtype = np.int32, shape = (self.num_chunks + 1, self.chunk_size + 1)) as chunk_memmap:
                    knn_chunks = knn_to_retrieved_chunks(
                        knn_indices,
                        chunk_memmap,
                        add_continuations = True,
                        num_chunks = self.num_chunks
                    )

                    knn_chunks_torch = torch.from_numpy(knn_chunks).to(device)
                    knn_chunks_torch = rearrange(knn_chunks_torch, 'b k r -> b 1 k r')

                    # concat retrieved knn chunks to all retrieved
                    # to be sent to Retro for chunked cross attention at the next iteration

                    retrieved = safe_cat(retrieved, knn_chunks_torch, dim = 1)

                print(f'retrieved at {curr_seq_len} / {self.max_seq_len}')

        return out

    def get_dataloader(self, **kwargs):
        return DataLoader(self.ds, **kwargs)

    def get_optimizer(self, **kwargs):
        return get_optimizer(self.retro.parameters(), **kwargs)

    def forward(self):
        raise NotImplemented
