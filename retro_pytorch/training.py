import numpy as np
from functools import partial
import json
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from retro_pytorch import RETRO, RETRODataset
from retro_pytorch.data import knn_to_retrieved_chunks
from retro_pytorch.optimizer import get_optimizer
from retro_pytorch.retrieval import text_folder_to_chunks_, chunks_to_precalculated_knn_, bert_embed, SOS_ID, EOS_ID
from retro_pytorch.utils import memmap, is_true_env_flag

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

def top_k(logits, thres = 0.9):
    num_logits = logits.shape[-1]
    k = max(int((1 - thres) * num_logits), 1)
    val, ind = torch.topk(logits, k)
    probs = torch.full_like(logits, float('-inf'))
    probs.scatter_(1, ind, val)
    return probs

def top_p(logits, thres = 0.9):
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

    sorted_indices_to_remove = cum_probs > (1 - thres)
    sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
    sorted_indices_to_remove[:, 0] = 0

    sorted_logits[sorted_indices_to_remove] = float('-inf')
    return sorted_logits.scatter(1, sorted_indices, sorted_logits)

# function that returns knn chunks from seq chunks
#
# 1. adds sos and eos to seq chunks
# 2. embeds the seq chunks with special tokens with frozen BERT
# 3. fetches the knn indices with faiss
# 4. gets the knn chunks as well as the continuation from a reference to the chunks data (memmap)
#

def knn_chunks_from_seq_chunks(
    seq_chunks,
    *,
    knn,
    faiss_index,
    num_chunks,
    chunk_size,
    chunks_memmap_path,
):
    b, device = seq_chunks.shape[0], seq_chunks.device

    # prepare last chunk with sos and eos tokens for BERT embed

    ones = torch.ones((b, 1), dtype = torch.bool, device = device)
    sos = ones * SOS_ID
    eos = ones * EOS_ID

    seq_chunks = torch.cat((sos, seq_chunks, eos), dim = 1)

    # embed with frozen BERT

    embeds = bert_embed(seq_chunks.cpu()) # fetch embeds on CPU for now

    # retrieval of knn with faiss

    _, knn_indices = faiss_index.search(embeds.numpy(), k = knn)

    # numpy to torch

    with memmap(chunks_memmap_path, dtype = np.int32, shape = (num_chunks + 1, chunk_size + 1)) as chunk_memmap:
        knn_chunks = knn_to_retrieved_chunks(
            knn_indices,
            chunk_memmap,
            add_continuations = True,
            num_chunks = num_chunks
        )

        knn_chunks_torch = torch.from_numpy(knn_chunks).to(device)

    return knn_chunks_torch

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
        processed_stats_json_path = './processed-stats.json',
        faiss_index_filename = 'knn.index',
        **index_kwargs
    ):
        super().__init__()
        assert isinstance(retro, RETRO), 'retro must be instance of RETRO'
        self.retro = retro

        force_reprocess = is_true_env_flag('REPROCESS')

        # store the processed training data statistics
        # number of chunks, number of sequences

        stats_path = Path(processed_stats_json_path)

        # if the statistics file does not exist, process folders of text
        # force reprocess by setting REPROCESS=1 when running training script

        if not stats_path.exists() or force_reprocess:
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
        else:
            print(f'found to be previously processed at {str(stats_path)}')
            self.stats = json.loads(stats_path.read_text())

        # get number of chunks and number of sequences

        num_chunks = self.stats['chunks']
        num_seqs = self.stats['seqs']

        # calculate knn memmap path and get the faiss index
        # todo - make sure if faiss_index_filename is found, do not reprocess unless flag is given

        knn_memmap_path, faiss_index = chunks_to_precalculated_knn_(
            num_chunks = num_chunks,
            chunk_size = chunk_size,
            chunk_memmap_path = chunks_memmap_path,
            doc_ids_memmap_path = doc_ids_memmap_path,
            num_nearest_neighbors = knn,
            num_extra_neighbors = knn_extra_neighbors,
            index_file = faiss_index_filename,
            force_reprocess = force_reprocess,
            **index_kwargs
        )

        # retro dataset

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

        self.chunk_size = chunk_size
        self.max_seq_len = self.retro.seq_len

        self.fetch_knn_chunks_fn = partial(
            knn_chunks_from_seq_chunks,
            knn = knn,
            chunk_size = chunk_size,
            num_chunks = num_chunks,
            chunks_memmap_path = chunks_memmap_path,
            faiss_index = faiss_index
        )

    @torch.no_grad()
    @eval_decorator
    def generate(
        self,
        start = None,
        retrieved = None,
        filter_fn = top_k,
        filter_thres = 0.9,
        temperature = 1.0,
    ):
        assert filter_fn in {top_k, top_p}, 'filter function must be either top-k or nucleus'

        device = next(self.retro.parameters()).device

        # if not prime tokens given, assume sampling from SOS token with batch size of 1

        if not exists(start):
            start = torch.full((1, 1), SOS_ID, device = device).long()

        b, start_seq_len = start.shape

        # move onto same device as RETRO

        start = start.to(device)

        # prepare retrieval related variables

        if start_seq_len >= self.chunk_size:
            seq_index = (start_seq_len // self.chunk_size) * self.chunk_size
            past_seq_chunks = rearrange(start[:, :seq_index], 'b (n c) -> (b n) c', c = self.chunk_size)

            retrieved = self.fetch_knn_chunks_fn(past_seq_chunks)
            retrieved = rearrange(retrieved, '(b n) k c -> b n k c', b = b)

        # get starting sequence index

        out = start

        # sampling loop

        for i in range(start_seq_len - 1, self.max_seq_len):

            logits = self.retro(out, retrieved = retrieved)
            logits = logits[:, i]

            logits = filter_fn(logits, thres = filter_thres)
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

                knn_chunks = self.fetch_knn_chunks_fn(last_chunk)

                # concat retrieved knn chunks to all retrieved
                # to be sent to Retro for chunked cross attention at the next iteration

                knn_chunks = rearrange(knn_chunks, 'b k r -> b 1 k r')
                retrieved = safe_cat(retrieved, knn_chunks, dim = 1)

                print(f'retrieved at {curr_seq_len} / {self.max_seq_len}')

        return out

    def get_dataloader(self, **kwargs):
        return DataLoader(self.ds, **kwargs)

    def get_optimizer(self, **kwargs):
        return get_optimizer(self.retro.parameters(), **kwargs)

    def forward(self):
        raise NotImplemented
