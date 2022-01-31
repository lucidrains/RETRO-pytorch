from functools import partial
import numpy as np
import torch
from torch.utils.data import Dataset

from retro_pytorch.retrieval import EOS_ID
from retro_pytorch.utils import memmap

# knn to retrieved chunks

def knn_to_retrieved_chunks(
    knns,
    chunks_memmap,
    *,
    add_continuations,
    num_chunks,
    pad_id = 0,
    eos_id = EOS_ID,
):

    # derive mask for no neighbors found (-1)

    no_neighbor_mask = knns == -1
    knns = np.maximum(knns, 0)

    # get neighbor and continuation chunks

    knn_chunks = chunks_memmap[knns]
    is_last_document_chunk = np.any(knn_chunks == eos_id, axis = -1, keepdims = True)

    # use presence of [EOS] in chunk as way to detect document boundaries
    # [EOS] in BERT tokenizer is 102

    retrieved = knn_chunks[..., :-1]

    if add_continuations:
        continuation_indices = np.clip(knns + 1, 0, num_chunks - 1) # chunks are stored contiguously
        continuation_chunks = chunks_memmap[continuation_indices][..., :-1]
        continuation_chunks *= ~is_last_document_chunk

        # combine neighbors with continuations

        retrieved = np.concatenate((retrieved, continuation_chunks), axis = -1)

    # mask out any nearest neighbor chunks that was -1 (not found at index time) to padding id

    retrieved = np.where(~no_neighbor_mask[..., None], retrieved, pad_id)
    return retrieved

# dataset

class RETRODataset(Dataset):
    def __init__(
        self,
        *,
        num_chunks,
        chunk_size,
        seq_len,
        num_sequences,
        num_neighbors,
        chunk_memmap_path,
        chunk_nn_memmap_path,
        seq_memmap_path,
        eos_id = EOS_ID,
        pad_id = 0.,
        add_continuations = True
    ):
        super().__init__()
        self.num_chunks = num_chunks
        self.num_sequences = num_sequences
        self.seq_num_chunks = seq_len // chunk_size
        self.eos_id = eos_id
        self.pad_id = pad_id

        num_chunks_with_padding = num_chunks + self.seq_num_chunks

        chunks_shape = (num_chunks_with_padding, chunk_size + 1)
        knn_shape = (num_chunks_with_padding, num_neighbors)

        self.add_continuations = add_continuations
        self.get_chunks = partial(memmap, chunk_memmap_path, dtype = np.int32, shape = chunks_shape)
        self.get_knns = partial(memmap, chunk_nn_memmap_path, dtype = np.int32, shape = knn_shape)
        self.get_seqs = partial(memmap, seq_memmap_path, dtype = np.int32, shape = (num_sequences,))

    def __len__(self):
        return self.num_sequences

    def __getitem__(self, ind):
        with self.get_chunks() as chunks_memmap, self.get_knns() as knns_memmap, self.get_seqs() as seqs_memmap:
            begin_chunk_index = seqs_memmap[ind]
            chunk_range = slice(begin_chunk_index, (begin_chunk_index + self.seq_num_chunks))

            chunks = chunks_memmap[chunk_range]

            # excise the last token, except for last token of last chunk

            seq_tokens = np.concatenate((chunks[:, :-1].flatten(), chunks[-1, -1:]))

            # mask out (with padding tokens) any token following an <eos> | disallow having more than 1 document in a sequence, as it would break RETRO's CCA

            seq_mask = np.cumsum(seq_tokens == self.eos_id, axis = 0)
            seq_mask = np.pad(seq_mask, (1, 0))[:-1] == 0.
            seq_tokens = np.where(seq_mask, seq_tokens, 0.)

            # derive retrieved tokens

            knns = knns_memmap[chunk_range]

            retrieved = knn_to_retrieved_chunks(
                knns,
                chunks_memmap,
                add_continuations = self.add_continuations,
                eos_id = self.eos_id,
                num_chunks = self.num_chunks
            )

        seq_tokens_torch = torch.from_numpy(seq_tokens).long()
        retrieved_torch = torch.from_numpy(retrieved).long()
        return seq_tokens_torch, retrieved_torch
