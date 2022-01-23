from functools import partial
import numpy as np
import torch
from torch.utils.data import Dataset

from retro_pytorch.retrieval import EOS_ID
from retro_pytorch.utils import memmap

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
        eos_id = EOS_ID
    ):
        super().__init__()
        self.num_chunks = num_chunks
        self.num_sequences = num_sequences
        self.seq_num_chunks = seq_len // chunk_size
        self.eos_id = eos_id

        shape = (num_chunks, chunk_size + 1)

        self.get_chunks = partial(memmap, chunk_memmap_path, dtype = np.int32, shape = shape)
        self.get_knns = partial(memmap, chunk_nn_memmap_path, dtype = np.int32, shape = (num_chunks, num_neighbors))
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

            # derive retrieved tokens

            knns = knns_memmap[chunk_range]

            # get neighbor and continuation chunks

            knn_chunks = chunks_memmap[knns][..., :-1]

            # use presence of [EOS] in chunk as way to detect document boundaries
            # [EOS] in BERT tokenizer is 102

            is_last_document_chunk = np.any(knn_chunks == self.eos_id, axis = -1, keepdims = True)

            continuation_indices = np.clip(knns + 1, 0, self.num_chunks - 1) # chunks are stored contiguously
            continuation_chunks = chunks_memmap[continuation_indices][..., :-1]
            continuation_chunks *= ~is_last_document_chunk

            # combine neighbors with continuations

            retrieved = np.concatenate((knn_chunks, continuation_chunks), axis = -1)

        return torch.from_numpy(seq_tokens).long(), torch.from_numpy(retrieved).long()
