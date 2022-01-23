from functools import partial
import numpy as np
import torch
from torch.utils.data import Dataset
from contextlib import contextmanager

@contextmanager
def memmap(*args, **kwargs):
    pointer = np.memmap(*args, **kwargs)
    yield pointer
    del pointer

# dataset

class RETRODataset(Dataset):
    def __init__(
        self,
        *,
        num_chunks,
        chunk_size,
        num_sequences,
        num_neighbors,
        chunk_memmap_path,
        chunk_continuation_memmap_path,
        chunk_nn_memmap_path,
        seq_memmap_path,
        mask_memmap_path = None,
        seq_len = 2048
    ):
        super().__init__()
        self.num_sequences = num_sequences
        self.seq_num_chunks = seq_len // chunk_size
        shape = (num_chunks, chunk_size + 1)

        self.get_chunks = partial(memmap, chunk_memmap_path, dtype = np.int32, shape = shape)
        self.get_masks = partial(memmap, mask_memmap_path, dtype = np.bool, shape = shape)
        self.get_continuations = partial(memmap, chunk_continuation_memmap_path, dtype = np.int32, shape = (num_chunks,))
        self.get_knns = partial(memmap, chunk_nn_memmap_path, dtype = np.int32, shape = (num_chunks, num_neighbors))
        self.get_seqs = partial(memmap, seq_memmap_path, dtype = np.int32, shape = (num_sequences,))

    def __len__(self):
        return self.num_sequences

    def __getitem__(self, ind):
        with self.get_chunks() as chunks_memmap, self.get_masks() as masks_memmap, self.get_continuations() as continuations_memmap, self.get_knns() as knns_memmap, self.get_seqs() as seqs_memmap:

            begin_chunk_index = seqs_memmap[ind]
            chunk_range = slice(begin_chunk_index, (begin_chunk_index + self.seq_num_chunks))

            chunks = chunks_memmap[chunk_range]

            # excise the last token, except for last token of last chunk
            seq_tokens = np.concatenate((chunks[:, :-1].flatten(), chunks[-1, -1:]))

            # derive retrieved tokens
            knns = knns_memmap[chunk_range]
            continuations = continuations_memmap[knns]

            # get neighbor and continuation chunks
            knn_chunks = chunks_memmap[knns]
            knn_masks = masks_memmap[knns]

            continuation_chunks = chunks_memmap[continuations]
            continuation_masks = masks_memmap[continuations]

            # combine neighbors with continuations

            retrieved = np.concatenate((knn_chunks[..., :-1], continuation_chunks[..., :-1]), axis = -1)
            retrieved_masks = np.concatenate((knn_masks[..., :-1], continuation_masks[..., :-1]), axis = -1)

        return torch.from_numpy(seq_tokens).long(), torch.from_numpy(retrieved).long(), torch.from_numpy(retrieved_masks)
