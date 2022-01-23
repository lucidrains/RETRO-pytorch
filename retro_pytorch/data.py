import numpy as np
import torch
from torch.utils.data import Dataset

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

        self.chunks = np.memmap(chunk_memmap_path, dtype = np.int32, shape = shape)
        self.masks = np.memmap(mask_memmap_path, dtype = np.bool, shape = shape)
        self.continuations = np.memmap(chunk_continuation_memmap_path, dtype = np.int32, shape = (num_chunks,))
        self.knns = np.memmap(chunk_nn_memmap_path, dtype = np.int32, shape = (num_chunks, num_neighbors))
        self.seqs = np.memmap(seq_memmap_path, dtype = np.int32, shape = (num_sequences,))

    def __len__(self):
        return self.num_sequences

    def __getitem__(self, ind):
        begin_chunk_index = self.seqs[ind]
        chunk_range = slice(begin_chunk_index, (begin_chunk_index + self.seq_num_chunks))

        chunks = self.chunks[chunk_range]

        # excise the last token, except for last token of last chunk
        seq_tokens = np.concatenate((chunks[:, :-1].flatten(), chunks[-1, -1:]))

        # derive retrieved tokens
        knns = self.knns[chunk_range]
        continuations = self.continuations[knns]

        # get neighbor and continuation chunks
        knn_chunks = self.chunks[knns]
        knn_masks = self.masks[knns]

        continuation_shape = continuations.shape
        continuations = continuations.flatten()

        continuation_chunks = self.chunks[continuations]
        continuation_masks = self.masks[continuations]

        continuation_chunks = continuation_chunks.reshape(*continuation_shape, -1)
        continuation_masks = continuation_masks.reshape(*continuation_shape, -1)

        # combine neighbors with continuations

        retrieved = np.concatenate((knn_chunks[..., :-1], continuation_chunks[..., :-1]), axis = -1)
        retrieved_masks = np.concatenate((knn_masks[..., :-1], continuation_masks[..., :-1]), axis = -1)

        return torch.from_numpy(seq_tokens).long(), torch.from_numpy(retrieved).long(), torch.from_numpy(retrieved_masks)
