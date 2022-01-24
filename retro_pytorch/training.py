import torch
from torch import nn
from torch.utils.data import DataLoader
from retro_pytorch.retrieval import text_folder_to_chunks_, chunks_to_precalculated_knn_
from retro_pytorch import RETRO, RETRODataset
from retro_pytorch.optimizer import get_optimizer

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

        knn_memmap_path = chunks_to_precalculated_knn_(
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

    def get_dataloader(self, **kwargs):
        return DataLoader(self.ds, **kwargs)

    def get_optimizer(self, **kwargs):
        return get_optimizer(self.retro.parameters(), **kwargs)

    def forward(self):
        raise NotImplemented
