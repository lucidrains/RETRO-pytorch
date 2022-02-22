from pathlib import Path
from math import ceil

import torch
import torch.nn.functional as F
import logging
import numpy as np
from einops import rearrange

import faiss
from autofaiss import build_index

from retro_pytorch.utils import memmap, reset_folder_

# constants

SOS_ID = 101
EOS_ID = 102
BERT_MODEL_DIM = 768
BERT_VOCAB_SIZE = 28996

TMP_PATH = Path('./.tmp')
INDEX_FOLDER_PATH = TMP_PATH / '.index'
EMBEDDING_TMP_SUBFOLDER = 'embeddings'

# helper functions

def exists(val):
    return val is not None

def range_chunked(max_value, *, batch_size):
    counter = 0
    while counter < max_value:
        curr = counter + batch_size
        curr = min(curr, max_value)
        yield slice(counter, curr)
        counter = curr

# indexing helper functions

def faiss_read_index(path):
    return faiss.read_index(str(path), faiss.IO_FLAG_MMAP | faiss.IO_FLAG_READ_ONLY)

# singleton globals

MODEL = None
TOKENIZER = None

def get_tokenizer():
    global TOKENIZER
    if not exists(TOKENIZER):
        TOKENIZER = torch.hub.load('huggingface/pytorch-transformers', 'tokenizer', 'bert-base-cased')
    return TOKENIZER

def get_bert():
    global MODEL
    if not exists(MODEL):
        MODEL = torch.hub.load('huggingface/pytorch-transformers', 'model', 'bert-base-cased')
        if torch.cuda.is_available():
            MODEL = MODEL.cuda()

    return MODEL

# tokenize

def tokenize(texts, add_special_tokens = True):
    if not isinstance(texts, (list, tuple)):
        texts = [texts]

    tokenizer = get_tokenizer()

    encoding = tokenizer.batch_encode_plus(
        texts,
        add_special_tokens = add_special_tokens,
        padding = True,
        return_tensors = 'pt'
    )

    token_ids = encoding.input_ids
    return token_ids

# text to chunks

def doc_text_to_chunks_and_seq_indices(
    *,
    doc_text,
    chunk_size = 64,
    seq_len = 2048,
    pad_id = 0
):
    assert (seq_len % chunk_size) == 0, 'sequence length must be divisible by chunk size'

    ids = tokenize(doc_text)
    ids = rearrange(ids, '1 ... -> ...')

    text_len = ids.shape[-1]

    # pad to multiple of chunk size with an extra token

    padding = chunk_size - ((text_len - 1) % chunk_size)
    ids = F.pad(ids, (0, padding))

    # split out very last token

    ids, last_token = ids[:-1], ids[-1:]
    ids = rearrange(ids, '(n c) -> n c', c = chunk_size)

    # first tokens of chunk [2:] and on will become the last token of chunk [1:]

    last_token_per_chunk = ids[1:, 0]
    all_last_tokens = torch.cat((last_token_per_chunk, last_token), dim = 0)
    all_last_tokens = rearrange(all_last_tokens, 'n -> n 1')

    # append all last tokens to ids for (num_chunks, chunk_size + 1)

    chunks_with_extra_token = torch.cat((ids, all_last_tokens), dim = -1)

    # calculate chunk indices starting at 0, spaced number of chunks of seq len apart

    total_chunks = ids.shape[0]
    num_chunks_per_seq = seq_len // chunk_size
    seq = torch.arange(0, total_chunks, num_chunks_per_seq)

    return chunks_with_extra_token, seq

def text_folder_to_chunks_(
    *,
    folder,
    chunks_memmap_path,
    seqs_memmap_path,
    doc_ids_memmap_path,
    chunk_size = 64,
    seq_len = 2048,
    glob = '**/*.txt',
    max_chunks = 1_000_000,
    max_seqs = 100_000
):
    paths = sorted([*Path(folder).glob(glob)])

    total_chunks = 0
    total_docs = 0
    total_seqs = 0

    chunks_shape = (max_chunks, chunk_size + 1)
    seqs_shape = (max_seqs,)
    doc_ids_shape = (max_chunks,)

    with memmap(chunks_memmap_path, shape = chunks_shape, dtype = np.int32, mode = 'w+') as chunks_memmap\
        , memmap(seqs_memmap_path, shape = seqs_shape, dtype = np.int32, mode = 'w+') as seqs_memmap\
        , memmap(doc_ids_memmap_path, shape = doc_ids_shape, dtype = np.int32, mode = 'w+') as doc_ids_memmap:

        for path in paths:
            print(f'processing {path}')

            chunks, seq = doc_text_to_chunks_and_seq_indices(
                doc_text = path.read_text(),
                chunk_size = chunk_size,
                seq_len = seq_len
            )

            doc_chunk_len = chunks.shape[0]
            doc_seq_len = seq.shape[0]

            chunks_memmap[total_chunks:(total_chunks + doc_chunk_len)] = chunks.numpy()
            seqs_memmap[total_seqs:(total_seqs + doc_seq_len)] = seq.numpy() + total_chunks
            doc_ids_memmap[total_chunks:(total_chunks + doc_chunk_len)] = np.full((doc_chunk_len,), total_docs)

            total_chunks += doc_chunk_len
            total_seqs += doc_seq_len
            total_docs += 1

    return dict(
        chunks = total_chunks,
        docs = total_docs,
        seqs = total_seqs
    )

# embedding function

@torch.no_grad()
def bert_embed(
    token_ids,
    return_cls_repr = False,
    eps = 1e-8,
    pad_id = 0.
):
    model = get_bert()
    mask = token_ids != pad_id

    if torch.cuda.is_available():
        token_ids = token_ids.cuda()

    outputs = model(
        input_ids = token_ids,
        attention_mask = mask,
        output_hidden_states = True
    )

    hidden_state = outputs.hidden_states[-1]

    if return_cls_repr:
        return hidden_state[:, 0]               # return [cls] as representation

    if not exists(mask):
        return hidden_state.mean(dim = 1)

    mask = mask[:, 1:]                          # mean all tokens excluding [cls], accounting for length
    mask = rearrange(mask, 'b n -> b n 1')

    numer = (hidden_state[:, 1:] * mask).sum(dim = 1)
    denom = mask.sum(dim = 1)
    masked_mean =  numer / (denom + eps)
    return masked_mean

# chunks to knn

def chunks_to_embeddings_(
    *,
    num_chunks,
    chunks_memmap_path,
    embeddings_memmap_path,
    chunk_size = 64,
    embed_dim = BERT_MODEL_DIM,
    batch_size = 16,
    use_cls_repr = False,
    pad_id = 0.
):
    chunks_shape = (num_chunks, chunk_size + 1)
    embed_shape = (num_chunks, embed_dim)

    with memmap(chunks_memmap_path, shape = chunks_shape, dtype = np.int32) as chunks\
        , memmap(embeddings_memmap_path, shape = embed_shape, dtype = np.float32, mode = 'w+') as embeddings:

        for dim_slice in range_chunked(num_chunks, batch_size = batch_size):
            batch_chunk_npy = chunks[dim_slice]

            batch_chunk = torch.from_numpy(batch_chunk_npy)

            cls_tokens = torch.full((batch_chunk.shape[0], 1), SOS_ID)
            batch_chunk = torch.cat((cls_tokens, batch_chunk), dim = 1)

            batch_chunk = batch_chunk[:, :-1] # omit last token, the first token of the next chunk, used for autoregressive training

            batch_embed = bert_embed(
                batch_chunk,
                return_cls_repr = use_cls_repr
            )

            embeddings[dim_slice] = batch_embed.detach().cpu().numpy()
            print(f'embedded {dim_slice.stop} / {num_chunks}')


def memmap_file_to_chunks_(
    memmap_path,
    *,
    folder,
    shape,
    dtype,
    max_rows_per_file = 500
):
    rows, _ = shape

    with memmap(memmap_path, shape = shape, dtype = dtype, mode = 'r') as f:
        root_path = TMP_PATH / folder
        reset_folder_(root_path)

        for ind, dim_slice in enumerate(range_chunked(rows, batch_size = max_rows_per_file)):
            filename = root_path / f'{ind}.npy'
            data_slice = f[dim_slice]

            np.save(str(filename), f[dim_slice])
            print(f'saved {str(filename)}')

def index_embeddings(
    embeddings_folder,
    *,
    index_file = 'knn.index',
    index_infos_file = 'index_infos.json',
    max_index_memory_usage = '100m',
    current_memory_available = '1G'
):
    embeddings_path = TMP_PATH / embeddings_folder
    index_path = INDEX_FOLDER_PATH / index_file

    reset_folder_(INDEX_FOLDER_PATH)

    build_index(
        embeddings = str(embeddings_path),
        index_path = str(index_path),
        index_infos_path = str(INDEX_FOLDER_PATH / index_infos_file),
        max_index_memory_usage = max_index_memory_usage,
        current_memory_available = current_memory_available,
        should_be_memory_mappable = True,
        use_gpu = torch.cuda.is_available(),
    )

    index = faiss_read_index(index_path)
    return index

def chunks_to_index_and_embed(
    *,
    num_chunks,
    chunk_size,
    chunk_memmap_path,
    use_cls_repr = False,
    max_rows_per_file = 500,
    chunks_to_embeddings_batch_size = 16,
    embed_dim = BERT_MODEL_DIM,
    index_file = 'knn.index',
    **index_kwargs
):
    embedding_path = f'{chunk_memmap_path}.embedded'
    embed_shape = (num_chunks, embed_dim)

    chunks_to_embeddings_(
        num_chunks = num_chunks,
        chunk_size = chunk_size,
        chunks_memmap_path = chunk_memmap_path,
        embeddings_memmap_path = embedding_path,
        use_cls_repr = use_cls_repr,
        batch_size = chunks_to_embeddings_batch_size,
        embed_dim = embed_dim
    )

    memmap_file_to_chunks_(
        embedding_path,
        shape = embed_shape,
        dtype = np.float32,
        folder = EMBEDDING_TMP_SUBFOLDER,
        max_rows_per_file = max_rows_per_file
    )

    index = index_embeddings(
        embeddings_folder = EMBEDDING_TMP_SUBFOLDER,
        index_file = index_file,
        **index_kwargs
    )

    embeddings = np.memmap(embedding_path, shape = embed_shape, dtype = np.float32, mode = 'r')
    return index, embeddings

def chunks_to_precalculated_knn_(
    *,
    num_nearest_neighbors,
    num_chunks,
    chunk_size,
    chunk_memmap_path,
    doc_ids_memmap_path,
    use_cls_repr = False,
    max_rows_per_file = 500,
    chunks_to_embeddings_batch_size = 16,
    embed_dim = BERT_MODEL_DIM,
    num_extra_neighbors = 10,
    force_reprocess = False,
    index_file = 'knn.index',
    **index_kwargs
):
    chunk_path = Path(chunk_memmap_path)
    knn_path = chunk_path.parents[0] / f'{chunk_path.stem}.knn{chunk_path.suffix}'
    index_path = INDEX_FOLDER_PATH / index_file

    # early return knn path and faiss index
    # unless if force_reprocess is True

    if index_path.exists() and knn_path.exists() and not force_reprocess:
        print(f'preprocessed knn found at {str(knn_path)}, faiss index reconstituted from {str(index_path)}')
        index = faiss_read_index(index_path)
        return knn_path, index

    # fetch the faiss index and calculated embeddings for the chunks

    index, embeddings = chunks_to_index_and_embed(
        num_chunks = num_chunks,
        chunk_size = chunk_size,
        chunk_memmap_path = chunk_memmap_path,
        index_file = index_file,
        **index_kwargs
    )

    total_neighbors_to_fetch = num_extra_neighbors + num_nearest_neighbors + 1

    with memmap(knn_path, shape = (num_chunks, num_nearest_neighbors), dtype = np.int32, mode = 'w+') as knns\
        , memmap(doc_ids_memmap_path, shape = (num_chunks,), dtype = np.int32, mode = 'r') as doc_ids:

        for dim_slice in range_chunked(num_chunks, batch_size = max_rows_per_file):
            query_vector = embeddings[dim_slice]

            distances, indices = index.search(query_vector, k = total_neighbors_to_fetch)

            # remove self from distances and indices

            distances = distances[:, 1:]
            indices = indices[:, 1:]

            # mask out any neighbors that belong to the same document to -1

            query_doc_ids = doc_ids[dim_slice]
            neighbor_doc_ids = doc_ids[indices]
            neighbor_from_same_doc = query_doc_ids[..., None] == neighbor_doc_ids

            indices = np.where(neighbor_from_same_doc, -1, indices)
            distances = np.where(neighbor_from_same_doc, 1e3, distances)

            # re-sort indices by updated distances

            indices = np.take_along_axis(indices, np.argsort(distances, axis = 1), axis = 1)

            # store nearest neighbors to knn memmap

            knns[dim_slice] = indices[:, :num_nearest_neighbors]

            print(f'knns calculated for {dim_slice.stop} / {num_chunks}')

    print(f'knn saved to {knn_path}')
    return knn_path, index
