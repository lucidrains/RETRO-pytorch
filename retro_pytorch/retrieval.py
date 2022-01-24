from pathlib import Path
from shutil import rmtree

import torch
import numpy as np

from einops import rearrange

import faiss
from autofaiss import build_index
from retro_pytorch.utils import memmap, reset_folder_

# constants

SOS_ID = 101
EOS_ID = 102
BERT_MODEL_DIM = 768
BERT_VOCAB_SIZE = 30522

TMP_PATH = Path('./.tmp')
EMBEDDING_TMP_SUBFOLDER = 'embeddings'

# singleton globals

MODEL = None
TOKENIZER = None

# functions

def exists(val):
    return val is not None

def get_tokenizer():
    global TOKENIZER
    if not exists(TOKENIZER):
        TOKENIZER = torch.hub.load('huggingface/pytorch-transformers', 'tokenizer', 'bert-base-cased')
    return TOKENIZER

def get_bert():
    global MODEL
    if not exists(MODEL):
        MODEL = torch.hub.load('huggingface/pytorch-transformers', 'model', 'bert-base-cased')
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

def range_chunked(max_value, *, batch_size):
    counter = 0
    while counter < max_value:
        curr = counter + batch_size
        curr = min(curr, max_value)
        yield slice(counter, curr)
        counter = curr

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
        chunks = chunks[:, :-1] # omit last token

        for dim_slice in range_chunked(num_chunks, batch_size = batch_size):
            batch_chunk_npy = chunks[dim_slice]

            batch_chunk = torch.from_numpy(batch_chunk_npy)

            cls_tokens = torch.full((batch_chunk.shape[0], 1), SOS_ID)
            batch_chunk = torch.cat((cls_tokens, batch_chunk), dim = 1)

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
    index_folder = TMP_PATH / '.index'
    embeddings_path = TMP_PATH / embeddings_folder
    reset_folder_(index_folder)

    index_path = str(index_folder / index_file)

    build_index(
        embeddings = str(embeddings_path),
        index_path = index_path,
        index_infos_path = str(index_folder / index_infos_file),
        max_index_memory_usage = max_index_memory_usage,
        current_memory_available = current_memory_available
    )

    index = faiss.read_index(index_path, faiss.IO_FLAG_MMAP | faiss.IO_FLAG_READ_ONLY)
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
    use_cls_repr = False,
    max_rows_per_file = 500,
    chunks_to_embeddings_batch_size = 16,
    embed_dim = BERT_MODEL_DIM,
    **index_kwargs
):

    index, embeddings = chunks_to_index_and_embed(
        num_chunks = num_chunks,
        chunk_size = chunk_size,
        chunk_memmap_path = chunk_memmap_path
    )

    chunk_path = Path(chunk_memmap_path)
    knn_path = chunk_path.parents[0] / f'{chunk_path.stem}.knn{chunk_path.suffix}'

    with memmap(knn_path, shape = (num_chunks, num_nearest_neighbors), dtype = np.int32, mode = 'w+') as knns:
        for dim_slice in range_chunked(num_chunks, batch_size = max_rows_per_file):
            query_vector = embeddings[dim_slice]

            _, indices = index.search(query_vector, k = num_nearest_neighbors + 1)

            indices_without_self = indices[:, 1:]
            knns[dim_slice] = indices_without_self

            print(f'knns calculated for {dim_slice.stop} / {num_chunks}')

    print(f'knn saved to {knn_path}')
