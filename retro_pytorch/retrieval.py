import torch
from einops import rearrange

MODEL = None
TOKENIZER = None

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

def tokenize(texts):
    if not isinstance(texts, (list, tuple)):
        texts = [texts]

    tokenizer = get_tokenizer()

    encoding = tokenizer.batch_encode_plus(
        texts,
        padding = True,
        add_special_tokens = True,
        return_attention_mask = True,
        return_tensors = 'pt'
    )

    token_ids = encoding.input_ids
    mask = encoding.attention_mask
    return token_ids, mask

def bert_embed(
    token_ids,
    mask = None,
    return_cls_repr = False,
    eps = 1e-8
):
    model = get_bert()

    outputs = model(
        input_ids = token_ids,
        attention_mask = None,
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
