import os
import gc
import pickle
import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from transformers import AutoTokenizer, AutoModel


SCRIPT_DIR = os.path.dirname(__file__)
CORPUS_FILE_NAME = 'corpus.pkl'
VECTORS_FILE_NAME = 'vector_passages.pkl'

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

tokenizer = AutoTokenizer.from_pretrained('intfloat/multilingual-e5-large', cache_dir='cache')
model = AutoModel.from_pretrained('intfloat/multilingual-e5-large', cache_dir='cache').to(device)


def average_pool(last_hidden_states: Tensor,
                 attention_mask: Tensor) -> Tensor:
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]


def vectorize(texts, prefix='passage: '):
    assert isinstance(texts, list) and isinstance(texts[0], str), 'texts должен быть списком строк (list of str)'
    assert prefix in ('passage: ', 'query: '), 'prefix должен быть либо "passage: ", либо "query: "'
    prefixed = list(map(lambda x: prefix + x, texts))
    batch_dict = tokenizer(prefixed, max_length=512, padding=True, truncation=True, return_tensors='pt')

    for k in batch_dict.keys():
        batch_dict[k] = batch_dict[k].to(device)

    with torch.no_grad():
        outputs = model(**batch_dict)
        embeddings = average_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
        embeddings = F.normalize(embeddings, p=2, dim=1)
    del outputs

    ret = embeddings.cpu().detach().numpy()
    del embeddings

    gc.collect()
    torch.cuda.empty_cache()

    return ret


if __name__ == '__main__':

    import sys

    def text_batches(corpus, batch_size):
        for i in range(0, len(corpus), batch_size):
            yield corpus[i:i+batch_size]

    if device.type == 'cpu':
        print('cuda is unavailable :(')
    elif device.type == 'cuda':
        print('cuda is available :)')

    with open(os.path.join(SCRIPT_DIR, CORPUS_FILE_NAME), 'rb') as fp:
        corpus = pickle.load(fp)

    vectorized_passages = []
    for batch in text_batches(corpus, batch_size=16):
        vectorized_passages.append(vectorize(batch, prefix='passage: '))

    passages_arr = np.concatenate(vectorized_passages, axis=0)

    with open(os.path.join(SCRIPT_DIR, VECTORS_FILE_NAME), 'wb') as fp:
        pickle.dump(passages_arr, fp)

    sys.exit(0)


with open(os.path.join(SCRIPT_DIR, CORPUS_FILE_NAME), 'rb') as fp:
    corpus = pickle.load(fp)
    # print(len(corpus))
with open(os.path.join(SCRIPT_DIR, VECTORS_FILE_NAME), 'rb') as fp:
    passages_arr = pickle.load(fp)
    # print(passages_arr.shape[0])

assert len(corpus) == passages_arr.shape[0], f'Количество предложений в corpus.pkl ({len(corpus)}) не совпадает с количеством эмбеддингов в vector_passages.pkl ({passages_arr.shape[0]})'


def search(query, n=10):
    vectorized_query = vectorize([query,], prefix='query: ')
    scores = np.matmul(vectorized_query, passages_arr.T)[0]
    top_indices = np.argpartition(-scores, n-1)[:n]
    sorted_top_indices = top_indices[np.argsort(-scores[top_indices])]
    results = [corpus[i] for i in sorted_top_indices]
    ret_scores = scores[sorted_top_indices]
    return results, ret_scores


__all__ = ['device', 'vectorize', 'corpus', 'passages_arr', 'search']