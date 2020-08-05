""" Re-do word2vec embedding for the filetered challenge requirements."""
import os
import json
import re
from pprint import pprint
from datetime import datetime
from collections import defaultdict

import numpy as np
import pandas as pd

from gensim.models import Word2Vec
from gensim.models.keyedvectors import KeyedVectors
from sklearn.manifold import TSNE

from preprocessing_util import remove_punctuation, remove_digits, remove_url, remove_stop_words_from_str, tokenize_str
from tc_data import TopCoder

def train_w2v_hyperparam():
    """ Build *ordered* bag of words from text corpus."""
    tc = TopCoder()
    req = tc.get_filtered_requirements() # no overview extraction
    sentences = [tokenize_str(remove_stop_words_from_str(remove_punctuation(remove_digits(r.lower())))) for cha_id, r in req.itertuples()]

    for epochs in range(5, 51, 5): # [5, 10, 15, ..., 49, 50]
        for window in range(5, 21, 5): #[5, 10, 15, 20]:
            for init_lr in (0.025, 0.02, 0.01, 0.002): # some random learning rate
                print('Hyper param:')
                pprint({'epochs': epochs, 'window': window, 'initial_learning_rate': init_lr})

                print('Training Word2Vec model', end='|', flush=True)
                model = Word2Vec(sentences=sentences, alpha=init_lr, window=window, min_count=10, iter=epochs, sg=1, hs=1, seed=42, min_alpha=2e-5, workers=8)
                
                print('Decomposing vectors', end='|', flush=True)
                vectors = np.asarray([model.wv[word] for word in model.wv.vocab])
                labels = np.asarray([word for word in model.wv.vocab])

                tsne = TSNE(n_components=2, init='pca', random_state=42, perplexity=50, n_iter=5000)
                reduced_vec = tsne.fit_transform(vectors)

                print('Saving decomposed vectors')
                fp = os.path.join(os.curdir, 'result', 'word2vec', f'w2v-epochs{epochs}-window{window}-init_lr{init_lr}.json')
                pd.DataFrame.from_dict({'label': labels, 'x': reduced_vec[:, 0], 'y': reduced_vec[:, 1]}, orient='columns').to_json(fp, orient='index')

if __name__ == "__main__":
    train_w2v_hyperparam()
