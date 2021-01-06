""" Build baseline model."""
import os
import string
from pprint import pprint

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn.preprocessing import LabelEncoder

from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk.tokenize import WordPunctTokenizer

from gensim.models import LdaModel
from gensim.corpora import Dictionary

from tc_data import TopCoder
import preprocessing_util as P

TC = TopCoder()

def clean_and_tokenize(doc):
    """ clean and tokenize an input document."""
    lemmatizer = WordNetLemmatizer()

    tc_stopwords = set(stopwords.words('english'))
    tc_stopwords.update(('project', 'overview', 'final', 'submission', 'documentation', 'provid', 'submission', 'deliverables'))

    word_only_doc = P.remove_digits(P.remove_punctuation(P.remove_url(doc.lower())))
    lemmatized_doc = ' '.join([lemmatizer.lemmatize(word) for word in word_only_doc.split()])
    clean_doc = P.remove_stop_words_from_str(lemmatized_doc, stop_words=tc_stopwords)

    return P.tokenize_str(clean_doc, min_len=-1, max_len=10000)

def train_lda_model():
    """ Train the LDA model with topcoder selected challenges requirements."""
    print('Start processing doc.')
    clean_docs = [clean_and_tokenize(doc) for doc in TC.get_filtered_requirements().requirements.tolist()]
    dictionary = Dictionary(clean_docs)
    
    corpus = [dictionary.doc2bow(doc) for doc in clean_docs]

    print('Training LDA...')
    lda = LdaModel(corpus=corpus, num_topics=10, id2word=dictionary, passes=50)
    lda.save('./baseline/ptma_lda.model')

    pprint(lda.top_topics(corpus))

def get_lda_ditribution(doc):
    """ Get lda probability."""
    lda = LdaModel.load('./baseline/ptma_lda.model')
    dictionary = Dictionary([doc])
    corpus = dictionary.doc2bow(doc)

    lda_dist_vec = np.zeros(10)
    for idx, prob in lda[corpus]:
        lda_dist_vec[idx] = prob

    return lda_dist_vec

def predict_target():
    """ Predicting targets using LogisticRegerssion"""
    req = TC.get_filtered_requirements()
    clean_req = req.requirements.apply(clean_and_tokenize)
    req_lda_dist = clean_req.apply(get_lda_ditribution)

    X = pd.DataFrame(req_lda_dist.tolist(), index=req_lda_dist.index)

    for target in ('total_prize', 'avg_score', 'number_of_registration', 'sub_reg_ratio'):
        print(f'Predicting target: {target} ...')
        y = TC.get_filtered_challenge_info()[target]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)

        le = LabelEncoder()
        le.fit(y_train)
        y_train_encode = le.transform(y_train)

        clf = LogisticRegression(C=1.0, tol=1e-6)
        fitted_clf = clf.fit(X_train.to_numpy(), y_train_encode)
        
        y_pred_encode = clf.predict(X_test.to_numpy())
        y_pred = le.inverse_transform(y_pred_encode)

        print('Simple test MMRE: {}'.format(np.mean(np.abs(y_pred - y_test.to_numpy()) / y_test.to_numpy())))

        le_all = LabelEncoder()
        le_all.fit(y)
        y_encode = le_all.transform(y)
        y_pred_all_encode = cross_val_predict(fitted_clf, X.to_numpy(), y_encode, cv=2)
        y_pred_all = le_all.inverse_transform(y_pred_all_encode)

        result_df = y.to_frame()
        result_df[f'{target}_pred'] = y_pred_all
        result_df.to_json(f'./baseline/{target}_result.json', orient='index')
        print('Finish predicting.\n')

if __name__ == "__main__":
    # train_lda_model()
    predict_target()
