""" Utility functions for text preprocessing."""

import re
import string

from gensim.parsing.preprocessing import STOPWORDS as GENSIM_STOP_WORDS

# Gensim stop word list is larger than scikit-learn & nltk stop wrods, but contains word "computer"
TC_STOP_WORDS = GENSIM_STOP_WORDS - {'computer'}

def remove_url(s):
    """ Remove url from given string s."""
    url_regex = r'(http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+)'
    return re.sub(url_regex, '', s)

def remove_punctuation(s: str):
    """ Remove punctuation from given string s"""
    return s.translate(s.maketrans({p: None for p in string.punctuation + '[‘’“”…]'}))

def remove_digits(s: str):
    """ Remove decimal digits or words containing decimal digits from given string s"""
    return re.sub(r'\w*\d\w*', '', s)

def remove_stop_words_from_str(s: str, stop_words=TC_STOP_WORDS, delimiter=' '):
    """ Remove the stop words using stop word list in Gensim"""
    return delimiter.join([word for word in s.split() if word.lower() not in stop_words])

def tokenize_str(s, min_len=2, max_len=20):
    """ Tokenize a string and return a list of str delimited by whitespace.
        Remove words too short (less than 2) or to long (greater than 20)
    """
    return [word for word in s.split() if min_len < len(word) < max_len]
