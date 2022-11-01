from typing import List, Any
from random import random

import itertools
import re
from collections import Counter
import math

### TO DELETE!!!
import tqdm

def tokenize(text):
    except_list = ".?!"
    p = re.compile(fr"[^\w'+{except_list}+']")

    tokens = p.split(text.lower())
    return list(filter(None, tokens))


def train(
        train_texts: List[str],
        train_labels: List[str],
        pretrain_params: Any = None) -> Any:
    """
    Trains classifier on the given train set represented as parallel lists of texts and corresponding labels.
    :param train_texts: a list of texts (str objects), one str per example
    :param train_labels: a list of labels, one label per example
    :param pretrain_params: parameters that were learned at the pretrain step
    :return: learnt parameters, or any object you like (it will be passed to the classify function)
    """

    # ############################ REPLACE THIS WITH YOUR CODE #############################
    texts_tokenized = [tokenize(text) for text in train_texts]
    
    train_set = list(zip(texts_tokenized, train_labels))
    
    pos = [text for text, label in train_set if label == 'pos']
    neg = [text for text, label in train_set if label == 'neg']
    
    pos_words = list(itertools.chain.from_iterable(pos))
    neg_words = list(itertools.chain.from_iterable(neg))
    
    pos_words_counter = Counter(pos_words)
    neg_words_counter = Counter(neg_words)
    
    pos_words_overall_count = sum(pos_words_counter.values())
    neg_words_overall_count = sum(neg_words_counter.values())
    
    pos_docs_count = len(pos)
    neg_docs_count = len(neg)
    
    vocab_size = len(pos_words_counter.keys()) + len(neg_words_counter.keys())
    
    out_params = {'pos_words_counter': pos_words_counter,
                  'neg_words_counter': neg_words_counter,
                  'pos_words_overall_count': pos_words_overall_count,
                  'neg_words_overall_count': neg_words_overall_count,
                  'vocab_size': vocab_size,
                  'pos_docs_count': pos_docs_count,
                  'neg_docs_count': neg_docs_count}
    return out_params


def pretrain(texts_list: List[List[str]]) -> Any:
    """
    Pretrain classifier on unlabeled texts. If your classifier cannot train on unlabeled data, skip this.
    :param texts_list: a list of list of texts (str objects), one str per example.
        It might be several sets of texts, for example, train and unlabeled sets.
    :return: learnt parameters, or any object you like (it will be passed to the train function)
    """
    # ############################ PUT YOUR CODE HERE #######################################
    return None


def classify_single(text, params):
    text_tokenized = tokenize(text)
    
    pos_words_counter = params['pos_words_counter']
    neg_words_counter = params['neg_words_counter']
    pos_words_overall_count = params['pos_words_overall_count']
    neg_words_overall_count = params['neg_words_overall_count']
    pos_docs_count = params['pos_docs_count']
    neg_docs_count = params['neg_docs_count']
    
    vocab_size = params['vocab_size']
    
    vocab = pos_words_counter.keys() | neg_words_counter.keys()
    unique_text_words = set(text_tokenized)
    
    # calc pos-neg probability
    N = len(text_tokenized)
    log_fact_n = math.log(math.factorial(N))

    pos_probability = log_fact_n
    neg_probability = log_fact_n
    
    for word in unique_text_words:
        k_w = text_tokenized.count(word)
        log_fact_k_w = math.log(math.factorial(k_w))
        
        p_w_pos = (pos_words_counter[word] + 1) / (pos_words_overall_count + vocab_size)
        pos_probability += k_w * math.log(p_w_pos) - log_fact_k_w
            
        p_w_neg = (neg_words_counter[word] + 1) / (neg_words_overall_count + vocab_size)
        neg_probability += k_w * math.log(p_w_neg) - log_fact_k_w
        
    pos_probability += math.log(pos_docs_count / (pos_docs_count + neg_docs_count))
    neg_probability += math.log(neg_docs_count / (pos_docs_count + neg_docs_count))
    
    if pos_probability > neg_probability:
        return 'pos'
    else:
        return 'neg'


def classify(texts: List[str], params: Any) -> List[str]:
    """
    Classify texts given previously learnt parameters.
    :param texts: texts to classify
    :param params: parameters received from train function
    :return: list of labels corresponding to the given list of texts
    """
       
    # ############################ REPLACE THIS WITH YOUR CODE #############################
    pred_s = []
    for text in tqdm.tqdm(texts):
        pred = classify_single(text, params)
        pred_s.append(pred)
    return pred_s
    # ############################ REPLACE THIS WITH YOUR CODE #############################
