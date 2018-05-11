# Author: Chris Moody <chrisemoody@gmail.com>
# License: MIT

# This simple example loads the newsgroups data from sklearn
# and train an LDA-like model on it
import os
import os.path
import pickle

import numpy as np

from lda2vec.model import LDA2Vec
import tensorflow as tf

data_dir = os.getenv('data_dir', '../lda2vec_clustering/') #py2
fn_vocab = '{data_dir:s}/vocab.pkl'.format(data_dir=data_dir)
fn_corpus = '{data_dir:s}/corpus.pkl'.format(data_dir=data_dir)
fn_flatnd = '{data_dir:s}/flattened.npy'.format(data_dir=data_dir)
fn_docids = '{data_dir:s}/doc_ids.npy'.format(data_dir=data_dir)
#fn_vectors = '{data_dir:s}/vectors.npy'.format(data_dir=data_dir)
vocab = pickle.load(open(fn_vocab, 'rb'))
corpus = pickle.load(open(fn_corpus, 'rb'))
flattened = np.load(fn_flatnd)
doc_ids = np.load(fn_docids)
#vectors = np.load(fn_vectors)


# Model Parameters
# Number of documents
n_docs = doc_ids.max() + 1
# Number of unique words in the vocabulary
n_vocab = flattened.max() + 1

model = LDA2Vec(n_documents=n_docs, n_vocab=n_vocab, d_hyperparams={},
                freqs=None, w_in=None, fixed_words=False, word2vec_only=False,
                meta_graph=None, save_graph_def=True, log_dir="./log")

# 'Strength' of the dircihlet prior; 200.0 seems to work well
clambda = 200.0
# Number of topics to fit
n_topics = int(os.getenv('n_topics', 20))
batchsize = 4096
# Power for neg sampling
power = float(os.getenv('power', 0.75))
# Intialize with pretrained word vectors
pretrained = bool(int(os.getenv('pretrained', True)))
# Sampling temperature
temperature = float(os.getenv('temperature', 1.0))
# Number of dimensions in a single word vector
n_units = int(os.getenv('n_units', 300))
# Get the string representation for every compact key
words = corpus.word_list(vocab)[:n_vocab]
# How many tokens are in each document
doc_idx, lengths = np.unique(doc_ids, return_counts=True)
doc_lengths = np.zeros(doc_ids.max() + 1, dtype='int32')
doc_lengths[doc_idx] = lengths
# Count all token frequencies
tok_idx, freq = np.unique(flattened, return_counts=True)
term_frequency = np.zeros(n_vocab, dtype='int32')
term_frequency[tok_idx] = freq

for key in sorted(locals().keys()):
    val = locals()[key]
    if len(str(val)) < 100 and '<' not in str(val):
        print(key, val)


# model.train(doc_ids, flattened, vocab, words, loss_switch_epochs=0, max_epochs=np.inf, save=True, save_every=50, outdir="./out",
#             summarize=True, summarize_every=50, metadata="metadata.tsv", metadata_docs="metadata.docs.tsv") #added vocab


# sess=tf.Session()
# #First let's load meta graph and restore weights
# saver = tf.train.import_meta_graph('./log_180503_1444/180503_1444_lda2vec-15403100.meta')
# saver.restore(sess, tf.train.latest_checkpoint('./log_180503_1444/'))

model = LDA2Vec(meta_graph='./log_180503_1444/180503_1444_lda2vec-15403100')
model.compute_similarity(doc_ids, [0], )