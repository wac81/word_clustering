# -*- coding: utf-8 -*-

# Author: Chris Moody <chrisemoody@gmail.com>
# License: MIT

# This simple example loads the newsgroups data from sklearn
# and train an LDA-like model on it
import logging
import pickle

from sklearn.datasets import fetch_20newsgroups
import numpy as np

from lda2vec import preprocess, Corpus
import jieba
import codecs
logging.basicConfig()

mode = 'chinese'
# mode = 'english'

# word2vec_path = '/Users/taha/Desktop/Code_stuff/Project_Better_World/Algo_engineering/'


if mode == 'chinese':
    # Fetch Chinese data
    texts = []
    line_count_limit = 1000
    fp = codecs.open('../data/weibo_lihang.txt', 'r', encoding='utf8')  # 文本文件，输入需要提取主题的文档
    stopwords = [u'的', u'吗', u'我', u'会', u'使用', u'跟', u'了', u'有', u'什么', u'这个', u'下', u'或者', u'能', u'要', u'怎么', u'呢', u'吧', u'都']  # 取出停用词
    count = 0
    for line in fp:
        count += 1
        if count == line_count_limit:
            break
        line = ' '.join(list(jieba.cut(line[2:])))
        # texts.append([w for w in line if w not in stopwords])  #clean
        texts.append(line)
else:
    # Fetch data
    remove = ('headers', 'footers', 'quotes')
    texts = fetch_20newsgroups(subset='train', remove=remove).data
    # Remove tokens with these substrings
    bad = set(["ax>", '`@("', '---', '===', '^^^'])


def clean_chinese(line):
    return ' '.join(w for w in line.split() if w not in stopwords)

def clean(line):
    return ' '.join(w for w in line.split() if not any(t in w for t in bad))

# Preprocess data
max_length = 10000   # Limit of 10k words per document
# Convert to unicode (spaCy only works with unicode)
if mode =='chinese':
    texts = [clean_chinese(d) for d in texts]
else:
    texts = [unicode(clean(d)) for d in texts]


data, vocab = preprocess.tokenize(texts, max_length, merge=False,
                                    n_threads=4)
n_words = len(vocab)
corpus = Corpus()
# Make a ranked list of rare vs frequent words
corpus.update_word_count(data)
corpus.finalize()
# The tokenization uses spaCy indices, and so may have gaps
# between indices for words that aren't present in our dataset.
# This builds a new compact index
compact = corpus.to_compact(data)
# Remove extremely rare words
pruned = corpus.filter_count(compact, min_count=10)
# Convert the compactified arrays into bag of words arrays
#bow = corpus.compact_to_bow(pruned)
# Words tend to have power law frequency, so selectively
# downsample the most prevalent words
#clean = corpus.subsample_frequent(pruned)
# Now flatten a 2D array of document per row and word position
# per column to a 1D array of words. This will also remove skips
# and OoV words
doc_ids = np.arange(pruned.shape[0])
flattened, (doc_ids,) = corpus.compact_to_flat(pruned, doc_ids)
# I changed skip from -1 to 1 and oov from -2 to 0 to use uint64; must reajust indexing
flattened = flattened - 2 
assert flattened.min() >= 0
# Fill in the pretrained word vectors

#TO DO : find a way to use embedding
# fn_wordvc = 'GoogleNews-vectors-negative300.bin'
# filename= word2vec_path + fn_wordvc
#vectors, s, f = corpus.compact_word_vectors(vocab,n_words=n_words) #removed filename from params
# Save all of the preprocessed files
pickle.dump(vocab, open('vocab.pkl', 'wb'))
pickle.dump(corpus, open('corpus.pkl', 'wb'))
np.save("flattened", flattened)
np.save("doc_ids", doc_ids)
np.save("pruned", pruned)
#np.save("bow", bow)
#np.save("vectors", vectors)
