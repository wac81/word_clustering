# -*- coding: utf-8 -*-

from featurize import Lda2VecFeaturizer
import re
import jieba
import jieba.analyse
import codecs
from sklearn.cluster import KMeans
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
txt1 = "cat dog animal pet lion tiger zebra monkey donkey cow buffalo pig goat sheep elephant fox horse bear "
txt2 = "computer mobile electronics chip graphic display memory cpu desktop science technology silicon google"
txt3 = "world president politics minister poverty trade business election vote republic bill petition"

save_filename = 'word2vec.model'
docs = [jieba.lcut(txt1), jieba.lcut(txt2), jieba.lcut(txt3)]

vectorSize = 100

model_word = Word2Vec(docs, size=vectorSize, window=5, min_count=3, workers=4)
# model_word.save_word2vec_format
model_word.wv.save_word2vec_format(save_filename, binary=True)
model = Lda2VecFeaturizer(word2vec_path=save_filename, n_topics=3)

dat, msgs = model.train(docs, epochs=5)

test = [unicode("science iphone electronics this and that is good"),unicode("politician money business bad is this and that")]

d, m = model.infer(test, epochs=5)
