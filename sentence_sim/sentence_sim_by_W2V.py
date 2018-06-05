# -*- coding: utf-8 -*-
import numpy as np
from gensim.models import Word2Vec
import jieba
import math

PI = np.pi

def sentncevector(sentence, w2v_model):
    sent_vec = []
    for w in jieba.lcut(sentence):
        if w in w2v_model.wv:
            sent_vec.append(w2v_model.wv[w])

    # print sent[0]
    if len(sent_vec) == 0:
        sent_vec = [w2v_model.wv[u'是'] for x in range(256)]
    sent_vec = np.array(sent_vec)
    sent_vec = np.sum(sent_vec, axis=0) / float(sent_vec.shape[0])  # 列相加平均
    # print sent.shape
    return sent_vec

def compute_sim(sentence_u, sentence_v):
    '''
    get  Similar score from sentence_u and sentence_v
    sim = 1 - arccos(u.v/||u|| ||v||)/pi
    :param sentence_u:
    :param sentence_v:
    :return:
    '''

    linalg_u = np.linalg.norm(sentence_u, ord=2)
    linalg_v = np.linalg.norm(sentence_v, ord=2)

    # sim_score = 1 - np.arccos(np.dot(sentence_u, sentence_v)/(linalg_u*linalg_v))/PI
    sim_score = 1 - math.acos(np.dot(sentence_u, sentence_v)/(linalg_u*linalg_v))/PI  #math.acos is very faster

    return sim_score

if __name__ == '__main__':
    # import tensorflow_hub as hub
    # embd = hub.Module("https://tfhub.dev/google/nnlm-en-dim128/1")
    # embedding = embd(["the quick brown fox jumps over the lazy dog."])

    fname = '/media/wac/backup/nlp_models/baike_lemma_w2v_20180313/w2v/model/model.w2v'

    w2v_model = Word2Vec.load(fname, mmap='r')
    s1 = '为什么我们不能去报复别人。真的想不通，为什么那么多人要求我们学会宽容？'
    s2 = '文中也给出了样例，有了一个初始训练好的模型，用户可以根据自己的需求再加入一些文本进行再次训练。'
    s1_vec = sentncevector(s1, w2v_model)
    s2_vec = sentncevector(s2, w2v_model)

    sim = compute_sim(s1_vec, s2_vec)
    print sim