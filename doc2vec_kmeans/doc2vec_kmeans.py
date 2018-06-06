# -*- coding: utf-8 -*-

import sys
import gensim
import numpy as np

from gensim.models.doc2vec import Doc2Vec
from sklearn.cluster import KMeans

TaggededDocument = gensim.models.doc2vec.TaggedDocument


def get_datasest(line_count_limit=1000):
    count = 0
    docs = []
    # with open("../data/cache-msgs.txt", 'r') as f:
    #     print f.readline()
    # for line in open("../data/weibo_lihang.txt", 'r'):
    for line in open("../data/cache-msgs.txt"):

        # delNOTNeedWords()
        count += 1
        docs = line.split('\r')
        # docs.append(line)

        if count == line_count_limit:
            break
    print len(docs)

    x_train = []
    # y = np.concatenate(np.ones(len(docs)))
    for i, text in enumerate(docs):
        word_list = text.split(' ')
        l = len(word_list)
        word_list[l - 1] = word_list[l - 1].strip()
        document = TaggededDocument(word_list, tags=[i])
        x_train.append(document)

    return x_train


def train(x_train, size=200, epoch_num=1):
    model_dm = Doc2Vec(x_train, min_count=1, window=3, size=size, sample=1e-3, negative=5, workers=4)
    model_dm.train(x_train, total_examples=model_dm.corpus_count, epochs=100)
    model_dm.save('model_dm.vec')

    return model_dm


def cluster(x_train):
    infered_vectors_list = []
    print "load doc2vec model..."
    model_dm = Doc2Vec.load("model_dm.vec")
    print "load train vectors..."
    i = 0
    for text, label in x_train:
        vector = model_dm.infer_vector(text)
        infered_vectors_list.append(vector)
        i += 1

    print "train kmean model..."
    kmean_model = KMeans(n_clusters=15)
    s = kmean_model.fit(infered_vectors_list)

    # 获取到所有句向量所属类别
    labels = kmean_model.labels_

    # 把是一类的放入到一个集合
    classCollects = {}
    for i in range(len(x_train)):
        if labels[i] in classCollects.keys():
            classCollects[labels[i]].append(x_train[i][0])
        else:
            classCollects[labels[i]] = [x_train[i][0]]

    print classCollects

    # #预测并写前100句
    # labels = kmean_model.predict(infered_vectors_list[0:100])
    #预测并写所有
    labels = kmean_model.predict(infered_vectors_list)

    cluster_centers = kmean_model.cluster_centers_


    '''
    两种写入方式
    '''
    with open("own_classify.txt", 'w') as wf:
        for i in range(len(infered_vectors_list)):
            string = ""
            text = x_train[i][0]
            for word in text:
                string = string + word
            string = string + '\t'
            string = string + str(labels[i])
            string = string + '\n'
            wf.write(string)

    for i in range(len(infered_vectors_list)):
        with open("../data/sentence2vec/" + str(labels[i]) + '.csv', 'a+') as f:
            string = ""
            text = x_train[i][0]
            for word in text:
                string = string + word
            string = string + '\t'
            string = string + str(labels[i])
            string = string + '\n'
            f.write(string)

    return cluster_centers


if __name__ == '__main__':
    x_train = get_datasest()
    # model_dm = train(x_train)
    cluster_centers = cluster(x_train)