# -*- coding: utf-8 -*-

import re
import jieba
import jieba.analyse
import codecs
from sklearn.cluster import KMeans
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence

import jieba.posseg as pseg

stopwords = codecs.open('../data/stopwords.txt', encoding='UTF-8').read()
# print stopwords

def delNOTNeedWords(content,customstopwords=None):
    # words = jieba.lcut(content)
    if customstopwords == None:
        import os
        file_stop_words = "../data/stopwords.txt"
        if os.path.exists(file_stop_words):
            stop_words = codecs.open(file_stop_words, encoding='UTF-8').read()
            customstopwords = stop_words

    result=''
    return_words = []
    # for w in words:
    #     if w not in stopwords:
    #         result += w.encode('utf-8')  # +"/"+str(w.flag)+" "  #去停用词
    words = pseg.lcut(content)

    for word, flag in words:
        # print word.encode('utf-8')
        if (word not in customstopwords and flag[0] in [u'n', u'f', u'a', u'z']):
            # ["/x","/zg","/uj","/ul","/e","/d","/uz","/y"]): #去停用词和其他词性，比如非名词动词等
            result += word.encode('utf-8')  # +"/"+str(w.flag)+" "  #去停用词
            return_words.append(word.encode('utf-8'))
    return result,return_words


def wordsCluster(text_path, line_count_limit=1000, cutwords='cutwords.txt', vectorSize=100, classCount=10):
    '''
    textUrl:输入文本的本地路径，
    fencijieguo：分词结果存储到本地路径，
    vectorSize：词向量大小，
    classCount：分类大小


    '''

    # 读取文本
    textstr = ''
    count = 0
    for line in open(text_path):
        # delNOTNeedWords()
        count += 1
        textstr += line
        if count == line_count_limit:
            break

    # 使用jieba分词

    # 分词结果放入到的文件路径
    outfenci = codecs.open(cutwords, "a+", 'utf-8')
    tempList = re.split(u'[。？！?!]', textstr)
    for row in tempList:
        if row != None and row != '':
            # 分词结果放入到文件中
            readline = ' '.join(list(jieba.cut(row, cut_all=False))) + '\n'
            outfenci.write(readline)
    outfenci.close()

    # word2vec向量化
    model = Word2Vec(LineSentence(cutwords), size=vectorSize, window=5, min_count=3, workers=4)

    # 获取model里面的所有关键词
    keys = model.wv.vocab.keys()

    # 获取词对应的词向量
    wordvector = []
    for key in keys:
        wordvector.append(model[key])

    # 聚类
    clf = KMeans(n_clusters=classCount)
    s = clf.fit(wordvector)
    print s
    # 获取到所有词向量所属类别
    labels = clf.labels_

    # 把是一类的放入到一个集合
    classCollects = {}
    for i in range(len(keys)):
        if labels[i] in classCollects.keys():
            classCollects[labels[i]].append(keys[i])
        else:
            classCollects[labels[i]] = [keys[i]]

    return classCollects

if __name__ == '__main__':
    classCollects = wordsCluster('../data/weibo_lihang.txt')

    results = {}
    for i, v in enumerate(classCollects.values()):
        temp = []
        for w in v:
            if delNOTNeedWords(w)[0].strip() == '':
                continue
            temp.append(delNOTNeedWords(w)[0])
        results[i] = temp
    print results