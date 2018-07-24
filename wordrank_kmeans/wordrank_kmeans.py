# -*- coding: utf-8 -*-
'''
mpich
编译mpich
http://www.mpich.org/downloads/

./configure --enable-mpi-thread-multiple
make;sudo make install

sudo apt-get install mercurial
sudo apt-get install libboost-all-dev (on Ubuntu)
sudo apt install libomp-dev


./install.sh  must modify


after train
model.save("wordrank")
model.save_word2vec_format("wordrank_in_word2vec.vec")

and

model.load
model.load_word2vec_format  (not saved necessary)


'''
import re
import jieba
import jieba.analyse
import codecs
from sklearn.cluster import KMeans
from gensim.models import Word2Vec
from gensim.models.wrappers import Wordrank
from gensim.models.word2vec import LineSentence

import jieba.posseg as pseg
import jieba
import os
import shutil

PATH = os.path.dirname(__file__)

stopwords = codecs.open(os.path.join(PATH, '../data/stopwords.txt'), encoding='UTF-8').read()

path_to_wordrank = '/home/wac/PycharmProjects/wordrank'
# path_to_wordrank = '/home/wac/PycharmProjects/xuzp/wordrank'
# path_to_wordrank = '/home/wac/wordrank'

corpus_file = 'cutwords.txt'

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


def wordsCluster(text_path, line_count_limit=1000, cutwords='cutwords.txt', vectorSize=128, window=15, iter=100, classCount=20):
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
    outfenci = codecs.open(cutwords, "w+", 'utf-8')
    tempList = re.split(u'[。？！?!]', textstr)
    for row in tempList:
        if row != None and row != '':
            # 分词结果放入到文件中
            readline = ' '.join(list(jieba.cut(row, cut_all=False))) + '\n'
            outfenci.write(readline)
    outfenci.close()

    # 获取当前文件路径
    current_path = os.path.abspath(__file__)
    # 获取当前文件的父目录
    dir_path = os.path.abspath(os.path.dirname(current_path) + os.path.sep + ".")



    if os.path.exists(dir_path + '/wr_model'):
        shutil.rmtree(dir_path + '/wr_model')
    try:
        model = Wordrank.train(path_to_wordrank,
                           corpus_file = corpus_file,
                           out_name=dir_path + '/wr_model',
                           size=vectorSize,
                           window=window,
                           iter=iter)
    except AttributeError as e:
        if 'sort_embeddings' in e.message:
            model = Wordrank.load_word2vec_format(dir_path + '/wr_model/wordrank.words.w2vformat')
            pass

    # model = Wordrank.load_word2vec_format(dir_path + '/wr_model/wordrank.words.w2vformat')

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
    classCollects = wordsCluster(os.path.join(PATH, '../data/cache-msgs.txt'), line_count_limit=10000000)

    results = {}
    for i, v in enumerate(classCollects.values()):
        temp = []
        for w in v:
            if delNOTNeedWords(w)[0].strip() == '':
                continue
            temp.append(delNOTNeedWords(w)[0])
        results[i] = temp
    print results

    for k in results:
        with open("../data/word2vec/" + str(k) + '.csv', 'w+') as f:
            f.write('\r\n'.join(results[k]))

