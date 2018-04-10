# -*- coding: utf-8 -*-

import codecs
from gensim.models import LdaModel
from gensim.corpora import Dictionary
from gensim import corpora, models
import numpy as np
import jieba

line_count_limit = 1000000
def get_dict():
    train = []
    fp = codecs.open('../data/weibo_lihang.txt', 'r', encoding='utf8')#文本文件，输入需要提取主题的文档
    stopwords = ['的', '吗', '我', '会', '使用', '跟', '了', '有', '什么', '这个', '下', '或者', '能', '要', '怎么', '呢', '吧', '都']#取出停用词
    count = 0
    for line in fp:
        count += 1
        if count == line_count_limit:
            break
        line = list(jieba.cut(line))
        train.append([w for w in line if w not in stopwords])

    dictionary = Dictionary(train)
    return dictionary,train
def train_save_model():
    dictionary, train=get_dict()
    corpus = [dictionary.doc2bow(text) for text in train]
    lda = LdaModel(corpus=corpus, id2word=dictionary, num_topics=20)
    #模型的保存/ 加载
    lda.save('test_lda.model')

def lda_sim(s1,s2):
    '''
    计算两个文档的相似度
    :param s1:
    :param s2:
    :return:
    '''
    lda = models.ldamodel.LdaModel.load('test_lda.model')
    test_doc = list(jieba.cut(s1))  # 新文档进行分词
    dictionary=get_dict()[0]
    doc_bow = dictionary.doc2bow(test_doc)  # 文档转换成bow
    doc_lda = lda[doc_bow]  # 得到新文档的主题分布
    # 输出新文档的主题分布
    # print(doc_lda)
    list_doc1 = [i[1] for i in doc_lda]
    # print('list_doc1',list_doc1)

    test_doc2 = list(jieba.cut(s2))  # 新文档进行分词
    doc_bow2 = dictionary.doc2bow(test_doc2)  # 文档转换成bow
    doc_lda2 = lda[doc_bow2]  # 得到新文档的主题分布
    # 输出新文档的主题分布
    # print(doc_lda)
    list_doc2 = [i[1] for i in doc_lda2]
    # print('list_doc2',list_doc2)
    try:
        sim = np.dot(list_doc1, list_doc2) / (np.linalg.norm(list_doc1) * np.linalg.norm(list_doc2))
    except ValueError:
        sim=0
    #得到文档之间的相似度，越大表示越相近
    return sim

if __name__ == '__main__':
    train_save_model()

    d1 = u'南京长江大桥在哪'
    # d1 = u'大桥公园在桥南  而长江大桥好玩的应该在桥北  可以坐132路公交车到桥北的北堡下面 有一站靠近长江边的  在桥下玩玩  然后直接从北堡爬楼梯上长江大桥  就省了很多引桥要走的冤枉路  具体你上132公交车问问司机师傅哪里下  我忘了'
    d2 = u'长江从上海至宜宾江段共79座长江大桥（含长江隧道），自逆流而上依次是： 上海市境内1座：崇明越江通道 上海长江大桥 上海市与江苏省之间1座：崇启大桥(在建) 崇海大桥（在建） 世界最大跨度的斜拉桥——苏通大桥江苏省境内11座：苏通大桥、江阴长江大桥、泰州长江大桥(在建)、扬中长江大桥(注：未跨长江南北)、润扬长江大桥、南京长江四桥(在建)、南京长江二桥、南京长江大桥(公路铁路两用)、南京过江隧道、南京长江三桥、南京大胜关长江大桥(在建)（铁路桥）； 安徽省境内4座：马鞍山长江大桥(在建)、芜湖长江大桥(公路铁路两用)、铜陵长江大桥、安庆长江大桥； 江西与湖北省界之间2座：九江长江大桥(公路铁路两用)、九江长江公路大桥 (在建) 湖北省境内22座（含隧道）：黄石长江大桥、鄂东长江大桥、黄冈长江大桥（公路铁路两用、在建）、鄂黄长江大桥、武汉阳逻长江大桥、武汉天兴洲长江大桥(公路铁路两用)、武汉二七长江大桥（在建）、武汉长江二桥、武汉长江隧道、武汉地铁2号线过江隧道（地铁隧道）（在建）、武汉长江大桥(公路铁路两用)、武汉鹦鹉洲长江大桥（在建）、武汉白沙洲长江大桥、武汉军山长江大桥、荆州长江大桥、枝城长江大桥(公路铁路两用)、宜昌长江大桥、宜昌长江铁路大桥(铁路桥)(在建)、葛洲坝三江大桥、夷陵长江大桥、西陵长江大桥、巴东长江大桥； 湖北省与湖南省之间1座：荆岳长江大桥（在建）。 重庆市境内32座：巫山长江大桥、奉节长江大桥、云阳长江大桥、万州长江二桥、万宜铁路万州长江大桥(铁路桥)(在 万里长江第一跨——润扬长江公路大桥建)、万州长江大桥、忠县长江大桥、忠州长江大桥、丰都长江大桥、涪陵李渡长江桥、涪陵长江大桥、涪陵石板沟长江大桥(在建)、长寿长江大桥、渝怀铁路长寿长江大桥(铁路桥)、重庆鱼嘴长江大桥、广阳岛长江大桥（未跨主航道，连接广阳岛和长江南岸）、重庆大佛寺长江大桥、重庆朝天门长江大桥、重庆东水门长江大桥（在建），重庆长江大桥、重庆长江大桥复线桥、重庆菜园坝长江大桥、重庆鹅公岩长江大桥、重庆李家沱大桥、重庆鱼洞长江大桥(在建)、重庆马桑溪大桥、白沙沱大桥(铁路桥)、地维长江大桥、江津观音岩长江大桥、江津迎宾长江大桥（在建）、江津长江大桥、永川长江大桥（在建）； 四川省境内6座：泸州泰安长江大桥、泸州铁路长江大桥(铁路桥)、泸州长江二桥、泸州长江大桥、江安长江大桥(在建)、宜宾长江大桥(在建)、合江长江一桥（在建）、合江长江二桥（在建）；'
    sim_score = lda_sim(d1, d2)
    print sim_score
