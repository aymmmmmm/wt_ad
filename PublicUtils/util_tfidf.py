#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time :2017/4/25 11:02
# @author : R XIE
import math
import pandas as pd
import numpy as np
import scipy.stats


def term_freq(data, var, wordlist, dt, window):
    data[dt] = pd.to_datetime(data[dt].values)
    etime = max(data[dt].values)
    stime = etime - pd.to_timedelta('%s hours' % window)
    sdata = data.loc[(data[dt] >= stime) & (data[dt] < etime), :]
    word_freq_window = {}
    for word in wordlist:
        if word in sdata[var].values:
            if word in word_freq_window:
                word_freq_window[word] += list(sdata[var].values).count(word)
            else:
                word_freq_window[word] = list(sdata[var].values).count(word)
        else:
            if word not in word_freq_window:
                word_freq_window[word] = 0
    return list(map(lambda x: x / (sum(word_freq_window.values())+np.spacing(1)), word_freq_window.values()))


def term_freq_base(data, var):
    wordlist = pd.unique(data[var].values)
    word_freq_window = {}
    for word in wordlist:
        if word in data[var].values:
            if word in word_freq_window:
                word_freq_window[word] += list(data[var].values).count(word)
            else:
                word_freq_window[word] = list(data[var].values).count(word)
        else:
            if word not in word_freq_window:
                word_freq_window[word] = 0
    return list(map(lambda x: x / sum(word_freq_window.values()), word_freq_window.values())), wordlist


def idf_by_window(corpus, words, window):
    """
    离散化数据IDF计算
    :param corpus:语料库/全体数据集，原始数据集，dataframe/ndarray 
    :param words: 词表，array
    :param window: 滑动窗口大小，窗口下数据即定义为文档，int
    :return: 返回由词和idf组成的键值对列表，dataframe #旧版生成list
    """
    # n_of_doc = round(len(corpus)/window, 0)
    n_of_doc = 0
    i = 0
    word_in_corpus_stat = {}
    while i < len(corpus):
        if i + window < len(corpus):
            doc = corpus[i:i + window]
        else:
            doc = corpus[i:len(corpus)]
        if isinstance(doc, pd.DataFrame):
            doc = doc.values
        else:
            doc = doc
        for word in words:
            if word in doc:
                if word in word_in_corpus_stat:
                    word_in_corpus_stat[word] += 1
                else:
                    word_in_corpus_stat[word] = 1
            else:
                if word not in word_in_corpus_stat:
                    word_in_corpus_stat[word] = 0
        i = i + window + 1
        n_of_doc += 1
    # idf = [(k, math.log(n_of_doc/(1+v), 10)*100) for (k, v) in word_in_corpus_stat.items()] # list in old version
    idf = pd.DataFrame([(k, math.log(n_of_doc / (1 + v), 10) * 100) for (k, v) in word_in_corpus_stat.items()])
    idf.columns = ['Word', 'IDF']
    # print(n_of_doc)
    # print([(k, v) for (k, v) in word_in_corpus_stat.items()])
    # print(idf)
    return idf


def tfidf_by_window(corpus, var, dt, idf, window, mode='B'):
    """
    离散化数据的TF-IDF计算
    要求预训练数据的IDF向量，要求数据具有时间戳（DateTime）
    :param corpus: 输入数据，原始数据集，dataframe类型
    :param var: 计算tf-idf的变量名，str
    :param dt: 输入数据中时间戳变量名，str
    :param idf: 预训练的变量IDF向量，由idf_by_window生成，dataframe #旧版生成list
    :param window: 窗口大小，定义文档大小
    :param mode: 结果输出模式
    :return: 返回每个窗口（文档）的时间、词、TF、IDF、TF-IDF组成的dataframe. 在模式A返回的数据框中带有group变量以标示各组，以避免时间重复的情况
    """
    # word_list = list(zip(*idf))[0]
    tfidf = pd.DataFrame()
    # idf = pd.DataFrame(idf)
    # idf.columns = ['Word', 'IDF']
    word_list = list(idf['Word'].values)
    i = 0
    while i < len(corpus[var]):
        word_freq_window = {}
        if i + window < len(corpus[var]):
            doc = corpus[i:i + window]
        else:
            doc = corpus[i:len(corpus[var])]
        for word in word_list:
            if word in doc[var].values:
                if word in word_freq_window:
                    word_freq_window[word] += list(doc[var].values).count(word)
                else:
                    word_freq_window[word] = list(doc[var].values).count(word)
            else:
                if word not in word_freq_window:
                    word_freq_window[word] = 0
        if mode == 'A':
            tf = pd.DataFrame([(k, math.log(v / len(doc[var]), 10) * 100) for (k, v) in word_freq_window.items()])
            tf.columns = ['Word', 'TF']
            tf['DateTime'] = max(doc[dt])
            tfidf_doc = pd.merge(tf, idf, on='Word')
            tfidf_doc['TF-IDF'] = tfidf_doc.TF * tfidf_doc.IDF
            tfidf_doc['group'] = i
            tfidf = tfidf.append(tfidf_doc, ignore_index=True)
        if mode == 'B':
            def listplus(a, b):
                x = a * b
                return x

            tf = [v / len(doc[var]) for (k, v) in word_freq_window.items()]
            tfidf_doc_list = [list(map(listplus, tf, idf.IDF.values))]
            tfidf_doc_df = pd.DataFrame({'DateTime': max(doc[dt]), 'TF-IDF': tfidf_doc_list})
            tfidf = tfidf.append(tfidf_doc_df, ignore_index=True)
        i = i + window + 1
    return tfidf


def vector_distance_euc(a, b):
    if isinstance(a, list):
        a = np.array(a)
    if isinstance(b, list):
        b = np.array(b)
    dist = np.linalg.norm(a - b)
    return dist


def remove_stopwords(stopwords, word):
    if isinstance(word, np.ndarray):
        word = word.tolist()
    for stopw in stopwords:
        if stopw in word:
            word.remove(stopw)
    return word


def cosine_sim(vec1, vec2):
    npvec1, npvec2 = np.array(vec1), np.array(vec2)
    return npvec1.dot(npvec2) / (math.sqrt((npvec1 ** 2).sum()) * math.sqrt((npvec2 ** 2).sum()))


def mahalanobis(vec1, vec2):
    npvec1, npvec2 = np.array(vec1), np.array(vec2)
    npvec = np.array([npvec1, npvec2])
    sub = npvec.T[0] - npvec.T[1]
    inv_sub = np.linalg.inv(np.cov(npvec1, npvec2))
    return math.sqrt(np.dot(inv_sub, sub).dot(sub.T))


def kldivergence(vec1, vec2):
    return scipy.stats.entropy(vec1, vec2)


from math import log


def kld(p, q):
    p, q = zip(*filter(lambda x: x[0] != 0 or x[1] != 0, list(zip(p, q))))  # 去掉二者都是0的概率值
    p = p + np.spacing(1)
    q = q + np.spacing(1)
    # print(p, q)
    return sum([_p * log(_p / _q, 2) for (_p, _q) in zip(p, q)])


def jsd(p, q):
    p, q = zip(*filter(lambda x: x[0] != 0 or x[1] != 0, list(zip(p, q))))
    M = [0.5 * (_p + _q) for _p, _q in zip(p, q)]
    p = p + np.spacing(1)
    q = q + np.spacing(1)
    M = M + np.spacing(1)
    return 0.5 * kld(p, M) + 0.5 * kld(q, M)


def hellinger(vec1, vec2):
    return np.sqrt(1 - np.sum(np.sqrt(vec1 * vec2)))


def bhattacharyya(vec1, vec2):
    return np.log(np.sum(np.sqrt(vec1 * vec2)))


def make_dist_df(indata, compvec, dt, var, mode='B'):
    """
    以数据框存储的带时间标识的TFIDF的距离计算函数
    计算某一时间TFIDF向量与参考向量的欧氏距离
    :param indata: 输入文档TFIDF数据，由tfidf_by_window生成，dataframe
    :param compvec: 参考向量，array
    :param dt: 时间变量名，str
    :param var: 输入数据的TFIDF存储变量名，str
    :param mode: 输入数据的模式，与tfidf_by_window使用的模式参数相同，str
    :return: 每时间的与参考向量欧式距离、向量标准差数据集，dataframe
    """
    ddf = pd.DataFrame()
    if mode == 'A':
        for cdt in np.unique(indata['group']):
            # print(cdt)
            sdata = indata.ix[indata.group == cdt,]
            dist = vector_distance_euc(sdata[var].values, compvec)
            sim = cosine_sim(sdata[var].values, compvec)
            kl_d = kldivergence(sdata[var].values, compvec)
            hellinger_d = hellinger(sdata[var].values, compvec)
            std_vec = np.std(sdata[var].values)
            # print(dist)
            cdt2 = sdata[dt].values[0]
            ddf_temp = pd.DataFrame({'datetime': cdt2, 'distance': dist, 'vec_std': std_vec, 'sim': sim}, index=['0'])
            ddf = ddf.append(ddf_temp, ignore_index=True)
    if mode == 'B':
        dist = list(map(lambda x: np.linalg.norm(x - compvec), np.array(indata[var])))
        std_vec = list(map(np.std, indata[var]))
        sim = list(map(lambda y: np.array(y).dot(np.array(compvec)) / (
            math.sqrt((np.array(y) ** 2).sum()) * math.sqrt((np.array(compvec) ** 2).sum())),
                       np.array(indata[var])))
        # kl_d = list(map(lambda z: scipy.stats.entropy(z, compvec), np.array(indata[var])))
        # hellinger_d = list(
        #     map(lambda v: np.sqrt(1 - np.sum(np.sqrt(np.array(v) * np.array(compvec)))), np.array(indata[var])))
        # health = kl_d
        ddf = pd.DataFrame({'datetime': indata[dt], 'distance': dist, 'vec_std': std_vec, 'sim': sim})
        ddf = ddf.sort_values('datetime')
    return ddf
