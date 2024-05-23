# -*- coding: UTF-8 -*-
import jieba
import  os
import json


def calculate_distinct_n(tokens, n):
    n_grams = set()
    total_n_grams = 0
    for i in range(len(tokens) - n + 1):
        n_gram = tuple(tokens[i:i + n])
        n_grams.add(n_gram)
        total_n_grams += 1
    # print(n_grams)
    distinct_n = len(n_grams) / total_n_grams if total_n_grams > 0 else 0
    return distinct_n

def get_dict(tokens, ngram, gdict=None):
    """
    get_dict
    统计n-gram频率并用dict存储
    """
    token_dict = {}
    if gdict is not None:
        token_dict = gdict
    tlen = len(tokens)
    for i in range(0, tlen - ngram + 1):
        ngram_token = "".join(tokens[i:(i + ngram)])
        if token_dict.get(ngram_token) is not None: 
            token_dict[ngram_token] += 1
        else:
            token_dict[ngram_token] = 1
    return token_dict

def calc_distinct_ngram(pair_list, ngram):
    """
    calc_distinct_ngram
    """
    ngram_total = 0.0
    ngram_distinct_count = 0.0
    pred_dict = {}
    for predict_tokens, _ in pair_list:
        get_dict(predict_tokens, ngram, pred_dict)
    for key, freq in pred_dict.items():
        ngram_total += freq
        ngram_distinct_count += 1 
        #if freq == 1:
        #    ngram_distinct_count += freq
    return ngram_distinct_count / ngram_total


if __name__ == '__main__':
    sentense = "好的，这是针对这道题的分析：【解析】测验的效度是指一个测验或量表实际测出所要测的心理特质的程度。影响测验效度的因素有系统误差、恒定效应、随机误差、随机效应、评分记分错误、测验时的生理节律、测验情境、测验时的心理状态、测验动机、测验材料的取样范围、测验材料的难度、测验材料的区分度、测验材料的信度、测验材料的长度、测验材料的编排、测验材料的时限、测验材料的编制者、测验材料的复本信度、测验材料的评分记分、测验材料的信度系数、测验材料的效标关联效度、测验材料的效标关联效度的信度、效标材料的信度、效标材料的效度、效标材料的长度、效标材料的时限、效标材料的复本信度、效标材料的评分记分、效标材料的效标关联效度、效标材料的效标关联效度的信度、效标材料的信度系数、效标材料的效标关联效度系数、效标材料的效标关联效度的信度系数、效标材料的信度系数的信度、效标材料的信度系数的效度、效标材料的信度系数的效标关联效度、效标材料的信度系数的效标关联效度的信度、效标材料的信度系数的效标关联效度的信度系数、效标材料的信度系数的效标关联效度的信度系数的信度、效标材料的信度系数的效标关联效度的信度系数的效度、效标材料的信度系数的效标关联效度的信度系数的效标关联效度、效标材料的信度系数的效标关联效度的信度系数的效标关联效度的信度、效标材料的信度系数的效标关联效度的信度系数的效标关联效度的信度系数的信度、效标材料的信度系数的效标关联效度的信度系数的效标关联效度的信度系数的效度、效标材料的信度系数的效标关联效度的信度系数的效标关联效度的信度系数的效标关联效度的信度、效标材料的信度系数的效标关联效度的信度系数的效标关联效度的信度系数的效标关联效度的信度系数的信度、效标材料的信度系数的效标关联效度的信度系数的效标关联效度的信度系数的效标关联效度的信度系数的效度、效标材料的信度系数的效标关联效度的信度系数的效标关联效度的信度系数的效标关联效度的信度系数的效标关联效度的信度、效标材料的信度系数的效标关联效度的信度系数的效标关联效度的信度系数的效标关联效度的信度系数的效标关联效度的信度系数的信度、效标材料的信度系数的效标关联效度的信度系数的效标关联效度的信度系数的效标关联效度的信度系数的效标关联效度的信度系数的效度、效标材料的信度系数的效标关联效度的信度系数的效标关联效度的信度系数的效标关联效度的信度系数的效标关联效度的信度系数的效标关联效度、效标材料的信度系数的效标关联效度的信度系数的效标关联效度的信度系数的效标关联效度的信度系数的效标关联效度的信度系数的效标关联效度的信度、效标材料的信度系数的效标关联效度的信度系数的效标关联效度的信度系数的效标关联效度的信度系数的效标关联效度的信度系数的效标关联效度的信度系数的信度、效标材料的信度系数的效标关联效度的信度系数的效标关联效度的信度系数的效标关联效度的信度系数的效标关联效度的信度系数的效标关联效度的信度系数的效度、效标材料的信度系数的效标关联效度"
    tokens = list(jieba.cut(sentense.replace(" ", "")))
    # print(tokens)
    # print(calc_distinct_ngram([(tokens, 1)], 2))
    dist_1 = calculate_distinct_n(tokens, 1)
    dist_2 = calculate_distinct_n(tokens, 2)
    dist_3 = calculate_distinct_n(tokens, 3)
    dist_4 = calculate_distinct_n(tokens, 4)
    print(dist_1, dist_2, dist_3, dist_4)
