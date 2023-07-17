#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/6/29 22:04
# @Author  : 草原上的风
# @File    : cut.py
import jieba
import jieba.posseg as psg
import logging
from settings import KEYWORDSPATH, STOPWORDSPATH
import string
import json

# 关闭jieba log输出
jieba.setLogLevel(logging.INFO)
# 加载关键词词典
jieba.load_userdict(KEYWORDSPATH)

# 单字分割,英文部分
letters = string.ascii_lowercase + "+"
# 单字分割,去除标点
filters = [",", "-", ".", " ", '?', '/']
stopwords = [stopword.strip() for stopword in open(STOPWORDSPATH, 'r', encoding='utf-8').readlines()]


def cut_sentence_by_word(sentence):
    """
    :param sentence:
    :return:
    """
    result = []
    temp = ''
    # print(f"sentence : {sentence}")
    for word in sentence:
        #  Concatenation of individual characters such as English, Korean
        if word.lower() in letters:
            temp += word
        else:
            if temp != "":  # When Chinese appears, add the English word to the result
                result.append(temp.lower())
                temp = ""
            result.append(word.strip())
    # Determine whether the last character is English
    if temp != "":
        result.append(temp.lower())
    return result


def cut(sentence, by_word=False, use_stopwords=False, with_sg=False):
    """
    :param sentence:
    :param by_word: Whether to divide words by a single character
    :param use_stopwords: Whether to use the stop word
    :param with_sg: Return part of speech or not
    :return: List of word segmentation results
    """
    result = []
    try:
        if by_word:
            result = cut_sentence_by_word(sentence)
        else:
            sentence = sentence.lower()
            result = [(item.word, item.flag) for item in psg.lcut(sentence)]
            # print('lcut resp', result)
            if not with_sg:
                result = [i[0] for i in result]
        # 是否使用停用词
        if use_stopwords:
            if with_sg:
                result = [i for i in result if i[0] not in stopwords]
            else:
                result = [i for i in result if i not in stopwords]
        if with_sg:
            result = [(i[0].strip(), i[1]) for i in result if len(i[0].strip()) > 0]
        else:
            result = [i.strip() for i in result if len(i.strip()) > 0]
    except:
        pass
    return result


if __name__ == '__main__':
    # sentence = "人工智能+python和C++哪个难"
    sentence = "python和c++哪个难?UI/UE呢?haha"
    save_res_path = r"E:\learn\AI\nlp_pro\nlp_learn\common_corpus\sim_q_cut.txt"

    sim_q_cut = []
    qa_dict = json.load(open(r"E:\learn\AI\nlp_pro\nlp_learn\common_corpus\qa_dict.json", 'r', encoding='utf-8'))
    q_list = list(qa_dict.keys())
    for q_content in q_list:
        resp = cut(q_content, by_word=0, use_stopwords=1, with_sg=0)
        resp_str = ' '.join(resp) + "\n"
        sim_q_cut.append(resp_str)
    with open(save_res_path,'w',encoding='utf-8') as fw:
        fw.write(''.join(sim_q_cut))
