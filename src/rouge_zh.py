#!/usr/bin/env python
# -*- coding:utf-8 -*-
'''
@Desc :rouge指标
'''
from rouge import Rouge
import jieba
import re
import logging
import warnings
import numpy as np

# 忽略所有警告
warnings.filterwarnings("ignore")
jieba.setLogLevel(logging.CRITICAL)

def cut_sentence(sentence, lowercase=True):
    """分词, 通过结巴分词取出英文数字的单词, 中文字符按单个取出

    Args:
        sentence (str): 待分词的句子
    Sample:
        >>> cut_sentence('磁控溅射制备Cu2 ZnSnS4薄膜的研究进展?')
        ['磁', '控', '溅', '射', '制', '备', 'Cu2', ' ', 'ZnSnS4', '薄', '膜', '的', '研', '究', '进', '展', '?']
        
    Returns:
        list: 分词后的结果
    """
    word_list = list(jieba.cut(sentence))
    new_word_list = []
    for w in word_list:
        w = w.lower() if lowercase else w
        # 判断是否包含中文
        is_chinese = re.search(r'[\u4e00-\u9fa5]', w)
        if is_chinese:
            new_word_list.extend(list(w))
        else:
            new_word_list.append(w)
    return new_word_list

def rouge_l_zh(references, candidates, is_tokenized=False, lowercase=True, use_jieba=True):
    """## 计算Rouge-l得分
    Rouge-l指标常用于评估自动文本摘要及翻译任务

    ### Args:
        - `references (list(list(str)))`: 人工参考句子
        - `candidates (list(str))`: 模型预测的句子
        - `is_tokenized (bool)`: 是否已经分词. Defaults to False.
        - `lowercase (bool)`: 是否将所有单词转换为小写. Defaults to True.
        - `use_jieba (bool)`: 是否使用jieba辅助分词。. Defaults to True.

    ### Sample:
        >>> rouge_l_zh([["我是一名学生"]], ["我是一名学生"])
        output: 1.0

    ### Returns:
        - `float`: rouge 分数, 0-100分
    """
    assert len(references) == len(candidates), "references and candidates must have the same length"

    temp_references = []
    temp_candidates = []
    if not is_tokenized:
        for reference_, candidate in zip(references, candidates):
            temp_reference_ = []
            for reference in reference_:
                reference = cut_sentence(reference, lowercase) if use_jieba else list(reference.lower() if lowercase else reference)
                temp_reference_.append(reference)
            candidate = cut_sentence(candidate, lowercase) if use_jieba else list(reference.lower() if lowercase else reference)
            temp_references.append(temp_reference_)
            temp_candidates.append(candidate)
    else:
        temp_references = references
        temp_candidates = candidates
    
    # 计算平均分
    scores = []
    rouge = Rouge()
    for i in range(len(temp_references)):
        one_scores = [rouge.get_scores(" ".join(temp_references[i][j]), " ".join(temp_candidates[i]))[0]["rouge-l"]["f"] for j in range(len(temp_references[i]))]
        score = max(one_scores)  # 多个参考句子取最大值
        scores.append(score)
    score = np.average(scores)  * 100
    score = round(score, 6)
    return score


def test_rouge_l_zh(pred, target):
    rouge = Rouge()
    scores = rouge.get_scores(" ".join(list(pred)), " ".join(list(target)))
    score = scores[0]["rouge-l"]
    # "f" stands for f1_score, "p" stands for precision, "r" stands for recall.
    print(score["f"])

if __name__ == "__main__":
    # test_rouge_l_zh("我是一名学生", "我是一名学生")

    a = rouge_l_zh([["我是一名学生"]], ["我是一名学生"])
    print(a)
