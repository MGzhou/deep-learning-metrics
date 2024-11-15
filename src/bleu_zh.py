#!/usr/bin/env python
# -*- coding:utf-8 -*-
'''
@Desc :bleu指标
'''
import jieba
import re
import logging
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu
from nltk.translate.bleu_score import SmoothingFunction
from sacrebleu.metrics import BLEU

import warnings

# 忽略所有警告
warnings.filterwarnings("ignore")
sacrelogger = logging.getLogger('sacrebleu')
sacrelogger.setLevel(logging.CRITICAL)  # 忽略警告
jieba.setLogLevel(logging.CRITICAL)  # 忽略警告

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


def bleu_zh(
    references, 
    candidates, 
    weights=(0.25, 0.25, 0.25, 0.25), 
    is_tokenized=False,
    lowercase=True, 
    use_jieba=True,
    use_corpus_bleu=True,
    model_name='nltk'
):
    """
    计算中文BLEU分数, 中文默认是按'字'为单位计算
    - 如果使用sacrebleu, 不需要进行分词, 使用sacrebleu自带的分词器

    Args:
        references (list(list(str))): 参考句子(人工答案), 默认有多个参考句子。
        candidates (list(str)): 待评估的句子(模型结果)
        weights (tuple, list): n-gram 权重. Defaults to (0.25, 0.25, 0.25, 0.25).
        is_tokenized (bool): 是否已经分词. Defaults to False.
        lowercase (bool): 是否将所有单词转换为小写. Defaults to True.
        use_jieba (bool): 是否使用jieba辅助分词, sacrebleu 时无效。. Defaults to True. 
        use_corpus_bleu (bool): 是否使用corpus_bleu, 如果False, 使用sentence_bleu. Defaults to True.
        model_name (str): 模型名称, ['nltk', 'sacrebleu']. Defaults to 'nltk'.

    Sample:
        >>> bleu_zh([['我是一名学生']], ['我是一名学生'])
        100.0
    
    Returns:
        float: BLEU 分数, 0-100分
    """

    assert len(references) == len(candidates), "references and candidates must have the same length"
    assert len(weights) == 4, "weights must have 4 elements"
    assert model_name in ['nltk', 'sacrebleu'], "model_name must be 'nltk' or 'sacrebleu'"

    temp_references = []
    temp_candidates = []

    if not is_tokenized:
        if model_name == 'nltk':
            for reference_, candidate in zip(references, candidates):
                temp_reference_ = []
                for reference in reference_:
                    reference = cut_sentence(reference, lowercase) if use_jieba else list(reference.lower() if lowercase else reference)
                    temp_reference_.append(reference)
                candidate = cut_sentence(candidate, lowercase) if use_jieba else list(candidate.lower() if lowercase else candidate)
                temp_references.append(temp_reference_)
                temp_candidates.append(candidate)
        else:
            temp_references = references
            temp_candidates = candidates
    else:
        temp_references = references
        temp_candidates = candidates
    # 计算BLEU 分数
    if use_corpus_bleu:
        if model_name == 'nltk':
            score = corpus_bleu(temp_references, temp_candidates, weights=weights) * 100
        else:
            bleu = BLEU(tokenize='zh')
            score = bleu.corpus_score(temp_candidates, temp_references).score
    else:
        sum_score = 0
        if model_name == 'nltk':
            for reference, candidate in zip(temp_references, temp_candidates):
                score = sentence_bleu(reference, candidate, weights=weights)
                sum_score += score
        else:
            bleu = BLEU(tokenize='zh')
            for reference, candidate in zip(temp_references, temp_candidates):
                score = bleu.sentence_score(candidate, reference).score
                sum_score += score
        if model_name == 'nltk':
            score = sum_score / len(temp_candidates) *100
        else:
            score = sum_score / len(temp_references)
    score = round(score, 6)
    return score



def test_sacrebleu_zh():
    """
    计算sacrebleu 中文 BLEU 分数
    全部为中文时, 结果一样。
    如果包含英文时, 结果不一样。
    """
    from sacrebleu.metrics import BLEU
    bleu = BLEU(tokenize='zh')
    references = [["并行电流模式控制的ZVZCS PWM DC/DC全桥变换器"]]
    candidates = ["基于并行电流模式的DC/DC变换器数字控制"]

    print(bleu.corpus_score(candidates, references).score)
    print(bleu.sentence_score(candidates[0], references[0]).score)


if __name__ == '__main__':
    # print(cut_sentence('基于ISFLA优化的液压APC-RBF神经网络智能控制器'))
    references = [["并行电流模式控制的ZVZCS PWM DC/DC全桥变换器"]]
    candidates = ["基于并行电流模式的DC/DC变换器数字控制"]

    print(bleu_zh(references, candidates, use_jieba=True, model_name="sacrebleu"))
    # test_sacrebleu_zh()
