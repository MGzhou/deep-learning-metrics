#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Time      :2023/11/16 13:58
# @Author    :ZMP

import sacrebleu
refs = [['今 天 天 气 晴 朗。']]
sys = ['今 天 的 天 气 是 晴 朗 的 。']
bleu = sacrebleu.corpus_bleu(sys, refs)
print(bleu.score)

