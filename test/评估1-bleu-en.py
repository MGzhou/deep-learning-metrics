#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Time      :2023/11/16 10:22
# @Author    :ZMP

import sacrebleu
refs = [['The dog bit the man.', 'It was not unexpected.', 'The man bit him first.'],
        ['The dog had bit the man.', 'No one was surprised.', 'The man had bitten the dog.']]
sys = ['The dog bit the man.', "It wasn't surprising.", 'The man had just bitten him.']
bleu = sacrebleu.corpus_bleu(sys, refs)
print(bleu.score)




