# 评估指标-metrics

本项目是汇集深度学习领域的多种评估指标。

目前包括 bleu 中文评估 | rouge 中文评估。

## 一、使用

### 实验环境

python 3.10

第三方库

```
jieba
nltk
sacrebleu
pandas
Rouge
```

### 使用例子

```python
# 导入bleu中文评估函数
from src.bleu_zh import bleu_zh

# 计算分数，0-100分，浮点数
score = bleu_zh([["我是一名学生"]], ["我是一名学生"])
print(score) # 100.0
```

更多例子请看 `demo` 文件夹。

## 二、文本生成评估指标

用于机器翻译，摘要生成，标题生成，LLM问答等文本生成任务

`BLEU`分数（Bilingual Evaluation Understudy）：

定义： BLEU是一种自动化评估指标，用于测量输出与参考句子之间的相似度。它计算n-gram重叠，可以对生成句子的准确性进行评估。
优点： 快速计算，广泛应用，尤其适用于较大的数据集。
缺点： 主要关注表面形式，不能完全反映模型的语义理解能力

`ROUGE`分数：

定义： 用于评估生成文本的重要性，常用于自动文摘和机器翻译。它考虑了n-gram的重叠以及关键词的匹配。
优点： 能够考虑生成文本的质量和内容覆盖。
缺点： 主要关注表面形式，不能完全反映模型的语义理解能力

`人工`评估：

定义： 通过人工评审员对翻译结果进行评分，通常包括流畅性、准确性、语法等方面的考量。
优点： 能够捕捉更复杂的语义和语法问题。
缺点： 费时费力，主观性较强。

## 分类评估指标

...


## 回归评估指标

...



## 参考

[大语言模型的评价方法探讨 | 数据学习者官方网站(Datalearner)](https://www.datalearner.com/llm-blogs/evaluation_methods_for_large_language_models)

[原始句子来源：ydli-ai/CSL (github.com)](https://github.com/ydli-ai/CSL)

[routged_data.json来源：pltrdy/rouge(github.com)](https://github.com/pltrdy/rouge)
