#!/usr/bin/env python
# -*- coding:utf-8 -*-
'''
@Time :2024/07/05 16:27:04
@Desc :使用例子
'''
import os, sys
import pandas as pd

def add_project_directory_to_sys_path(folder_level=0):
    """
    将项目目录添加到系统路径中，确保项目内的模块可以被导入。
    """
    # 当前文件路径
    current_file_path = os.path.abspath(__file__)

    project_path = os.path.dirname(current_file_path)  # 当前文件所在目录
    for i in range(folder_level):
        # 获取项目路径，从当前文件路径向上移动一个目录层级
        project_path = os.path.dirname(project_path)
    # 将项目路径追加到系统路径
    sys.path.append(project_path)

# 添加运行环境, D:\PycharmProjects\NLP_COM\metrics
add_project_directory_to_sys_path(folder_level=1)

from src.rouge_zh import rouge_l_zh



if __name__ == '__main__':

    data_path = "../data/predictions-0.csv"
    data = pd.read_csv(data_path)
    print(f"前数量：{len(data)}")
    # 去掉重复，去除有缺失数据
    data = data.drop_duplicates().dropna()

    print(f"后数量：{len(data)}")

    # data = data.head(2000)  # 取2000个
    references = [[i.strip()] for i in data['Actual Text'].to_list()]
    candidates = [i.strip() for i in data['Generated Text'].to_list()]
    s = rouge_l_zh(references=references, candidates=candidates)
    print(f"rouge_l_zh: {s}")

