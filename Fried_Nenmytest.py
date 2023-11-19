# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 13:28:13 2019

@author: liuxin
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# from regression import linear_scores, ridge_scores, lasso_scores, elasticNet_scores
import scipy.stats
"""
    构造降序排序矩阵
"""
def rank_matrix(matrix):
    cnum = matrix.shape[1]
    rnum = matrix.shape[0]
    ## 升序排序索引
    sorts = np.argsort(matrix)
    ## 降序排序索引
    # sorts = np.argsort(-matrix)

    for i in range(rnum):
        k = 1
        n = 0
        flag = False
        nsum = 0
        for j in range(cnum):
            n = n+1
            ## 相同排名评分序值
            if j < cnum-1 and matrix[i, sorts[i,j]] == matrix[i, sorts[i,j + 1]]:
                flag = True;
                k = k + 1;
                nsum += j + 1;
            elif (j == cnum-1 or (j < cnum-1 and matrix[i, sorts[i,j]] != matrix[i, sorts[i,j + 1]])) and flag:
                nsum += j + 1
                flag = False;
                for q in range(k):
                    matrix[i,sorts[i,j - k + q + 1]] = nsum / k
                k = 1
                flag = False
                nsum = 0
            else:
                matrix[i, sorts[i,j]] = j + 1
                continue
    return matrix

def rank_data(df):
    result = np.zeros(shape=(df.shape[1]), dtype=np.float)
    for i in range(len(df)):
        obj=df.iloc[i,:]
        tmp=obj.rank(ascending = True).values
        result=np.vstack([result,tmp])
    result=result[1:]
    return result

"""
    Friedman检验
    参数：数据集个数n, 算法种数k, 排序矩阵rank_matrix(k x n)
    函数返回检验结果（对应于排序矩阵列顺序的一维数组）
"""
def friedman(n, k, rank_matrix):
    # 计算每一列的排序和
    sumr = sum(list(map(lambda x: np.mean(x) ** 2, rank_matrix.T)))
    result = 12 * n / (k * ( k + 1)) * (sumr - k * (k + 1) ** 2 / 4)
    result = (n - 1) * result /(n * (k - 1) - result)
    return result

"""
    Nemenyi检验
    参数：数据集个数n, 算法种数k, 排序矩阵rank_matrix(k x n)
    函数返回CD值
"""

def nemenyi(n, k, q):
    return q * (np.sqrt(k * (k + 1) / (6 * n)))
    
