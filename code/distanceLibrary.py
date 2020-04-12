# -*- coding: UTF-8 -*-

import numpy as np

def euclideanDistance(d1, d2):
    '''
    计算欧氏距离
    d1,d2 : 两个向量
    '''
    return np.sqrt(((d1 - d2)**2).sum())


def manhattanDistance(d1, d2):
    '''
    计算曼哈顿距离
    d1,d2 : 两个向量
    '''
    return (np.abs(d1 - d2)).sum()