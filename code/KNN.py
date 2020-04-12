# -*- coding: UTF-8 -*-

#KNN用于手写数字识别

import numpy as np
import math
import time
from distanceLibrary import euclideanDistance,manhattanDistance

def compareDistance(d, kMinDistances, trainLabel, kLabels):
    '''
    比较距离，如果当前距离比已有最大距离小的话，更新距离列表以及标签列表
    d : 当前距离
    kMinDistances : 维护k个最小距离的列表
    trainLabel : 当前测试数据的标签
    kLabels : 维护k个最小距离对应的标签列表
    '''
    maxDis = max(kMinDistances)
    maxIdx = kMinDistances.index(maxDis)
    if d < maxDis:
        #将当前距离及对应的标签加入到距离列表及标签列表中去
        kMinDistances[maxIdx] = d
        kLabels[maxIdx] = trainLabel

def weight(distances, weight_method = None):
    '''
    对距离进行加权，默认不加权
    distances : 维护k个最小距离的列表
    weight_method : 加权方法
    '''
    num = len(distances)
    weighted_distances = [1 for i in range(num)]
    #第一种加权方法，归一化
    if weight_method == 'Normalize':
        d_min = distances[-1]
        d_max = distances[0]
        if d_min == d_max:
            return weighted_distances
        for i in range(num - 1):
            weighted_distances[i] = (d_max - distances[i]) / (d_max - d_min)
    #第二种加权方法，倒数
    elif weight_method == 'Reciprocal':
        const = 1e-3
        for i in range(num):
            weighted_distances[i] = 1 / (distances[i] + const)
    #第三种加权方法，高斯
    elif weight_method == 'Gauss':
        a = 10
        sigma = 20
        for i in range(num):
            weighted_distances[i] = a * math.e**(-distances[i]**2 / (2 * sigma**2))
    #不加权
    else:
        return weighted_distances
    return weighted_distances

def mostPossibleLabel(kMinDistances, kLabels, weight_method = None):
    '''
    找出最可能的标签，默认不加权
    kMinDistances : 维护k个最小距离的列表
    kLabels : 维护k个最小距离对应的标签列表
    weight_method : 加权方法
    '''
    weighted_distances = weight(kMinDistances, weight_method)
    labelsDict = {}
    #对每个标签计数/计算权重
    for i in range(len(kLabels)):
        label = kLabels[i]
        if label in labelsDict:
            labelsDict[label] += weighted_distances[i]
        else:
            labelsDict[label] = weighted_distances[i]
    #返回计数最多的标签
    return max(labelsDict, key = labelsDict.get)

def KNN(trainDatas, trainLabels, testDatas, k, distance_method = 'Euclidean', **kw):
    '''
    KNN函数，默认采用欧氏距离
    trainDatas : 训练数据
    trainLabels : 训练数据对应的标签
    testDatas : 测试数据
    k : 超参数
    distance_method : 距离度量方法
    **kw : 可能传入加权方法weight_method，不传入的话采取不加权的策略；
           也可能传入是否使用kd树参数usingKDTree，不传入或者传入false的话不使用kd树
    '''
    weight_method = None
    usingKDTree = False
    #是否使用距离加权
    if 'weight_method' in kw:
        weight_method = kw['weight_method']
    #是否使用kd树
    if 'usingKDTree' in kw:
        usingKDTree = kw['usingKDTree']

    kdTree = None
    if usingKDTree:
        from KDTree import KDTree
        #根据训练数据构建一棵kd树
        kdTree = KDTree(trainDatas, np.array(trainLabels))

    trainNum = len(trainDatas)
    testNum = len(testDatas)

    testLabels = []

    for i in range(testNum):
        #从大到小排列的k个距离
        kMinDistances = [float('inf') for i in range(k)]
        #与k个距离一一对应的标签
        kLabels = ['0' for i in range(k)]
        testData = testDatas[i, :]
        #使用kd树查找
        if usingKDTree:
            kMinDistances, kLabels = kdTree.searchKNearest(testData, k, distance_method)
        #不使用kd树，用最简单的遍历方法
        else:
            #对每张测试图片都需要遍历训练集得到距离最近的k张图片的标签
            for j in range(trainNum):
                trainData = trainDatas[j, :]
                trainLabel = trainLabels[j]
                d = float('inf')
                #曼哈顿距离
                if distance_method == 'Manhattan':
                    d = manhattanDistance(testData, trainData)
                #欧氏距离
                else:
                    d = euclideanDistance(testData, trainData)
                compareDistance(d, kMinDistances, trainLabel, kLabels)
        #对于当前测试图片，得到其最可能的标签，加入返回列表中
        testLabels.append(mostPossibleLabel(kMinDistances, kLabels, weight_method))

    return testLabels