# -*- coding: UTF-8 -*-

#kd树

import numpy as np
from distanceLibrary import euclideanDistance,manhattanDistance

class KDNode(object):
    '''
    KD树的节点
    vector : 向量点
    label : 向量对应的标签
    leftNode : 左节点
    rightNode : 右节点
    splitDim : 在哪一维度上进行切分
    '''
    def __init__(self, vector, label, splitDim, leftNode, rightNode):
        self.vector = vector
        self.label = label
        self.splitDim = splitDim
        self.leftNode = leftNode
        self.rightNode = rightNode

class KDTree(object):
    '''
    KD树
    dataset : 数据集，每一行为一个向量
    '''
    def __init__(self, dataset, labels):
        self.rootNode = self.makeKDTree(dataset, labels)
        
    def maxVarIndex(self, subDataset):
        '''
        找出方差最大的那一维
        subDataset : 数据集，每一行为一个向量
        '''
        #各个维度上的方差
        varOfAllDim = np.sum((np.square(subDataset - np.mean(subDataset, axis = 0))), axis = 0) / len(subDataset)
        #方差最大的那个维度
        return np.argmax(varOfAllDim)

    def makeKDTree(self, dataset, labels):
        '''
        以递归的方式构建一棵KD树，返回根节点
        dataset : 数据集，每一行为一个向量
        labels : 每一个向量对应的标签
        '''
        #非节点，直接返回None
        if len(dataset) == 0:
            return None
        #叶子节点的左右节点皆空
        if len(dataset) == 1:
            return KDNode(dataset, labels[0], None, None, None)
        #需要在方差最大的那一维进行切分
        splitDim = self.maxVarIndex(dataset)
        sortIndex = (dataset[:, splitDim]).argsort()
        #中位数
        middle = len(sortIndex) // 2
        #左节点是比中位数小（或等于）的
        leftDataset = dataset[sortIndex[0 : middle], :]
        leftLabels = labels[sortIndex[0 : middle]]
        #当前节点存放的向量
        nodeVector = dataset[sortIndex[middle], :]
        label = labels[sortIndex[middle]]
        #右节点是比中位数大（或等于）的
        rightDataset = dataset[sortIndex[middle + 1 :], :]
        rightLabels = labels[sortIndex[middle + 1 :]]
        return KDNode(nodeVector, label, splitDim, self.makeKDTree(leftDataset, leftLabels), self.makeKDTree(rightDataset, rightLabels))

    def updateKNearest(self, distance, kMinDistances, node, kNodes):
        '''
        更新k个最近距离及其对应的k个节点
        distance : 目标向量与当前节点向量的距离
        kMinDistances : 维护着k个最小距离的列表
        node : 当前节点
        kNodes : 与k个最小距离对应的k个节点
        '''
        maxDis = max(kMinDistances)
        maxIdx = kMinDistances.index(maxDis)
        if distance < maxDis:
            #将当前距离及对应的节点加入到距离列表及标签列表中去
            kMinDistances[maxIdx] = distance
            kNodes[maxIdx] = node

    def buildSearchPath(self, node, nodesList, vector, kMinDistances, kNodes, distance_method):
        '''
        根据需要查找的向量构建一个搜索路径
        node : 构建的搜索路径的根节点
        nodesList : 保存搜索路径上各个节点的列表
        vector : 需要查找的向量
        kMinDistances : 维护着k个最小距离的列表
        kNodes : 与k个最小距离对应的k个节点
        distance_method : 距离度量方法
        '''
        while node:
            nodesList.append(node)
            distance = float('inf')
            #计算需要查找的向量到当前节点向量的距离
            if distance_method == 'Manhattan':
                distance = manhattanDistance(vector, node.vector)
            else:
                distance = euclideanDistance(vector, node.vector)
            #根据计算得到的距离更新距离列表及节点列表
            self.updateKNearest(distance, kMinDistances, node, kNodes)
            #当前节点的切分维度
            splitDim = node.splitDim
            if splitDim:
                #在切分维度上比当前节点向量小或相等的话进入左节点
                if vector[splitDim] <= node.vector[splitDim]:
                    node = node.leftNode
                #否则进入右节点
                else:
                    node = node.rightNode
            #已经到达叶子节点
            else:
                node = None
            

    def searchKNearest(self, vector, k, distance_method):
        '''
        在kd树中找出距所需查找向量最近的k个节点，返回k个最小距离及对应的标签
        vector : 需要进行查找的向量
        k : 超参数
        distance_method : 距离度量方法
        '''
        nodesList = []
        kMinDistances = [float('inf') for _ in range(k)]
        kNodes = [None for _ in range(k)]
        #先根据所需查找向量，以kd树的根节点为当前根节点构建搜索路径
        self.buildSearchPath(self.rootNode, nodesList, vector, kMinDistances, kNodes, distance_method)
        while nodesList:
            #将搜索路径上的节点出栈
            node = nodesList.pop()
            #此节点左右子节点都存在的情况下才有可能进入另一个子节点空间进行搜索
            if node.leftNode and node.rightNode:
                splitDim = node.splitDim
                #只有在待查找向量到切分超平面的距离小于当前k个最小距离的最大者的时候才有必要进入另一节点空间
                if abs(vector[splitDim] - node.vector[splitDim]) < kMinDistances[0]:
                    nextNode = None
                    #如果待查找向量在当前节点的左子节点空间，则进入右子节点空间
                    if vector[splitDim] <= node.vector[splitDim]:
                        nextNode = node.rightNode
                    #如果待查找向量在当前节点的右子节点空间，则进入左子节点空间
                    else:
                        nextNode = node.leftNode
                    #以左子节点或右子节点为根节点构建搜索路径，搜索路径上各个节点加入nodesList中
                    self.buildSearchPath(nextNode, nodesList, vector, kMinDistances, kNodes, distance_method)
        kLables = [item.label for item in kNodes]
        return kMinDistances, kLables