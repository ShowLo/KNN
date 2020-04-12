# -*- coding: UTF-8 -*-

import numpy as np
import os
from KNN import KNN
import time


def readTxt(filePath):
    '''
    将存为txt文件的图像01矩阵读出来
    filePath : 文件存放路径
    '''
    content = []
    with open(filePath, 'r') as f:
        for line in f:
            oneLine = []
            for l in line.strip():
                oneLine.append(int(l))
            content.append(oneLine)
    return np.array(content, dtype = int)

def main():
    #存放训练及测试数据的文件夹
    trainFolder = './digits/trainingDigits'
    testFolder = './digits/testDigits'

    trainFiles = os.listdir(trainFolder)
    testFiles = os.listdir(testFolder)

    #数字图像的尺寸
    (height, width) = (readTxt(os.path.join(testFolder, testFiles[0]))).shape
    
    trainPhotos = np.zeros((len(trainFiles), height * width))
    trainLabels = []
    testPhotos = np.zeros((len(testFiles), height * width))
    testLabels = []
    trainIndex, testIndex = 0, 0
    #读入训练数据以及对应的标签
    for trainFile in trainFiles:
        #每一行为一张图片经过reshape后的数据
        trainPhotos[trainIndex, :] = (readTxt(os.path.join(trainFolder, trainFile))).reshape(-1)
        trainLabels.append(trainFile.split('_')[0])
        trainIndex += 1
    #读入测试数据以及对应的标签
    for testFile in testFiles:
        testPhotos[testIndex, :] = (readTxt(os.path.join(testFolder, testFile))).reshape(-1)
        testLabels.append(testFile.split('_')[0])
        testIndex += 1
    
    #进行测试
    k = 5
    distance_method = 'Manhattan'
    weight_method = 'Normalize'
    usingKDTree = True
    testLabels_KNN = KNN(trainPhotos, trainLabels, testPhotos, k)
    #testLabels_KNN = KNN(trainPhotos, trainLabels, testPhotos, k, distance_method, weight_method = weight_method)
    #testLabels_KNN = KNN(trainPhotos, trainLabels, testPhotos, k, distance_method, weight_method = weight_method, usingKDTree = usingKDTree)

    testLabels = np.array(testLabels)
    testLabels_KNN = np.array(testLabels_KNN)
    #计算准确率
    accuracy = sum(testLabels == testLabels_KNN) / len(testLabels)

    print('Using KNN, and the parameters are as following : ...')
    print('k = ' + str(k))
    if 'distance_method' not in dir():
        distance_method = 'Euclidean'
    print('The distance measurement method : ' + distance_method)
    if 'weight_method' not in dir():
        print('Without using any Distance-Weighted Rule')
    else:
        print('The Distance-Weighted Rule : ' + weight_method)
    if 'usingKDTree' not in dir() or not usingKDTree:
        print('Without using KD-Tree')
    else:
        print('Using KD-Tree')
    print('The accuracy is %.3f%%' % (accuracy * 100))


if __name__ == '__main__':
    startTime = time.clock()
    main()
    print('The total time cost is %.1fs' % (time.clock() - startTime))