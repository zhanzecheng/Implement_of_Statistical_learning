# -*- coding: utf-8 -*-
"""
# @Time    : 2018/3/17 下午8:07
# @Author  : zhanzecheng
# @File    : model.py
# @Software: PyCharm
"""
import numpy as np
import random

class Perceptron:
    '''
    感知机的定义为 f(x) = sign(w * x + b)
    其中 w是n维向量(n为训练数据的维度)
        b是一维
    '''
    def __init__(self):

        self._W = []
        self._b = 0
        self._epochs = 100
        self._learningRate = 1


    def _forword(self, X):
        '''
        感知机的前向传播
        :param X: 训练数据
        :return: 感知机的分类结果
        '''
        logit = np.zeros((len(X), 1))

        for count, x in enumerate(X):
            tmp = 1 if np.matmul(self._W , x) + self._b > 0 else -1
            logit[count] = tmp
        return logit

    def _loss(self, logit, label):
        '''
        感知机分错的个数和索引
        :param logit: 感知机预测的结果
        :param label: 训练集真实的结果
        :return: 感知机分错的个数和索引
        '''
        count = 0
        wrong = []
        for index, (a, b )in enumerate(zip(logit, label)):
            if a != b:
                count += 1
                wrong.append(index)

        self._wrong = count
        return wrong

    def _backford(self, X_wrong, y):
        '''
        更新感知机的参数
        :param X_wrong: 随机选取的分错的点
        :param y: 随机选取的分错的真实坐标
        :return:
        '''
        self._W = self._W + y * X_wrong * self._learningRate
        self._b = self._b + y * self._learningRate

    def fit(self, X_train, y_train):
        '''
        训练感知机
        :param X_train: 训练集
        :param y_train: 标签
        :return:
        '''
        self._W = np.zeros(shape=(1, X_train.shape[1]))
        for epoch in range(self._epochs):
            # 1. First we should do the forward
            logit = self._forword(X_train)

            # 2. Then we should use stochastic gradient descent to optimize our W & b
            wrong = self._loss(logit=logit, label=y_train)

            if len(wrong) == 0:
                print('we succeed the training')
                break

            X_wrong_index = random.choice(wrong)
            self._backford(X_train[X_wrong_index], y_train[X_wrong_index])

            print('Now we have ', str(self._wrong), ' wrong points')
        print('Now we finish our training')

    def predict(self, X_test):
        '''
        利用感知机来预测
        :param X_test: 测试集
        :return:
        '''
        return self._forword(X_test)

def test():
    X = np.asarray([[1, 1], [3, 3], [4, 3]])
    y = np.asarray([[-1], [1], [1]])
    model = Perceptron()
    model.fit( X, y)

