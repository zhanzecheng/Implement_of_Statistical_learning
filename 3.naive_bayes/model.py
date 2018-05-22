# -*- coding: utf-8 -*-
"""
# @Time    : 2018/5/22 上午10:00
# @Author  : zhanzecheng
# @File    : model.py
# @Software: PyCharm
"""
import numpy as np

class NBayes:
    def __init__(self):
        pass

    def _getProbality(self, trainData, trainLabel):
        """
        目前实现的是二分类的朴素贝叶斯
        :param trainData:
        :param trainLabel:
        :return:
        """
        if type(trainData) != np.ndarray:
            print('---> convert train type to array')
        dataLen = len(trainData)
        # 得到总共有多少特征
        # 计算类别1发生的概率
        pAbusive = sum(trainLabel) / dataLen
        p0 = []
        p1 = []

        featureNum = len(trainData[0])
        for feature in range(featureNum):
            numsWord = len(trainData[0][feature])
            # 1初始化，防止概率为0的情况. 这里使用的是拉普拉斯平滑方式
            p0Num = np.ones(numsWord)
            p1Num = np.ones(numsWord)
            # 以拉普拉斯平滑方式来初始化分母
            p0Denom = 2.0
            p1Denom = 2.0
            for i in range(dataLen):
                if trainLabel[i] == 1:
                    # 利用了numpy的矩阵相加便捷性
                    p1Num += trainData[i][feature]
                    p1Denom += sum(trainData[i][feature])
                else:
                    p0Num += trainData[i][feature]
                    p0Denom += sum(trainData[i][feature])
            # 这里利用log的性质进行变换，可以把原来相乘的表达式变成相加
            p1Vect = np.log(p1Num / p1Denom)
            p0Vect = np.log(p0Num / p0Denom)
            p0.append(p0Vect)
            p1.append(p1Vect)
        return p0, p1, pAbusive


    def _to_categorical(self, y, num_classes=None):
        """
        Converts a class vector (integers) to binary class matrix.
        """
        y = np.array(y, dtype='int')
        input_shape = y.shape
        if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
            input_shape = tuple(input_shape[:-1])
        y = y.ravel()
        if not num_classes:
            num_classes = np.max(y) + 1
        n = y.shape[0]
        categorical = np.zeros((n, num_classes), dtype=np.float32)
        categorical[np.arange(n), y] = 1
        output_shape = input_shape + (num_classes,)
        categorical = np.reshape(categorical, output_shape)
        return categorical

    def train(self, x, y=None):
        x = self._to_categorical(x)
        self.x = x
        self.p0Vec, self.p1Vec, self.pA = self._getProbality(x, y)

    def predict(self, test):
        """
        这里仅实现了对于batch_size为1的predict
        :param test:
        :return:
        """
        x = self.x[0]
        m, n = x.shape
        predict = np.zeros((m, n))
        for row, d in enumerate(test):
            # TODO: 这里对于没有见过的特征值，仅是简单的赋值为固定值
            if d > n:
                d = 0
            predict[row][int(d)] = 1

        p1 = np.log(self.pA)
        for vec2Classify, p1Vec in zip(predict, self.p1Vec):
            p1 += sum(vec2Classify * p1Vec)

        p0 = np.log(1.0 - self.pA)
        for vec2Classify, p0Vec in zip(predict, self.p0Vec):
            p0 += sum(vec2Classify * p0Vec)

        if p1 > p0:
            return 1
        else:
            return 0

if __name__ == '__main__':
    cls = NBayes()
    x = [[1, 2], [2, 3], [2, 1]]
    y = [0, 1, 1]
    cls.train(x, y)
    print(cls.predict([2, 1]))