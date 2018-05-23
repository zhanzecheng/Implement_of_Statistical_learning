# -*- coding: utf-8 -*-
"""
# @Time    : 2018/5/23 上午10:13
# @Author  : zhanzecheng
# @File    : model.py
# @Software: PyCharm
"""
import math

class DecisionTree:
    def __init__(self):
        pass

    def _calcShannonEnt(self, dataSet):
        """
        该函数是用来计算 label 熵 H(x) = -sum(p * log(p))
        :param dataSet:
        :return:
        """
        dataLen = len(dataSet)
        labelCount = {}
        for featVec in dataSet:
            label = featVec[-1]
            if label not in labelCount.keys():
                labelCount[label] = 1
            else:
                labelCount[label] += 1
        shannonEnt = 0.0
        # 这里用这个循环来做公式中的sum
        for key in labelCount:
            prob = labelCount[key] / dataLen
            shannonEnt -= prob * math.log(prob, 2)

        return shannonEnt

    def _chooseBestFeatureToSplit(self, dataSet):
        """
        利用[信息增益比]来选择最佳划分维度, 信息增益比相对于信息增益可以优化有较多种类的特征
        :param dataSet:
        :return:
        """
        # 其中- 1是要减去标签
        numFeatures = len(dataSet[0]) - 1
        baseEntropy = self._calcShannonEnt(dataSet)
        bestInfoGainRation = 0.0
        bestFeature = -1
        for i in range(numFeatures):
            # 取出该种特征对应的值
            featList = [example[i] for example in dataSet]
            uniqueVals = set(featList)
            newEntropy = 0
            splitInfo = 0
            for value in uniqueVals:
                subDataSet = self._splitDataSet(dataSet, i, value)  # 每个唯一值对应的剩余feature的组成子集
                prob = len(subDataSet) / float(len(dataSet))
                newEntropy += prob * self._calcShannonEnt(subDataSet)
                splitInfo += -prob * math.log(prob, 2)
            infoGain = baseEntropy - newEntropy
            if (splitInfo == 0): # fix the overflow bug
                continue
            infoGainRatio = infoGain / splitInfo             #这个feature的infoGainRatio
            if (infoGainRatio > bestInfoGainRation):          #选择最大的gain ratio
                bestInfoGainRation = infoGainRatio
                bestFeature = i                              #选择最大的gain ratio对应的feature
        return bestFeature

    def _majorityCnt(self, classList):
            """
            以投票机制来选出max类别
            :param classList:
            :return:
            """
            classCount = {}
            for vote in classList:
                if vote not in classCount.keys():
                    classCount[vote] = 0
                classCount[vote] += 1

            sortedClassCount = sorted(classCount.items(), key=lambda x: x[1], reverse=True)
            return sortedClassCount[0][0]

    def _splitDataSet(self, dataSet, axis, value):
        """
        输入：数据集，选择维度，选择值
        输出：划分数据集
        描述：按照给定特征划分数据集；去除选择维度中等于选择值的项
        :param dataSet:
        :param axos:
        :param value:
        :return:
        """
        retDataSet = []
        for featVec in dataSet:
            if featVec[axis] == value:  # 只看当第i列的值＝value时的item
                reduceFeatVec = featVec[:].copy()
                del reduceFeatVec[axis]
                retDataSet.append(reduceFeatVec)
        return retDataSet

    def _createTree(self, dataSet, labels):
        """
        以递归的方式来构造决策树
        伪代码:
        1）若数据集中所有实例属于同一类C，则T为单节点树，并将类C作为该节点的类标记，返回T
        2）若特征(A)为空，则返回D中实例中出现最多的类别作为该节点的类别，返回T
        3）否则，计算A中各特征对D的信息增益，选择信息增益最大的特征Ag
        4）[可选]如果Ag小于阈值e，则置T为单节点树，并将D中实例数最大的类作为该节点的标记类，返回T
        5）否则，对Ag的每一可能值ai，依Ag=ai将D分割为若干非空子集Di，对其递归的进行调用
        :param dataSet:
        :param labels:
        :return:
        """

        # 得到数据集中各类别
        classList = [example[-1] for example in dataSet]
        # 如果所有实例都属于同一类C， 则停止划分
        if classList.count(classList[0]) == len(classList):
            return classList[0]
        # 如果特征为空,则返回出现次数最多的类
        if len(dataSet[0]) == 1:
            return self._majorityCnt(classList)

        # 否则，计算信息熵增益，并且选取最大的作为分类标准
        # TODO: implement the function
        bestFeat = self._chooseBestFeatureToSplit(dataSet)

        # 获得特征名
        bestFeatName = labels[bestFeat]
        # 用字典的方式来建立树
        myTree = {bestFeatName:{}}

        # 剔除该特征
        del labels[bestFeat]

        # 得到该特征所有的可能取值
        featValues = [example[bestFeat] for example in dataSet]
        uniqueVals = set(featValues)
        for value in uniqueVals:
            subLabels = labels[:]
            # 递归的建造树
            myTree[bestFeatName][value] = self._createTree(self._splitDataSet(dataSet, bestFeat, value), subLabels)

        return myTree

    def _classify(self, inputTree, featLabels, testVec):
        """
        递归的找出测试数据所属于的类别
        :param inputTree:
        :param featLabels:
        :param testVec:
        :return:
        """
        firstStr = list(inputTree.keys())[0]
        secondDict = inputTree[firstStr]
        featIndex = featLabels.index(firstStr)
        # 递归的访问分类树
        for key in secondDict.keys():
            if testVec[featIndex] == key:
                if type(secondDict[key]).__name__ == 'dict':
                    # 如果secondDict[key]仍然是字典，则继续向下层走
                    classLabel = self._classify(secondDict[key], featLabels, testVec)
                else:
                    # 如果secondDict[key]已经只是分类标签了，则返回这个类别标签
                    classLabel = secondDict[key]
        return classLabel



    def train(self,dataSet, label):
        """
        该函数用来训练一个决策树
        :param dataSet:
        :param label:
        :return:
        """
        featureNum = len(dataSet[0])
        featureList = []
        # 得到特征名
        for i in range(featureNum):
            featureList.append(str(i))
        # 联合label和train
        for count, d in enumerate(dataSet):
            d.extend([label[count]])

        self.tree = self._createTree(dataSet, featureList)

    def predict(self, dataSet):
        """
        利用训练好的决策树来进行分类
        :param data:
        :return:
        """
        featureNum = len(dataSet)
        featureList = []
        # 得到特征名
        for i in range(featureNum):
            featureList.append(str(i))

        return self._classify(self.tree, featLabels=featureList, testVec=dataSet)




if __name__ == '__main__':
    data = [
        [1, 2, 3],
        [1, 2, 4],
        [1, 2, 5],
        [0, 2, 1],
        [0, 11, 2]
    ]
    label = [0, 0, 0, 1, 1]
    cls = DecisionTree()
    # 训练决策树
    cls.train(data, label)
    print(cls.tree)
    data = [1, 1, 1]
    # 测试决策树
    print(cls.predict(data))