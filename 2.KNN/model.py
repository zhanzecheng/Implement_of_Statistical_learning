# -*- coding: utf-8 -*-
"""
# @Time    : 2018/5/21 上午9:50
# @Author  : zhanzecheng
# @File    : model.py
# @Software: PyCharm
"""
import numpy as np

class KD_node():
    #定义的kd树节点
    def __init__(self, point=None, split=None, LL=None, RR=None):
        #节点值
        self.point = point
        #节点分割维度
        self.split = split
        #节点左孩子
        self.left = LL
        #节点右孩子
        self.right = RR


class KDTree:
    def __init__(self, data=None):
        '''
        建树
        :param data:
        '''
        self.root = None
        self.root = self._createNode(self.root, split=self._maxVar(data), data=data)

        pass

    def _createNode(self, root, split=None, data=None):
        '''
        创建kd树
        :param split:
        :param data:
        :return:
        '''
        if len(data) == 0:
            return None
        # 在以split划分的维度上找到中位数
        data = list(data)
        data.sort(key= lambda x : x[split])
        data = np.array(data)
        # 下面用来求中位数
        median = len(data) // 2
        # 下面递归建立左右子树
        root = KD_node(data[median], split)

        root.left = self._createNode(root.left,
                         split=self._maxVar(data[:median]),
                         data=data[:median])
        root.right = self._createNode(root.left,
                         split=self._maxVar(data[median+1:]),
                         data=data[median+1:])
        return root


    def _maxVar(self, data=None):
        '''
        用来求数据方差最大的维度
        :param data:
        :return:
        '''
        if len(data) == 0:
            return 0
        # 按列求均值
        data_mean = np.mean(data, axis=0)
        # numpy 按列减是直接减的
        mean_diff = data - data_mean
        # 求得方差
        data_var = np.sum(mean_diff ** 2, axis=0) / len(data)
        # 求得方差最大位置， 为所要划分的维度
        re = np.where(data_var == np.max(data_var))
        return re[0][0]


    # 下面是kdtree的查找
    def _computeDist(self, pt1, pt2):
        '''
        计算两个实例点的特征距离
        :param pt1: first
        :param pt2: second
        :return: float
        '''
        pt1 = np.array(pt1)
        pt2 = np.array(pt2)
        return np.sqrt(np.sum(np.square((pt1 - pt2))))

    def findNN(self, query, k):
        '''
        查看目标点的最近k个点
        :param query: 需要查询的点
        :param k: 需要多少个临近点
        :return:
        '''
        node_K = []
        nodeList = []
        result = []
        temp_root = self.root
        # 为了方便，在找到叶子节点同时，把所走过的父节点的距离都保存下来，下一次回溯访问就只需要访问子节点，不需要再访问一遍父节点。
        # 下面是为了找到目标点在KD树的划分
        while temp_root:
            nodeList.append(temp_root)
            dd = self._computeDist(query, temp_root.point)
            if len(node_K) < k:
                node_K.append(dd)
                result.append(temp_root.point)
            else:
                # 选出队列里面最大的元素
                max_dist = max(node_K)
                if dd < max_dist:
                    # 类似于优先队列 把该元素pop出来
                    # TODO: 换成优先队列来实现
                    index = node_K.index(max_dist)
                    del node_K[index], result[index]
                    node_K.append(dd)
                    result.append(temp_root.point)
            ss = temp_root.split
            # 找到最靠近的叶子节点
            if query[ss] <= temp_root.point[ss]:
                temp_root = temp_root.left
            else:
                temp_root = temp_root.right

        # 回溯访问父节点
        while nodeList:
            back_point = nodeList.pop()
            ss = back_point.split
            print('父亲节点 : ', back_point.point, '维度 ：', back_point.split)
            max_dist = max(node_K)
            # 若满足进入该父节点的另外一个子节点的条件
            if len(node_K) < k or abs(query[ss] - back_point.point[ss]) < max_dist:
                # 进入另外一个子节点
                if query[ss] <= back_point.point[ss]:
                    temp_root = back_point.right
                else:
                    temp_root = back_point.left
                # 若不是叶子节点
                if temp_root:
                    nodeList.append(temp_root)
                    curDist = self._computeDist(temp_root.point, query)

                    if max_dist > curDist and len(node_K) == k:
                        index = node_K.index(max_dist)
                        del node_K[index], result[index]
                        node_K.append(curDist)
                        result.append(temp_root.point)
                    elif len(node_K) < k:
                        node_K.append(curDist)
                        result.append(temp_root.point)

        return node_K, result


        

if __name__ == '__main__':
    data = [[1, 2], [2, 8], [3, 4], [4, 7], [2, 6], [6, 22], [7, 8]]

    kdtree = KDTree(data)
    distance, point = kdtree.findNN([1,1], 3)
    print('----> distance: ', distance)
    print('----> point: ', point)






