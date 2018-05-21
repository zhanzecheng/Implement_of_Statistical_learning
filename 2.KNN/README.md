# KD树实现的K邻近算法

## 使用方式

    data = [[1, 2], [2, 8], [3, 4], [4, 7], [2, 6], [6, 22], [7, 8]]
    kdtree = KDTree(data)
    distance, point = kdtree.findNN([1,1], 3)

## 模型定义

    利用kd树来划分特征空间

## 损失函数
    不含显性的学习过程

