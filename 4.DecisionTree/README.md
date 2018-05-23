# C4.5方式生成的决策树

## 使用方式

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
    
    data = [1, 1, 1]
    # 测试决策树
    print(cls.predict(data))

## 模型定义

    利用C4.5算法来生成决策树

## 损失函数
    不含剪枝过程
    
## TODO:
    增加正则化：
       信息增益最小限度
       剪枝过程

