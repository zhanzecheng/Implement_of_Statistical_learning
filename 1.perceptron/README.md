# 感知机

## 使用方式

    X = np.asarray([[1, 1], [3, 3], [4, 3]])
    y = np.asarray([[-1], [1], [1]])
    model = Perceptron()
    model.fit( X, y)

## 模型定义

    y = sign(w * x + b)

## 损失函数
利用分错的点到感知机的距离来当作损失函数

## 优化方式

    w += 学习率 * y * x
    b += 学习率 * y

以此方式来来回迭代，完成w和b的修改