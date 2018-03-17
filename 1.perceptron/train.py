# -*- coding: utf-8 -*-
"""
# @Time    : 2018/3/17 下午9:09
# @Author  : zhanzecheng
# @File    : train.py
# @Software: PyCharm
"""

import numpy as np

from model import Perceptron

def main():
    X = np.asarray([[1, 1], [3, 3], [4, 3]])
    y = np.asarray([[-1], [1], [1]])
    model = Perceptron()
    model.fit( X, y)

if __name__ == '__main__':
    main()
