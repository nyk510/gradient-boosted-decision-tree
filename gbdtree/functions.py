"""
よく使う便利関数の定義
"""

import numpy as np


def sigmoid(x):
    return 1. / (1. + np.exp(-x))


def logistic_loss(y, t):
    loss = - (t * np.log(y) + (1 - t) * np.log(1 - y))
    return loss


def least_square(y, t):
    loss = np.linalg.norm(y - t)
    return loss


class Objective(object):
    """
    目的関数の abstract class

    目的関数は callable な object でなくてはいけません
    また, call 時には正解ラベルと予測値を受け取って, 目的関数の gradient と hessian の配列を返すような関数である必要があります
    """

    def __init__(self, activate):
        self.activate = activate

    def __call__(self, y, t):
        raise NotImplementedError("Objective must implement `__call__` function")


class CrossEntropy(Objective):
    """
    Cross Entropy Loss Function です

    * 活性化関数 シグモイド関数
    * ロス関数:交差エントロピー
    """

    def __init__(self):
        super().__init__(activate=sigmoid)
        return

    def __call__(self, y, t):
        pred = self.activate(y)
        grad = pred - t
        hess = pred * (1. - pred)
        return grad, hess


class LeastSquare(Objective):
    """
    二乗ロス関数
    * 活性化関数: f(x) = x
    * 目的関数: || x - y || ^ 2
    """

    def __init__(self, ):
        # 二乗ロス関数の時活性化関数は恒等写像
        super().__init__(lambda x: x)

    def __call__(self, y, t):
        pred = self.activate(y)
        grad = 2. * (pred - t)
        hess = 2. * np.ones_like(pred)
        return grad, hess
