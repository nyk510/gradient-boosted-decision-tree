import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1./(1. + np.exp(-x))

def logistic_loss(y,t):
    loss = - (t * np.log(y) + (1-t)*np.log(1-y))
    return loss

def leastsquare(y,t):
    loss = np.linalg.norm(y - t)
    return loss

class ObjFunction(object):
    def __init__(self):
        pass

    def __call__(self,y,t):
        pass

class Entropy(ObjFunction):
    """
    活性化関数:シグモイド関数
    ロス関数:交差エントロピー
    """
    def __init__(self):
        self.activate = sigmoid
        return

    def __call__(self,y,t):
        pred = self.activate(y)
        grad = pred - t
        hess = pred * (1. - pred)
        return grad,hess

class LeastSquare(ObjFunction):
    def __init__(self,):
        self.activate = lambda x:x

    def __call__(self,y,t):
        pred = self.activate(y)
        grad = 2. * (y-t)
        hess = 2. * np.ones_like(y)
        return grad,hess
