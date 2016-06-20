import numpy as np
import matplotlib.pyplot as plt

class ObjFunction(object):
    def __init__(self):
        pass

    def loss(self,x,t)

    def taylor(self,x):
        pass

class Entropy(ObjFunction):
    """
    活性化関数:シグモイド関数
    ロス関数:交差エントロピー
    """
    def __init__(self):
        return

    def sigmoid(self,x):
        return 1./(1. + np.exp(-x))

    def __call__(self,y,t):
        pred = self.sigmoid(y)
        grad = pred - t
        hess = pred * (1-pred)
        return grad,hess

    def loss(self,y,t):
        pred = self.sigmoid(y)
        loss = - (t * np.log(pred) + (1-t) * np.log(1-pred)).sum()
        return loss

def entropy(y,t):
    return - (t * np.exp(y) + (1.-t) * np.exp(1-y)).sum(axis=0)

class Node(object):

    def __init__(self,x,t,y,obj_function=Entropy()):
        self.x = x
        self.t = t
        self.obj_function = obj_function
        self.featrue = None
        self.threshoud = None
        self.right = None
        self.left = None
        self.loss = self.get_loss_value(x,t)
        self.has_children = False
        self.already_calulated_gain = False

        # predict values clustered in this node.
        self.y = y

    def predict(self,x):
        """
        x:  ndarray like.
        return: ndarray like. same dimension as x.
        """
        if has_children:
            if x[:,self.featrue] > self.threshoud:
                return self.left(x)
            else:
                return self.right(x)
        else:
            return self.y

    def get_loss_value(self,x,t):
        y = self.predict(x)
        grad,hess = self.obj_function(x)

        return l

    def get_children_loss(self,idx):
        return self.get_loss_value(self.x[idx],self.t[idx])

    def build(self,best_gain):
        """best_gainと同じ値を持つノードを成長させます
        子ノードが存在する場合は、子ノードにbest_gainをもつものがあるかどうかチェックして再起的"build"の呼び出し
        """

        if self.has_children and self.best_gain == best_gain:
            if self.left.best_gain > self.right.best_gain:
                self.left.build(gain)
            else:
                self.right.build(gain)

        elif self.has_children is False:
            self.featrue = f_idx = self.best_feature_idx
            self.threshoud = threshoud = self.best_threshoud
            x = self.x
            t = self.t

            left_x,left_t = x[x[:,f_idx]< threshoud],t[x[:,f_idx] < threshoud]
            right_x,right_t = x[x[:,f_idx] >= threshoud],t[x[:,f_idx] >= threshoud]

            self.left = Node(x=left_x,t=left_t,loss_function=self.loss_function)
            self.right = Node(x=right_x,t=right_t,loss_function=self.loss_function)
            self.has_children = True

            return

    def calculate_bestgain(self):
        """自分以下のノードが分割されたときの最も良いgainの値を計算して、それを返す
        末端のノードの際にはそれに加えてどの特徴indexで閾値を幾つで分割すれば良いかも同時に保存します。
        updateされるパラメータ
        best_gain
        best_feature_idx: 最も良い分割を与えるindex
        best_threshoud: 最も良い分割を与える閾値
        """

        # 親ノードのとき子ノードに計算を再起的に呼び出し
        if self.has_children:
            l = self.left.calculate_bestgain()
            r = self.right.calculate_bestgain()
            self.best_gain = min(l,r)
            return self.best_gain

        # 子ノードがいなくてもすでに計算したことがあればそれを使う
        if self.already_calulated_gain:
            return self.best_gain

        # 自分が末端ノードのときは分割を行ったときのgainを計算
        # いろいろ初期化
        best_gain = 0.
        best_threshoud = None
        best_feature_idx = None

        # すべての特徴量で、分割の最適化を行って最も良い分割を探索
        for f_idx in range(num_feature):

            # ユニークなデータ点とその中間点を取得
            # 中間点は分類するときの基準値 threshoud を決定するために使う
            # 入力変数がカテゴリ値のときは考えていません
            data_f = np.unique(x[:,f_idx])
            sep_points = (data_f[1:] + data_f[:-1]) / 2.

            for threshoud in sep_points:

                left_idx = x[:,f_idx] < threshoud
                right_idx = x[:,f_idx] >= threshoud
                loss_left = self.get_children_loss(idx=left_idx)
                loss_right = self.get_children_loss(idx=right_idx)
                gain = self.loss - loss_left - loss_right

                if best_gain < gain:
                    best_gain = gain
                    best_threshoud = threshoud
                    best_feature_idx = f_idx

        self.best_gain = best_gain
        self.best_feature_idx = best_feature_idx
        self.best_threshoud = best_threshoud

        # 一度計算したら再度分割されるまでは同じなのでスキップさせる
        self.already_calulated_gain = True

        return best_gain

class GradientBoostedDT(object):

    def __init__(self,):
        self.x = x
        self.t = t

    def fit(self,x,t,max_depth=8,gamma=.1,num_iter=20,eta=.1):
        """木を成長させます
        max_depth: 分割の最大値
        gamma: 木を一つ成長させることに対するペナルティ
        """
        self.trees = []
        t_i = t
        
        for i in range(num_iter):
            root_node = Node(x=x,t=t_i,y=0.)

            for depth in range(max_depth):
                best_gain = root_node.calculate_bestgain()
                root_node.build(best_gain=best_gain)

            self.trees.append(root_node)
            f_i = root_node(x)
            t_i = t_i - f_i

        return

    def predict(self,x):


if __name__ == '__main__':
    np.random.seed = 71
    x = np.random.normal(size=10)
    t = np.sin(x) + np.random.normal(scale=.1,size=len(x))
    root_node = Node(x=x,t=t,loss_function=entropy)
    print(root_node.calculate_bestgain)
