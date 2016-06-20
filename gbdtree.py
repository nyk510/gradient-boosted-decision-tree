import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors, ticker, cm

def sigmoid(x):
    return 1./(1. + np.exp(-x))

def logistic_loss(y,t):
    loss = - (t * np.log(y) + (1-t)*np.log(1-y))
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
        return

    def __call__(self,y,t):
        pred = sigmoid(y)
        grad = pred - t
        hess = pred * (1. - pred)
        return grad,hess

class Node(object):
    def __init__(self,x,t,grad,hess,lam=1e-4,obj_function=Entropy()):
        self.x = x
        self.t = t
        self.grad = grad
        self.hess = hess
        self.lam = lam
        self.obj_function = obj_function
        self.featrue = None
        self.threshoud = None
        self.right = None
        self.left = None
        self.has_children = False
        self.already_calulated_gain = False
        self.num_feature = x.shape[1]
        self.num_data = x.shape[0]

        # predict values clustered in this node.
        self.y = - grad.sum() / (lam + hess.sum())

        # optimal lossfunction value.
        self.loss = self.get_objval()

    def predict(self,x):
        """
        x:  ndarray like.
        return: ndarray like. same dimension as x.
        """
        if self.has_children:
            return np.where(x[:,self.featrue] < self.threshoud,
                     self.left.predict(x),
                     self.right.predict(x))
        else:
            return self.y

    def get_loss_value(self,grad,hess):
        """勾配、ヘシアン情報から、二次近似されたロス関数の値を計算します
        """
        obj_val = - grad.sum() ** 2. / (self.lam + hess.sum()) / 2.
        return obj_val

    def get_children_loss(self,idx):
        return self.get_loss_value(self.grad[idx],self.hess[idx])

    def build(self,best_gain):
        """best_gainと同じ値を持つノードを成長させます
        子ノードが存在する場合は、子ノードにbest_gainをもつものがあるかどうかチェックして再起的"build"の呼び出し
        """

        if self.has_children:
            if self.left.best_gain > self.right.best_gain:
                self.left.build(best_gain)
            else:
                self.right.build(best_gain)

        elif self.has_children is False:
            self.featrue = f_idx = self.best_feature_idx
            self.threshoud = threshoud = self.best_threshoud
            x = self.x
            t = self.t

            left_idx = x[:,f_idx] < threshoud
            right_idx = x[:,f_idx] >= threshoud

            print('left:{0}, right:{1}, feature_index:{2}'.format(
                sum(left_idx),sum(right_idx),f_idx))

            l_x,l_t,l_g,l_h = x[left_idx], t[left_idx], self.grad[left_idx], self.hess[left_idx]
            r_x,r_t,r_g,r_h = x[right_idx], t[right_idx], self.grad[right_idx], self.hess[right_idx]

            self.left = Node(x=l_x,t=l_t,grad=l_g,hess=l_h)
            self.right = Node(x=r_x,t=r_g,grad=r_g,hess=r_h)
            self.has_children = True
            self.already_calulated_gain = False

            return

        else:
            print('buildがうまく行っていません')
            raise

    def calculate_bestgain(self):
        """自分以下のノードが分割されたときの最も良いgainの値を計算して、それを返す
        末端のノードの際にはそれに加えてどの特徴indexで閾値を幾つで分割すれば良いかも同時に保存
        updateされるパラメータ
        best_gain
        best_feature_idx: 最も良い分割を与えるindex
        best_threshoud: 最も良い分割を与える閾値
        """

        # 親ノードのとき子ノードに計算を再起的に呼び出し
        if self.has_children:
            l = self.left.calculate_bestgain()
            r = self.right.calculate_bestgain()
            self.best_gain = max(l,r)
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
        for f_idx in range(self.num_feature):

            # data数が1でこれ以上分割できないときはそのまま終了
            if self.num_data <= 1:
                break

            # ユニークなデータ点とその中間点を取得
            # 中間点は分類するときの基準値 threshoud を決定するために使う
            # 入力変数がカテゴリ値のときは考えていません
            data_f = np.unique(self.x[:,f_idx])
            sep_points = (data_f[1:] + data_f[:-1]) / 2.

            for threshoud in sep_points:
                # print('feature_index: {0}'.format(f_idx))
                left_idx = self.x[:,f_idx] < threshoud
                right_idx = self.x[:,f_idx] >= threshoud
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

    def get_objval(self):
        if self.has_children:
            return self.left.get_objval() + self.right.get_objval()

        # 末端ノードの時真面目に計算
        loss = self.get_loss_value(grad=self.grad,hess=self.hess)
        return loss

    def show_network(self):
        pass

class GradientBoostedDT(object):

    def __init__(self,regobj=Entropy(),activate=sigmoid):
        self.x = x
        self.t = t
        self.regobj = regobj
        self.activate = sigmoid
        self.loss = logistic_loss

    def fit(self,x,t,max_depth=3,gamma=1.,num_iter=20,eta=.1,lam=.01):
        """
        max_depth: 分割の最大値
        gamma: 木を一つ成長させることに対するペナルティ
        num_iter: boostingを繰り返す回数
        eta: boostingのステップサイズ
        lam: 目的関数の正則化パラメータ
        """

        self.max_depth = max_depth
        self.gamma = gamma
        self.eta = eta

        self.trees = []

        self.f = np.zeros_like(t)
        self.loss_log = []

        for i in range(num_iter):
            # 直前の予測値と目的の値とで勾配とヘシアンを計算
            grad,hess = self.regobj(self.f,t)

            root_node = Node(x=x,t=t,grad=grad,hess=hess,lam=lam)

            for depth in range(max_depth):
                print('object_value:{0}\n'.format(root_node.get_objval()))
                print('iterate: {0},\tdepth: {1}'.format(i,depth))

                best_gain = root_node.calculate_bestgain()
                print('Gain: {0}'.format(best_gain))

                if best_gain < gamma:
                    break
                else:
                    root_node.build(best_gain=best_gain)

            self.trees.append(root_node)
            f_i = root_node.predict(x)
            self.f += eta * f_i
            loss = self.train_loss()
            self.loss_log.append(loss)

        return

    def train_loss(self):
        a = self.activate(self.f)
        loss = self.loss(a,self.t)
        return loss.sum()

    def predict(self,x):
        a = np.zeros_like(x[:,0])
        for i,tree in enumerate(self.trees):
            a += self.eta * tree.predict(x)
        pred = self.activate(a)
        return pred


if __name__ == '__main__':
    np.random.seed = 71
    x = (
    np.random.normal(loc=.5,scale=1.,size=200).reshape(100,2),
    np.random.normal(loc=-.5,scale=1.,size=200).reshape(100,2),
    )
    t = np.zeros_like(x[0]),np.ones_like(x[1])
    x = np.append(x[0],x[1],axis=0)
    t = np.append(t[0],t[1],axis=0)[:,0]

    crf = GradientBoostedDT(regobj=Entropy())
    crf.fit(x=x,t=t)

    plt.title('seqence of training loss')
    plt.plot(crf.loss_log,'o-',label='training loss')
    plt.legend()
    plt.show()

    plt.figure(figsize=(6,6))

    xx = np.linspace(start=-4,stop=4,num=50)
    yy = np.linspace(start=-4,stop=4,num=50)
    X,Y = np.meshgrid(xx,yy)
    Z = [crf.predict(np.array([a,b]).reshape(1,2))[0] for a in xx for b in yy]
    Z = np.array(Z).reshape(len(xx),len(yy))
    plt.contourf(X,Y,Z,6,cmap=cm.PuBu_r)
    cbar = plt.colorbar()

    plt.plot(x[:100,0],x[:100,1],"o")
    plt.plot(x[100:,0],x[100:,1],"o")

    plt.savefig('contour_predict_and_trainingdata.png',dpi=100)
