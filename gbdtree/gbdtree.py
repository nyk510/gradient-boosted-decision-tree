import numpy as np

from .functions import CrossEntropy, Objective, logistic_loss
from .utils import get_logger

logger = get_logger(__name__, "INFO")


class Node(object):
    """
    Gradient Boosting で作成する木構造のノード
    """

    def __init__(self, x, t, grad, hess, lam=1e-4, obj_function="cross_entropy"):
        """
        :param x:
        :param t:
        :param grad:
        :param hess:
        :param lam:
        :param obj_function: 目的関数
        """
        if (len(x.shape) == 1):
            x = x.reshape(-1, 1)
        self.x = x
        self.t = t
        self.grad = grad
        self.hess = hess
        self.lam = lam

        if obj_function == "cross_entropy":
            self.obj_function = CrossEntropy()
        elif isinstance(obj_function, Objective):
            self.obj_function = obj_function
        else:
            raise ValueError("obj_function must be `Objective` instance")

        self.feature = None
        self.threshold = None
        self.right = None
        self.left = None
        self.has_children = False
        self.already_calculated_gain = False
        self.num_feature = x.shape[1]
        self.num_data = x.shape[0]

        # predict values clustered in this node.
        self.y = - grad.sum() / (lam + hess.sum())

        self.loss = self.calculate_children_objective()

        self.best_gain = 0.
        self.best_threshold = None
        self.best_feature_idx = None

    def predict(self, x):
        """
        :param np.ndarray x:
        :return: np.ndarray
        """
        if self.has_children:
            return np.where(x[:, self.feature] < self.threshold,
                            self.left.predict(x),
                            self.right.predict(x))
        else:
            return self.y

    def calculate_objective_value(self, grad, hess):
        """
        勾配、ヘシアン情報から、二次近似された objetive function の値を計算
        """
        obj_val = - grad.sum() ** 2. / (self.lam + hess.sum()) / 2.
        return obj_val

    def calculate_index_obj(self, idx):
        """自分の持っているデータの中の一部を使ってobjective functionの値を計算

        idx: 求めたいgrad及びhessのindex lo
        """
        return self.calculate_objective_value(self.grad[idx], self.hess[idx])

    def build(self, best_gain):
        """
        best_gain と同じ値を持つノードの分割を実行する.

        子ノードが存在する場合は、子ノードどちらかに `best_gain` を持つものが存在するので
        * `best_gain` をもつものがあるかどうかチェック
        * 子ノードに対し再起的に `build"` の呼び出し
        """

        if self.has_children:
            if self.left.best_gain > self.right.best_gain:
                self.left.build(best_gain)
            else:
                self.right.build(best_gain)

        else:
            self.feature = f_idx = self.best_feature_idx
            self.threshold = threshoud = self.best_threshold
            x = self.x
            t = self.t

            left_idx = x[:, f_idx] < threshoud
            right_idx = x[:, f_idx] >= threshoud

            logger.debug('left:{0}\tright:{1}\tfeature_index:{2}'.format(
                sum(left_idx), sum(right_idx), f_idx))

            l_x, l_t, l_g, l_h = x[left_idx], t[left_idx], self.grad[left_idx], self.hess[left_idx]
            r_x, r_t, r_g, r_h = x[right_idx], t[right_idx], self.grad[right_idx], self.hess[right_idx]

            self.left = Node(x=l_x, t=l_t, grad=l_g, hess=l_h)
            self.right = Node(x=r_x, t=r_g, grad=r_g, hess=r_h)
            self.has_children = True
            self.already_calculated_gain = False

    def calculate_best_gain(self):
        """
        自分以下のノードが分割されたときの最も良いgainの値を計算して、それを返す
        末端のノードの際にはそれに加えてどの特徴indexで閾値を幾つで分割すれば良いかも同時に保存
        """

        # 親ノードのとき子ノードに計算を再起的に呼び出し
        if self.has_children:
            l = self.left.calculate_best_gain()
            r = self.right.calculate_best_gain()
            self.best_gain = max(l, r)
            return self.best_gain

        # 以下はすべて末端ノード
        # 計算済みであればそれを返す
        if self.already_calculated_gain:
            return self.best_gain

        # 以下は計算していない末端ノードに対する計算になる
        # 自分に属するデータが１つしかないときこれ以上分割できないので終了
        if self.num_data <= 1:
            return self.best_gain

        # すべての特徴量で、分割の最適化を行って最も良い分割を探索
        for f_idx in range(self.num_feature):

            # ユニークなデータ点とその中間点を取得
            # 中間点は分類するときの基準値 threshold を決定するために使う
            # 入力変数がカテゴリ値のときは考えていない
            logger.debug(f_idx)
            data_f = np.unique(self.x[:, f_idx])
            sep_points = (data_f[1:] + data_f[:-1]) / 2.

            for threshold in sep_points:
                left_idx = self.x[:, f_idx] < threshold
                right_idx = self.x[:, f_idx] >= threshold
                loss_left = self.calculate_index_obj(idx=left_idx)
                loss_right = self.calculate_index_obj(idx=right_idx)
                gain = self.loss - loss_left - loss_right

                # 既に計算されている最も大きなゲインより大きい場合には更新する
                if self.best_gain < gain:
                    self.best_gain = gain
                    self.best_threshold = threshold
                    self.best_feature_idx = f_idx

        # 一度計算したら再度分割されるまでは同じなので, already_calculated_gain = true とする
        self.already_calculated_gain = True

        return self.best_gain

    def calculate_children_objective(self):
        """
        自分の持っているノードすべての目的関数を計算する
        :return: 目的関数値
        :rtype float
        """
        if self.has_children:
            return self.left.calculate_children_objective() + self.right.calculate_children_objective()

        # 末端ノードの時真面目に計算
        loss = self.calculate_objective_value(grad=self.grad, hess=self.hess)
        return loss

    def show_network(self):
        """
        ネットワーク構造を木構造で記述したいなあと思っていた. 未実装.
        :return:
        """
        pass


class GradientBoostedDT(object):
    """
    Gradient Boosted Decision Tree による予測モデル
    """

    def __init__(self, regobj="cross_entropy", loss="logistic",
                 max_depth=8, gamma=1., num_iter=20, eta=.1, lam=.01,
                 ):
        """
        :param str | Objective regobj:
            回帰する目的関数 or それを表す文字列. (文字列は今は `cross_entropy` のみに対応)
            要するに call した時に (grad, hess) の tuple を返す必要がある.
        :param str | () => loss: ロス関数. `logistic` or callable object
        :param int max_depth: 分割の最大値
        :param float gamma: 木を一つ成長させることに対するペナルティ
        :param int num_iter: boostingを繰り返す回数
        :param float eta: boostingのステップサイズ
        :param float lam: 目的関数の正則化パラメータ
        """
        if regobj == "cross_entropy":
            self.regobj = CrossEntropy()
        elif isinstance(regobj, Objective):
            self.regobj = regobj
        else:
            logger.warning(type(regobj))
            raise ValueError("Only support `cross_entropy`. Actually: {}".format(regobj))

        if loss == "logistic":
            self.loss = logistic_loss
        elif hasattr(loss, "__call__"):
            self.loss = loss
        else:
            raise ValueError("arg `loss` is only supported `logistic`. ")

        self.activate = self.regobj.activate
        self.trees = []
        self.max_depth = max_depth
        self.gamma = gamma
        self.num_iter = num_iter
        self.eta = eta
        self.lam = lam
        self.loss_log = None
        self.pred_log = None
        self.f = None

    def fit(self, x, t, valid_data=None, verbose=1):
        """
        :param np.ndarray x: 特徴量の numpy array. shape = (n_samples, n_features)
        :param np.ndarray t: 目的変数の numpy array. shape = (n_samples, )
        :param [np.ndarray, np.ndarray] | None valid_data:
            (x, t) で構成された validation data. None 以外が与えら得た時各 iteration ごとにこのデータを用いて validation loss を計算する.
        :param int verbose:
        :return:
        """
        if verbose > 0:
            logger.setLevel("INFO")
        elif verbose > 1:
            logger.setLevel("DEBUG")

        if (len(x.shape) == 1):
            x = x.reshape(-1, 1)
        self.f = np.zeros_like(t)
        self.loss_log = []
        if valid_data is not None:
            self.pred_log = []

        for i in range(self.num_iter):
            # 直前の予測値と目的の値とで勾配とヘシアンを計算
            grad, hess = self.regobj(self.f, t)

            root_node = Node(x=x, t=t, grad=grad, hess=hess, lam=self.lam)

            for depth in range(self.max_depth):
                logger.debug('object_value:\t{0:.2f}'.format(root_node.calculate_children_objective()))
                logger.debug('iterate:\t{0},\tdepth:\t{1}'.format(i, depth))

                best_gain = root_node.calculate_best_gain()
                logger.debug('Best Gain:\t{0:.2f}'.format(best_gain))

                if best_gain < self.gamma:
                    break
                else:
                    root_node.build(best_gain=best_gain)

            self.trees.append(root_node)
            f_i = root_node.predict(x)
            self.f += self.eta * f_i
            train_loss = self._current_train_loss(t)
            logger.info('iterate:{0}\tloss:{1:.2f}'.format(i, train_loss))
            self.loss_log.append(train_loss)

            if valid_data is not None:
                valid_x, valid_t = valid_data
                pred = self.predict(valid_x)
                pred_loss = self.loss(pred, valid_t).sum()
                self.pred_log.append(pred_loss)
                logger.info('testloss:\t{0:.2f}'.format(pred_loss))
        return self

    def _current_train_loss(self, t):
        """
        学習途中でのロス関数の値を計算する
        :param np.ndarray t: target values
        :return: loss values
        :rtype: float
        """
        a = self.activate(self.f)
        loss = self.loss(a, t)
        return loss.sum()

    def predict(self, x, use_trees=None):
        """
        :param np.ndarray x:
        :param None | int use_trees:
            予測に用いる木の数. 存在している木の数よりも大きい時や
            負の値が設定された時はすべての木を用いて予測する
        :return:
        """
        if use_trees and 1 <= use_trees <= len(self.trees):
            nodes = self.trees[:use_trees]
        else:
            nodes = self.trees

        a = np.zeros_like(x[:, 0])
        for i, tree in enumerate(nodes):
            a += self.eta * tree.predict(x)
        pred = self.activate(a)
        return pred
