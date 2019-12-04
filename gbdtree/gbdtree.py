import numpy as np
from collections import defaultdict
from typing import Union, List

from .functions import CrossEntropy, Objective, logistic_loss
from .utils import get_logger

logger = get_logger(__name__, "DEBUG")


def calculate_objective(grad, hess, lam):
    obj_val = - grad.sum() ** 2. / (lam + hess.sum()) / 2.
    return obj_val


class Node(object):
    """
    Gradient Boosting で作成する木構造のノード object
    """

    def __init__(self, x, t, grad, hess, lam=1e-4, depth=0):
        """
        Args:
            x: この Node の特徴量.
            t: この Node の目的変数
            grad: データの gradient.
            hess: データの hessian
            lam: L2 正則化項. (ゼロ以上)
            depth: この Node の深さ.
        """
        if lam <= 0:
            raise ValueError(f'`lam` must be over zero. Actually, {lam}')

        if len(x.shape) == 1:
            x = x.reshape(-1, 1)

        self.x = x
        self.t = t
        self.grad = grad
        self.hess = hess
        self.lam = lam
        self.depth = depth

        self.split_feature = None
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

        self.best_gain = - np.inf
        self.leaf_has_best_gain = None
        self.best_threshold = None
        self.best_feature_idx = None

    def __str__(self):
        return f'depth={self.depth}_N={self.num_data}'

    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        :param np.ndarray x:
        :return: np.ndarray
        """
        # 子供ノードがある場合それに予測をさせる
        if self.has_children:
            return np.where(x[:, self.split_feature] < self.threshold,
                            self.left.predict(x),
                            self.right.predict(x))
        else:
            return self.y

    def calculate_index_obj(self, idx) -> float:
        """
        自分の持っているデータの中の一部を使って `objective function` の値を計算
        :param idx: 求めたいデータの index の配列. shape = (n_samples, )
        :return objective value
        :rtype float
        """
        return calculate_objective(self.grad[idx], self.hess[idx], self.lam)

    def build(self):
        """
        この node で分割を実行

        NOTE: build を実行する前に `calculate_best_split` を実行して最適な分割特徴量と分割点を計算している必要があります。

        Returns:

        """
        if not self.already_calculated_gain:
            raise ValueError('分割前にかならず最適な分割点を探してる必要があります.')
        self.split_feature = f_idx = self.best_feature_idx
        self.threshold = threshold = self.best_threshold
        x = self.x
        t = self.t

        left_idx = x[:, f_idx] < threshold
        right_idx = x[:, f_idx] >= threshold

        logger.debug('left-count:{0}\tright-count:{1}\tfeature_index:{2}'.format(
            sum(left_idx), sum(right_idx), f_idx))

        l_x, l_t, l_g, l_h = x[left_idx], t[left_idx], self.grad[left_idx], self.hess[left_idx]
        r_x, r_t, r_g, r_h = x[right_idx], t[right_idx], self.grad[right_idx], self.hess[right_idx]

        self.left = Node(x=l_x, t=l_t, grad=l_g, hess=l_h, lam=self.lam, depth=self.depth + 1)
        self.right = Node(x=r_x, t=r_g, grad=r_g, hess=r_h, lam=self.lam, depth=self.depth + 1)
        self.has_children = True
        self.already_calculated_gain = False
        return self.left, self.right

    def calculate_best_split(self, max_depth=5) -> (float, Union[None, 'Node']):
        """
        自分以下のノードが分割されたときの最も良い `gain` の値を計算しそれを返す

        max_depth 以上の node などそれ以上分割が出来ない node の時 None, None の tuple を返す


        末端のノードの際にはそれに加えて以下を保存する
        + どの `index` で分割を行うか - `best_feature_idx`
        + 閾値を幾つで分割するか - `best_threshold`

        Returns:
            best_gain, and Node (which has the best gain).
        """

        if self.depth == max_depth:
            return - np.inf, None

        # 親ノードのとき子ノードに計算を再起的に呼び出し
        if self.has_children:
            l, node_l = self.left.calculate_best_split()
            r, node_r = self.right.calculate_best_split()

            if l is None and r is None:
                return - np.inf, None

            best_gain = max(l, r)
            if l > r:
                self.leaf_has_best_gain = node_l
            else:
                self.leaf_has_best_gain = node_r
            return best_gain, self.leaf_has_best_gain

        # 以下はすべて末端ノード
        # 計算済みであればそれを返す
        if self.already_calculated_gain:
            return self.best_gain, self

        # 以下は計算していない末端ノードに対する計算になる
        # 自分に属するデータが１つしかないときこれ以上分割できないので終了
        # [TODO] 最小の split 数は変更できるようにしたい.
        # instance 引数に取るか関数の引数に取るかは要検討
        if self.num_data <= 1:
            return - np.inf, self

        # すべての特徴量で、分割の最適化を行って最も良い分割を探索
        logger.debug('start search best separate point')
        for f_idx in range(self.num_feature):
            # ユニークなデータ点とその中間点を取得
            # 中間点は分類するときの基準値 threshold を決定するために使う
            # 入力変数がカテゴリ値のときは考えていない
            data_f = np.unique(self.x[:, f_idx])
            sep_points = (data_f[1:] + data_f[:-1]) / 2.

            for threshold in sep_points:
                left_idx = self.x[:, f_idx] < threshold
                right_idx = self.x[:, f_idx] >= threshold
                loss_left = self.calculate_index_obj(idx=left_idx)
                loss_right = self.calculate_index_obj(idx=right_idx)
                gain = self.loss - loss_left - loss_right

                # 既に計算されている最も大きなゲインより大きい場合には更新する
                if gain > self.best_gain:
                    self.best_gain = gain
                    self.best_threshold = threshold
                    self.best_feature_idx = f_idx

            logger.debug(f'Split Index: {f_idx}\tSep Points: {len(sep_points)}\tCurrent Gain: {self.best_gain:.3e}')

        logger.debug('new gain: {:.3e}@{}'.format(self.best_gain, str(self)))
        # 一度計算したら再度分割されるまで best gain の値は同じになるため
        # すでに計算済みであることが分かるように already_calculated_gain = true とする
        self.already_calculated_gain = True

        return self.best_gain, self

    def feature_importance(self, type='gain') -> Union[dict, None]:
        if not self.has_children:
            return None

        data = defaultdict(int)
        if type == 'gain':
            x = self.best_gain
        else:
            x = 1
        data[self.split_feature] += x
        for node in [self.right, self.left]:
            d_i = node.feature_importance()
            if d_i is None:
                continue
            for k, v in d_i.items():
                data[k] += v

        return data

    def calculate_children_objective(self):
        """
        自分の持っているノードすべての目的関数を計算する
        :return: 目的関数値
        :rtype float
        """
        if self.has_children:
            return self.left.calculate_children_objective() + self.right.calculate_children_objective()

        # 末端ノードの時真面目に計算
        loss = calculate_objective(grad=self.grad, hess=self.hess, lam=self.lam)
        return loss

    def _describe(self):
        """
        この Node の情報を dict で出力するメソッド
        :return: Node 情報の dictionary.
            key は以下のような情報を持つ
            * data
                * id: ノードId
                * num_children: このノード以下に存在するデータの数

                > これ以下はこのノードが子ノードを持つ場合のみ出力
                * feature_id: 分割が行われた特徴量の id
                * gain: Node の分割により改善される Objective の値 (Gain)
        :rtype dict
        """
        retval = {
            "data": {
                "num_data": self.num_data,
                'depth': self.depth,
                'y': self.y
            }
        }
        if self.has_children:
            retval["data"]["feature_id"] = self.best_feature_idx
            retval["data"]["gain"] = self.best_gain

        return retval

    def show_network(self):
        """
        ノードの情報とノード同士のつながりを出力する
        自分の子ノードに対しても再帰的に呼び出しを行う.
        :return: node, edige の tuple
        :rtype list[dict] list[dict]
        """
        if self.has_children is False:
            return self._describe()

        retval = {}
        retval['data'] = self._describe()
        retval['data']['right'] = self.right.show_network()
        retval['data']['left'] = self.left.show_network()
        return retval


class GradientBoostedDT(object):
    """
    Gradient Boosted Decision Tree による予測モデル
    """

    def __init__(self, objective="cross_entropy", loss="logistic",
                 max_leaves=8, gamma=1., num_iter=20, eta=.1, reg_lambda=.01, max_depth=5,
                 ):
        """
        :param str | Objective objective:
            回帰する目的関数 or それを表す文字列. (文字列は今は `cross_entropy` のみに対応)
            要するに call した時に (grad, hess) の tuple を返す必要がある.
        :param str | () => loss: ロス関数. `logistic` or callable object
        :param int max_leaves: 分割の最大値
        :param float gamma: 
            木を一つ成長させることに対するペナルティ. 
            gain の値が gamma を超えない場合木の分割を stop する. 
        :param int num_iter: boostingを繰り返す回数
        :param float eta: boostingのステップサイズ
        :param float reg_lambda: 目的関数の正則化パラメータ
        """
        if objective == "cross_entropy":
            self.objective = CrossEntropy()
        elif isinstance(objective, Objective):
            self.objective = objective
        else:
            logger.warning(type(objective))
            raise ValueError("Only support `cross_entropy`. Actually: {}".format(objective))

        if loss == "logistic":
            self.loss = logistic_loss
        elif hasattr(loss, "__call__"):
            self.loss = loss
        else:
            raise ValueError("arg `loss` is only supported `logistic`. ")

        self.activate = self.objective.activate
        self.trees = []  # type: List[Node]
        self.max_leaves = max_leaves
        self.max_depth = max_depth
        self.gamma = gamma
        self.num_iter = num_iter
        self.eta = eta
        self.reg_lambda = reg_lambda
        self.training_loss = None
        self.validation_loss = None
        self.f = None

    def fit(self, x, t, validation_data=None, verbose=1):
        """
        :param np.ndarray x: 特徴量の numpy array. shape = (n_samples, n_features)
        :param np.ndarray t: 目的変数の numpy array. shape = (n_samples, )
        :param [np.ndarray, np.ndarray] | None validation_data:
            (x, y) で構成された validation data.
            None 以外が与えら得た時各 iteration ごとにこのデータを用いて validation loss を計算する.
        :param int verbose:
        :return:
        """
        if verbose >= 2:
            logger.setLevel("DEBUG")
        elif verbose >= 1:
            logger.setLevel("INFO")

        if len(x.shape) == 1:
            x = x.reshape(-1, 1)

        assert len(x) == len(t)
        self.f = np.zeros_like(t)
        self.training_loss = []
        if validation_data is not None:
            self.validation_loss = []

        for i in range(self.num_iter):
            logger.info('start build new Tree')
            # 直前の予測値と目的の値とで勾配とヘシアンを計算
            grad, hess = self.objective(self.f, t)
            root_node = Node(x=x, t=t, grad=grad, hess=hess, lam=self.reg_lambda)

            for depth in range(self.max_leaves):
                # logger.debug('object_value:\y{0:.2f}'.format(root_node.calculate_children_objective()))
                # logger.debug('iterate:\y{0},\tdepth:\y{1}'.format(i, depth))
                best_gain, best_node = root_node.calculate_best_split(self.max_depth)
                # logger.debug('Best Gain:\y{0:.2f}'.format(best_gain))

                if best_node is None:
                    break

                if best_gain < self.gamma:
                    logger.info(f'best gain {best_gain:.3e} below gamma {self.gamma:.3e}. stop build nodes.')
                    break
                else:
                    logger.info('build new node {} gain={:.4f}'.format(best_node, best_gain))
                    best_node.build()

            self.trees.append(root_node)
            f_i = root_node.predict(x)
            self.f += self.eta * f_i
            train_loss = self._current_train_loss(t)

            logger.info('=' * 30)
            logger.info('end tree iteration')
            logger.info('iterate:{0}\tloss:{1:.2e}'.format(i, train_loss))
            if len(self.training_loss) > 0:
                diff = self.training_loss[-1] - train_loss
                logger.info(f'(improve: {diff:.3e})')
            self.training_loss.append(train_loss)

            if validation_data is not None:
                valid_x, valid_t = validation_data
                pred = self.predict(valid_x)
                pred_loss = self.loss(pred, valid_t).mean()
                self.validation_loss.append(pred_loss)
                logger.info('valid loss:\t{0:.3e}'.format(pred_loss))
        return self

    def _current_train_loss(self, t) -> float:
        """
        学習途中でのロス関数の値を計算する
        :param np.ndarray t: target values
        :return: loss values
        :rtype: float
        """
        a = self.activate(self.f)
        loss = self.loss(a, t)
        return loss.mean()

    def predict(self, x, use_trees=None) -> np.ndarray:
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

    def show_network(self) -> dict:
        """
        学習済みの木をいい感じに dict で出力する.

        Returns:
        """
        data = {}

        tree_data = []
        for i, tree in enumerate(self.trees):
            d_i = tree.show_network()
            tree_data.append({
                'index': i,
                'data': d_i
            })
        data['trees'] = tree_data
        return data

    def feature_importance(self, type='gain') -> dict:
        """
        feature importance の計算

        Args:
            type: importance type. `"gain"` or `"split"`.

        Returns:
            importance dict. key is index of feature.
        """
        data = defaultdict(float)
        for t in self.trees:
            d_i = t.feature_importance(type)
            if d_i is None:
                continue
            for k, v in d_i.items():
                data[k] += v
        return data
