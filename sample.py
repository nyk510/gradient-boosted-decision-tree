"""シンプルな人工データに対する回帰問題（普通の回帰と二値分類）
"""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib import colors
from sklearn.cross_validation import train_test_split

import gbdtree as gb

color_map = colors.Colormap("Set1")


def generate_continuous_data(true_function="default", x_scale=2., num_samples=100, noise_scale=.2, seed=71):
    """
    連続関数に対する人工データを作成
    :param () => | str true_function:
    :param float x_scale:
    :param int num_samples:
    :param float noise_scale:
    :param int seed:
    :return:
    :rtype (np.ndarray, np.ndarray)
    """
    np.random.seed(seed)
    if true_function == "default":
        true_function = np.sin
    x = np.linspace(-x_scale, x_scale, num_samples)
    t = true_function(x) + np.random.normal(loc=0., scale=noise_scale, size=num_samples)
    return x, t


def regression_sample(x_scale=3.):
    """
    regression problem for continuous targets
    :return:
    """
    true_func = np.sin
    x, t = generate_continuous_data(true_function=true_func, x_scale=x_scale)

    # GradientBoostedDTの定義
    # 連続変数に対しての回帰問題なｄので
    # 目的関数：二乗ロス（LeastSquare)
    # 活性化関数：恒等写像（f(x)=x)
    # 今の当てはまりがどの程度なのか評価するロス関数に二乗ロス関数を与える
    clf = gb.GradientBoostedDT(regobj=gb.LeastSquare(), loss=gb.least_square, max_depth=8, num_iter=30, gamma=.5)
    clf.fit(x=x, t=t)

    # plot result of predict accuracy
    xx = np.linspace(-x_scale, x_scale, 100).reshape(100, 1)
    y = clf.predict(xx)
    print(color_map)
    plt.figure(figsize=(6, 6))
    plt.plot(xx, y, "-", label='Predict', color="C0")
    plt.plot(xx, true_func(xx), "--", label='True Function', color="C1")
    plt.scatter(x, t, s=50, label='Training Data', linewidth=1., edgecolors="C1", color="white")
    plt.legend(loc=4)
    plt.title("Regression for Continuous Targets")
    plt.xlabel("Input")
    plt.ylabel("Target")
    plt.savefig('experiment_figures/regression.png')
    plt.show()


def binary_classification_sample():
    """入力次元数二次元のサンプル問題
    """
    np.random.seed = 71
    x = (
        np.random.normal(loc=1., scale=1., size=400).reshape(200, 2),
        np.random.normal(loc=-1., scale=1., size=400).reshape(200, 2),
    )
    t = np.zeros_like(x[0]), np.ones_like(x[1])
    x = np.append(x[0], x[1], axis=0)
    t = np.append(t[0], t[1], axis=0)[:, 0]

    x_train, x_test, t_train, t_test = train_test_split(x, t, test_size=.3, random_state=71)

    # 二値分類問題なので目的関数を交差エントロピー、活性化関数をシグモイドに設定
    regobj = gb.CrossEntropy()

    # ロス関数はロジスティクスロス
    loss = gb.logistic_loss

    clf = gb.GradientBoostedDT(regobj, loss, max_depth=4, gamma=.1, lam=1e-2, eta=.1, num_iter=40)
    clf.fit(x=x_train, t=t_train, valid_data=(x_test, t_test))

    plt.title('Training Transitions')
    plt.plot(clf.loss_log, 'o-', label='Training Loss')
    plt.plot(clf.pred_log, 'o-', label='Validation Loss')
    plt.xlabel("Iterations")
    plt.ylabel("Loss Function")
    plt.legend()
    plt.show()

    plt.figure(figsize=(6, 6))

    xx = np.linspace(start=-4, stop=4, num=50)
    yy = np.linspace(start=-4, stop=4, num=50)
    X, Y = np.meshgrid(xx, yy)
    Z = [clf.predict(np.array([a, b]).reshape(1, 2))[0] for a in xx for b in yy]
    Z = np.array(Z).reshape(len(xx), len(yy))
    plt.contourf(X, Y, Z, 6, cmap=cm.PuBu_r)
    cbar = plt.colorbar()
    plt.plot(x[:200, 0], x[:200, 1], "o", label="t = 0")
    plt.plot(x[200:, 0], x[200:, 1], "o", label="t = 1")
    plt.legend()
    plt.savefig('experiment_figures/binary_classification.png', dpi=100)

    pred_prob = clf.predict(x_test)
    pred_t = np.where(pred_prob >= .5, 1, 0)
    acc = np.where(pred_t == t_test, 1, 0).sum() / len(t_test)
    acc_str = 'test accuracy:{0:.2f}'.format(acc)
    print(acc_str)


if __name__ == '__main__':
    regression_sample()
    binary_classification_sample()
