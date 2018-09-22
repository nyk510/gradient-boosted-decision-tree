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
    :return: 入力変数 x, その正解ラベル t の tuple
    :rtype (np.ndarray, np.ndarray)
    """
    np.random.seed(seed)
    if true_function == "default":
        true_function = np.sin
    x = np.linspace(-x_scale, x_scale, num_samples)
    t = true_function(x) + np.random.normal(loc=0., scale=noise_scale, size=num_samples)
    return x, t


def regression_sample(true_func=np.sin, x_scale=3.):
    """
    regression problem for continuous targets
    :param float x_scale: データのスケール. [-x_scale, x_scale] の範囲のデータを生成する.
    :return:
    """
    x, t = generate_continuous_data(true_function=true_func, x_scale=x_scale)

    trained_models = []
    iteration_dist = [5, 10, 20, 40, 100]
    for n_iter in iteration_dist:
        # GradientBoostedDTの定義
        # 連続変数に対しての回帰問題なので
        # 目的関数：二乗ロス（LeastSquare)
        # 活性化関数：恒等写像（f(x)=x)
        # 今の当てはまりがどの程度なのか評価するロス関数に二乗ロス関数を与える
        rmse_objective = gb.LeastSquare()
        loss_function = gb.functions.least_square
        clf = gb.GradientBoostedDT(
            objective=rmse_objective, loss=loss_function,
            max_depth=4, num_iter=n_iter, gamma=.01, lam=.1, eta=.1)
        clf.fit(x=x, t=t)
        trained_models.append(clf)

    x_test = np.linspace(-x_scale, x_scale, 100).reshape(100, 1)
    fig = plt.figure(figsize=(6, 6))
    ax_i = fig.add_subplot(1, 1, 1)
    ax_i.plot(x_test, true_func(x_test), "--", label='True Function', color="C0")
    ax_i.scatter(x, t, s=50, label='Training Data', linewidth=1., edgecolors="C0", color="white")
    ax_i.set_xlabel("Input")
    ax_i.set_ylabel("Target")

    for i, (n_iter, model) in enumerate(zip(iteration_dist, trained_models)):
        y = model.predict(x_test)
        ax_i.plot(x_test, y, "-", label='n_iter: {}'.format(n_iter), color=cm.viridis(i / len(iteration_dist), 1))
    ax_i.legend(loc=4)
    ax_i.set_title("Transition by Number of Iterations")
    fig.savefig('experiment_figures/regression.png')


def binary_classification_sample():
    """入力次元数二次元のサンプル問題
    """
    np.random.seed = 71
    x = (
        np.random.normal(loc=.7, scale=1., size=400).reshape(200, 2),
        np.random.normal(loc=-.7, scale=1., size=400).reshape(200, 2),
    )
    t = np.zeros_like(x[0]), np.ones_like(x[1])
    x = np.append(x[0], x[1], axis=0)
    t = np.append(t[0], t[1], axis=0)[:, 0]

    x_train, x_test, t_train, t_test = train_test_split(x, t, test_size=.3, random_state=71)

    # 二値分類問題なので目的関数を交差エントロピー、活性化関数をシグモイドに設定
    regobj = gb.CrossEntropy()

    # ロス関数はロジスティクスロス
    loss = gb.logistic_loss

    clf = gb.GradientBoostedDT(regobj, loss, max_depth=5, gamma=.05, lam=3e-2, eta=.1, num_iter=50)
    clf.fit(x=x_train, t=t_train, validation_data=(x_test, t_test))

    networks = clf.show_network()
    import json
    with open('./view/src/assets/node_edge.json', "w") as f:
        json.dump(list(networks), f)

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_title('Training Transitions')
    ax.plot(clf.training_loss, 'o-', label='Training')
    ax.plot(clf.validation_loss, 'o-', label='Validation')
    ax.set_xlabel("Iterations")
    ax.set_ylabel("Loss Transition")
    ax.legend()
    fig.savefig("experiment_figures/training_transitions.png", dpi=150)

    plt.figure(figsize=(6, 6))

    xx = np.linspace(start=-4, stop=4, num=50)
    yy = np.linspace(start=-4, stop=4, num=50)
    X, Y = np.meshgrid(xx, yy)
    Z = [clf.predict(np.array([a, b]).reshape(1, 2))[0] for a in xx for b in yy]
    Z = np.array(Z).reshape(len(xx), len(yy))
    levels = np.linspace(0, 1, 11)
    plt.contour(X, Y, Z, levels, colors=["gray"], alpha=.05)
    plt.contourf(X, Y, Z, levels, cmap=cm.RdBu)
    # plt.contour(X, Y, Z, levels, cmap=cm.PuBu_r)
    cbar = plt.colorbar()
    plt.scatter(x[:200, 0], x[:200, 1], s=50, label="t = 0", edgecolors="C1", alpha=.7, linewidth=1, facecolor="white")
    plt.scatter(x[200:, 0], x[200:, 1], s=50, label="t = 1", edgecolors="C0", alpha=.7, linewidth=1, facecolor="white")
    plt.legend()
    plt.title("binary regression")
    plt.tight_layout()
    plt.savefig('experiment_figures/binary_classification.png', dpi=100)

    pred_prob = clf.predict(x_test)
    pred_t = np.where(pred_prob >= .5, 1, 0)
    acc = np.where(pred_t == t_test, 1, 0).sum() / len(t_test)
    acc_str = 'test accuracy:{0:.2f}'.format(acc)
    print(acc_str)


if __name__ == '__main__':
    def test_function(x):
        return 1 / (1. + np.exp(-4 * x)) + .5 * np.sin(4 * x)
    regression_sample(true_func=test_function, x_scale=1.)
    binary_classification_sample()
