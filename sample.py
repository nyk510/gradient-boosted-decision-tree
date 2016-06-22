"""シンプルな人工データに対する回帰問題（普通の回帰と二値分類）
"""
import gbdtree as gb

import matplotlib.pyplot as plt
from matplotlib import colors, ticker, cm
from sklearn.metrics import accuracy_score
from sklearn.cross_validation import train_test_split
import numpy as np
import seaborn as sns

def regression_sample():
    # regression problem for continuity value t
    # make training data
    sample_size = 100
    true_func = np.sin
    np.random.seed(71)

    x = np.linspace(-2,2,sample_size)
    t = true_func(x) + np.random.normal(scale=.2,size=sample_size)
    x = x.reshape(sample_size,1)

    # GradientBoostedDTの定義
    # 連続変数に対しての回帰問題なｄので
        # 目的関数：二乗ロス（LeastSquare)
        # 活性化関数：恒等写像（f(x)=x)
    # 今の当てはまりがどの程度なのか評価するロス関数に二乗ロス関数を与える
    clf = gb.GradientBoostedDT(regobj=gb.LeastSquare(),loss=gb.leastsquare)
    clf.fit(x=x,t=t,max_depth=8,num_iter=20,gamma=.5)

    # plot result of predict accuracy
    xx = np.linspace(-2,2,100).reshape(100,1)
    y = clf.predict(xx)

    plt.figure(figsize=(6,6))
    plt.plot(xx,y,"-",label='predict')
    plt.plot(xx,true_func(xx),"-",label='true function')
    plt.plot(x,t,'o',label='training data')
    plt.legend(loc=4)
    plt.savefig('experiment_figures/regression.png')
    plt.show()

def binary_classification_sample():
    """入力次元数二次元のサンプル問題
    """
    np.random.seed = 71
    x = (
    np.random.normal(loc=1.,scale=1.,size=400).reshape(200,2),
    np.random.normal(loc=-1.,scale=1.,size=400).reshape(200,2),
    )
    t = np.zeros_like(x[0]),np.ones_like(x[1])
    x = np.append(x[0],x[1],axis=0)
    t = np.append(t[0],t[1],axis=0)[:,0]

    x_train,x_test,t_train,t_test = train_test_split(x,t,test_size=.3,random_state=71)

    # 二値分類問題なので目的関数を交差エントロピー、活性化関数をシグモイドに設定
    regobj = gb.Entropy()

    # ロス関数はロジスティクスロス
    loss = gb.logistic_loss

    clf = gb.GradientBoostedDT(regobj,loss,test_data=(x_test,t_test))
    clf.fit(x=x_train,t=t_train,max_depth=4,gamma=.1,lam=1e-2,eta=.1,num_iter=40)

    plt.title('seqence of training loss')
    plt.plot(clf.loss_log,'o-',label='training loss')
    plt.plot(clf.pred_log,'o-',label='test loss')
    plt.legend()
    plt.show()

    plt.figure(figsize=(6,6))

    xx = np.linspace(start=-4,stop=4,num=50)
    yy = np.linspace(start=-4,stop=4,num=50)
    X,Y = np.meshgrid(xx,yy)
    Z = [clf.predict(np.array([a,b]).reshape(1,2))[0] for a in xx for b in yy]
    Z = np.array(Z).reshape(len(xx),len(yy))
    plt.contourf(X,Y,Z,6,cmap=cm.PuBu_r)
    cbar = plt.colorbar()
    plt.plot(x[:200,0],x[:200,1],"o")
    plt.plot(x[200:,0],x[200:,1],"o")
    plt.savefig('experiment_figures/binary_classification.png',dpi=100)

    pred_prob = clf.predict(x_test)
    pred_t = np.where(pred_prob >= .5,1,0)
    acc = np.where(pred_t == t_test,1,0).sum() / len(t_test)
    acc_str = 'test accuracy:{0}'.format(acc)
    print(acc_str)

if __name__ == '__main__':
    sns.set_context('notebook')
    sns.set_style('ticks')
    regression_sample()
    binary_classification_sample()
