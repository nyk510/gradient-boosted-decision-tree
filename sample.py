import gbdtree as gb
import functions as fn
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

def regression_sample():
    # regression problem for continuity value t
    # make training data
    sample_size = 100
    true_func = np.sin
    np.random.seed(71)

    x = np.random.normal(scale=1.0,loc=0.0,size=sample_size)
    t = true_func(x) + np.random.normal(scale=.2,size=sample_size)
    x = x.reshape(sample_size,1)

    # GradientBoostedDTの定義
    # 連続変数に対しての回帰問題なｄので
        # 目的関数：二乗ロス（LeastSquare)
        # 活性化関数：恒等写像（f(x)=x)
    # 今の当てはまりがどの程度なのか評価するロス関数に二乗ロス関数を与える
    clf = gb.GradientBoostedDT(regobj=fn.LeastSquare(),loss=fn.leastsquare)
    clf.fit(x=x,t=t,num_iter=20,gamma=.5)

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

if __name__ == '__main__':
    sns.set_context('notebook')
    sns.set_style('ticks')
    regression_sample()
