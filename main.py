import gbdtree as gb
import functions as fn

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_mldata
from sklearn.metrics import accuracy_score
from logging import getLogger,FileHandler,Formatter,DEBUG

import pandas as pd

if __name__ == '__main__':
    logger = getLogger(__name__)
    fh = FileHandler('main.log',"w")
    fmter = Formatter('{asctime}\t{name}\t{message}',style='{')
    fh.setLevel('INFO')
    fh.setFormatter(fmter)
    logger.setLevel('INFO')
    logger.addHandler(fh)

    mnist = fetch_mldata('MNIST original')
    logger.info('This is MNIST Original dataset')
    logger.debug('finish fetch datasets')

    target = 3,8,
    logger.info('target: {0},{1}'.format(*target))

    idx = (mnist.target == target[0]) | (mnist.target == target[1])
    logger.debug(idx.shape)
    x = mnist.data[idx] / 255.
    t = mnist.target[idx]
    t = np.where(t==target[0],0.,1.,)

    np.random.seed(71)
    perm = np.random.permutation(len(t))
    x_train,t_train = x[perm[:2000]],t[perm[:2000]]
    x_test,t_test = x[perm[2000:]],t[perm[2000:]]
    logger.info('training datasize: {0}'.format(t_train.shape[0]))
    logger.info('test datasize: {0}'.format(t_test.shape[0]))
    regobj = fn.Entropy()
    loss = fn.logistic_loss

    clf = gb.GradientBoostedDT(regobj,loss,test_data=(x_test,t_test))
    clf.fit(x_train,t_train,num_iter=30,eta=.4)

    plt.title('seqence of training loss')
    plt.plot(clf.loss_log,'o-',label='training loss')
    plt.plot(clf.pred_log,'o-',label='test loss')
    plt.legend()
    plt.show()

    pred_prob = clf.predict(x_test)
    pred_cls = np.where(pred_prob>.5,1.,.0)
    df_pred = pd.DataFrame({'probability':pred_prob,'predict':pred_cls,'true':t_test})
    df_pred.to_csv('predict.csv')
    acc = accuracy_score(t_test,pred_cls)
    logger.info('accuracy:{0}'.format(acc))
