from logging import getLogger, FileHandler, Formatter

import gbdtree.functions as fn
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_mldata
from sklearn.metrics import accuracy_score

import gbdtree as gb

if __name__ == '__main__':
    logger = getLogger(__name__)
    fh = FileHandler('mnist.log', "w")
    fmter = Formatter('{asctime}\t{name}\t{message}', style='{')
    fh.setLevel('INFO')
    fh.setFormatter(fmter)
    logger.setLevel('INFO')
    logger.addHandler(fh)

    mnist = fetch_mldata('MNIST original')
    logger.info('This is MNIST Original dataset')
    logger.debug('finish fetch datasets')

    # target of image number.
    # note: it is difficult problem to decide 3 or 8.
    target = 3, 8,
    logger.info('target: {0},{1}'.format(*target))

    idx = (mnist.target == target[0]) | (mnist.target == target[1])
    logger.debug(idx.shape)
    x = mnist.data[idx] / 255.
    t = mnist.target[idx]
    t = np.where(t == target[0], 0., 1., )

    # split train and test dataset
    # I shoud have use sklearn.cross_validation.train_test_split...
    np.random.seed(71)
    perm = np.random.permutation(len(t))
    x_train, t_train = x[perm[:2000]], t[perm[:2000]]
    x_test, t_test = x[perm[2000:]], t[perm[2000:]]

    logger.info('training datasize: {0}'.format(t_train.shape[0]))
    logger.info('test datasize: {0}'.format(t_test.shape[0]))

    # setup regression object for training and
    # loss function for evaluating the predict quarity
    regobj = fn.CrossEntropy()
    loss = fn.logistic_loss

    clf = gb.GradientBoostedDT(regobj, loss, test_data=(x_test, t_test))
    clf.fit(x_train, t_train, num_iter=30, eta=.4)

    plt.title('seqence of training and test loss')
    plt.plot(clf.training_loss, 'o-', label='training loss')
    plt.plot(clf.validation_loss, 'o-', label='test loss')
    plt.yscale('log')
    plt.legend()
    plt.show()

    pred_prob = clf.predict(x_test)
    pred_cls = np.where(pred_prob > .5, 1., .0)
    df_pred = pd.DataFrame({'probability': pred_prob, 'predict': pred_cls, 'true': t_test})
    df_pred.to_csv('predict.csv')
    acc = accuracy_score(t_test, pred_cls)
    logger.info('accuracy:{0}'.format(acc))
