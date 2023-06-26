import warnings
import numpy as np
from numpy.random import RandomState

from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist, pdist, squareform
from scipy.stats.mstats import mquantiles
import scipy.sparse as sp

from sklearn.model_selection import KFold
from sklearn.preprocessing import OneHotEncoder
import multiprocessing
warnings.filterwarnings("error")


def permt(X1, X2, y1, y2, dist):

    rng = np.random.default_rng()
    rints = rng.integers(low=0, high=32000, size=2)
    rng = np.random.default_rng(rints[0])
    uni1 = np.unique(y1)
    uni2 = np.unique(y2)
    y11 = y1.copy()
    y22 = y2.copy()
    rng.shuffle(y11)

    rng = np.random.default_rng(rints[1])
    rng.shuffle(y22)

    center11 = np.asarray([X1[y11 == el].mean(axis=0) for el in uni1])
    center22 = np.asarray([X2[y22 == el].mean(axis=0) for el in uni2])
    t = cdist(center11, center22)
    row_ind, col_ind = linear_sum_assignment(t)
    if 'mean' in dist:
        s_perm = t[row_ind, col_ind].sum() / (uni1.size * np.sqrt(X1.shape[1]))
    elif 'max' in dist:
        s_perm = t[row_ind, col_ind].max() / np.sqrt(X1.shape[1])
    return s_perm


class CStabDist:
    def __init__(self, n_iter=30, split=2, dist='mean', B=20):
        self.X = None
        self.iter = n_iter
        self.split = split
        self.cs_1score = None
        self.cs_1score_mean = -1
        self.set1 = None
        self.set2 = None
        self.Affinity = None
        self.infer = False
        self.hdist = dist
        self.labels = None
        self.score = None
        self.score_perm = None
        self.B = B

    @staticmethod
    def __1learn__(X, iter, split):
        if split == 2:
            kf = list(KFold(n_splits=2, random_state=RandomState(12345+iter), shuffle=True).split(X))
            t1, t2 = kf[0]
            X1 = X[t1]
            X2 = X[t2]
        else:
            kf = list(KFold(n_splits=split, random_state=RandomState(12345+iter), shuffle=True).split(X))
            t1_, t1 = kf[0]
            t2_, t2 = kf[1]
            X1 = X[t1]
            X2 = X[t2]

        center1 = X1.mean(axis=0).reshape(1, -1)
        center2 = X2.mean(axis=0).reshape(1, -1)
        t = cdist(center1, center2).ravel()[0]
        s = t/np.sqrt(X.shape[1])
        return s

    @staticmethod
    def __learn__(X, algo_cluster, split, iter, dist, B):
        if split == 2:
            kf = list(KFold(n_splits=2, random_state=RandomState(12345+iter), shuffle=True).split(X))
            t1, t2 = kf[0]
        else:
            kf = list(KFold(n_splits=split, random_state=RandomState(12345+iter), shuffle=True).split(X))
            t1, t1_ = kf[0]
            t2, t2_ = kf[1]

        X1 = X[t1]
        X2 = X[t2]
        y1 = algo_cluster.fit_predict(X1)
        y2 = algo_cluster.fit_predict(X2)

        uni1 = np.unique(y1)
        uni2 = np.unique(y2)
        if uni1.size == 1 or uni2.size == 1:
            return np.nan

        center1 = np.asarray([X1[y1 == el].mean(axis=0) for el in uni1])
        center2 = np.asarray([X2[y2 == el].mean(axis=0) for el in uni2])
        t = cdist(center1, center2)
        row_ind, col_ind = linear_sum_assignment(t)
        if 'mean' in dist:
            s = t[row_ind, col_ind].sum()/(uni1.size*np.sqrt(X.shape[1]))
        elif 'max' in dist:
            s = t[row_ind, col_ind].max() / np.sqrt(X.shape[1])
        else:
            raise ValueError('wrong dist type')

        pool_work = multiprocessing.Pool(processes=multiprocessing.cpu_count())
        null_s = [pool_work.apply_async(permt, (X1, X2, y1, y2, dist)) for _ in np.arange(0, B, dtype=np.int32)]
        null_s = [e.get() for e in null_s]
        pool_work.close()
        pool_work.join()
        null_s = np.array(null_s)

        return s, (t1, y1), (t2, y2), null_s

    @staticmethod
    def __infer__(X, algo_cluster, split, iter):
        if split == 2:
            kf = list(KFold(n_splits=2, random_state=RandomState(12345+iter), shuffle=True).split(X))
            t1, t2 = kf[0]
        else:
            kf = list(KFold(n_splits=split, random_state=RandomState(12345+iter), shuffle=True).split(X))
            t1, t1_ = kf[0]
            t2, t2_ = kf[1]

        X1 = X[t1]
        X2 = X[t2]
        y1 = algo_cluster.fit(X1).predict(X)
        y2 = algo_cluster.fit(X2).predict(X)
        return y1, y2

    def fit(self, X, cluster=None, noise=None):
        assert cluster is not None
        self.X = X
        cs_score_obj = [self.__learn__(self.X, cluster, self.split, it, self.hdist, self.B) for it in np.arange(0, self.iter, dtype=np.int32)]
        # print('CStabDist: ', self.cs_1score_mean, cs_score.mean())
        cs_score = np.asarray([score_v for score_v, tup1, tup2, _ in cs_score_obj])
        self.set1 = [tup1 for _, tup1, _, _ in cs_score_obj]
        self.set2 = [tup2 for _, _, tup2, _ in cs_score_obj]
        cs_score_perm = np.hstack([score_p for _, _, _, score_p in cs_score_obj])
        self.score = cs_score
        self.score_perm = cs_score_perm

        return cs_score

    def nfr(self, X, cluster=None):
        assert cluster is not None
        self.X = X
        encoder = []
        l = [self.__infer__(self.X, cluster, self.split, it) for it in np.arange(0, self.iter, dtype=np.int32)]
        l1 = [t1 for t1, _ in l]
        l2 = [t2 for _, t2 in l]
        l = l1 + l2
        scale_val = 2*len(l)
        del l1, l2
        for el in l:
            encoder.append(OneHotEncoder().fit_transform(el.reshape(-1, 1)))
        self.Affinity = sp.hstack(encoder, format='csr')
        self.Affinity.data *= 1./np.sqrt(scale_val)
        self.labels = np.vstack(l).T

    def get_labels(self):
        return self.labels

