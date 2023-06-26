import numpy as np
import matplotlib.pyplot as plt

import csv
import DStab.CStabDist as ds_cs
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering

from scipy.stats.mstats import mquantiles
from scipy.io import savemat
import time

def read_csv(f):
    with open(f) as fp:
        reader = csv.reader(fp, delimiter=" ", quotechar='"')
        # next(reader, None)  # skip the headers
        data_read = [row for row in reader]

    data = []
    for d in data_read:
        l = []
        for e in d:
            if len(e)>0:
                l.append(e)
        data.append(l)
    data = np.array(data)
    return data


for index in np.arange(0, 9, dtype=np.int32):
# for index in [9]:
    if index == 0:
        f = 'R15.txt'
        kmax = 21
        Data = np.loadtxt(f, delimiter='\t')
        st = StandardScaler(with_std=True, with_mean=True)
        X_ = st.fit_transform(Data[:, :-1])
        y_ = Data[:, -1].astype(np.int32)
    elif index == 1:
        f = 'D31.txt'
        kmax = 41
        Data = np.loadtxt(f, delimiter='\t')
        st = StandardScaler(with_std=True, with_mean=True)
        X_ = st.fit_transform(Data[:, :-1])
        y_ = Data[:, -1].astype(np.int32)
    elif index == 2:
        f = 'gauss.csv'
        kmax = 12
        Data = np.loadtxt(f, delimiter=',')
        st = StandardScaler(with_std=True, with_mean=True)
        X_ = st.fit_transform(Data[:, :-1])
        y_ = Data[:, -1].astype(np.int32)
    elif index == 3:
        f = 's1.txt'
        kmax = 21
        Data = read_csv(f)
        st = StandardScaler(with_std=True, with_mean=True)
        X_ = st.fit_transform(Data)
        y_ = np.zeros(Data.shape[0], dtype=np.int32)
    elif index == 4:
        f = 's2.txt'
        kmax = 21
        Data = read_csv(f)
        st = StandardScaler(with_std=True, with_mean=True)
        X_ = st.fit_transform(Data)
        y_ = np.zeros(Data.shape[0], dtype=np.int32)
    elif index == 5:
        f = 's2.txt'
        kmax = 21
        Data = read_csv(f)
        st = StandardScaler(with_std=True, with_mean=True)
        X_ = st.fit_transform(Data)
        y_ = np.zeros(Data.shape[0], dtype=np.int32)
    elif index == 6:
        f = 's3.txt'
        kmax = 21
        Data = read_csv(f)
        st = StandardScaler(with_std=True, with_mean=True)
        X_ = st.fit_transform(Data)
        y_ = np.zeros(Data.shape[0], dtype=np.int32)
    elif index == 7:
        f = 's4.txt'
        kmax = 21
        Data = read_csv(f)
        st = StandardScaler(with_std=True, with_mean=True)
        X_ = st.fit_transform(Data)
        y_ = np.zeros(Data.shape[0], dtype=np.int32)
    elif index == 8:
        f = 'unbalance.txt'
        kmax = 12
        Data = read_csv(f)
        st = StandardScaler(with_std=True, with_mean=True)
        X_ = st.fit_transform(Data)
        y_ = np.zeros(Data.shape[0], dtype=np.int32)
    elif index == 9:
        f = 'uniform.csv'
        kmax = 12
        Data = np.loadtxt(f, delimiter=' ')
        st = StandardScaler(with_std=True, with_mean=True)
        X_ = st.fit_transform(Data)
        y_ = np.zeros(Data.shape[0], dtype=np.int32)

    X = X_
    y = y_
    nf = f.split('.')[0]
    cs = ds_cs.CStabDist(n_iter=50)
    #raise Exception

    print('File: ', f)

    score = []
    rscore = []

    Ks = np.arange(2, kmax, dtype=np.int32)
    score = []
    permscore = []
    for ii, k in enumerate(Ks):
        s1 = time.time()
        score.append(cs.fit(X, KMeans(n_clusters=k, n_init=8)))
        permscore.append(cs.score_perm.copy())
        s2 = time.time()
        print(s2-s1)
        print()
    score = np.array(score)
    permscore = np.array(permscore)
    mdict = {}

    mdict['score'] = score
    mdict['null'] = permscore
    mdict['kmax'] = kmax
    savemat('./' + nf + '_km_res_tmp.mat', mdict)
