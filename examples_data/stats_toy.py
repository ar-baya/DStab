import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

import csv
from scipy.stats.mstats import mquantiles
from scipy.io import loadmat
from scipy.stats import mannwhitneyu

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

# for index in [9, 10]:
for index in np.arange(0, 11, dtype=np.int32):
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
    elif index == 10:
        f = 'normal.csv'
        kmax = 12
        Data = np.loadtxt(f, delimiter=' ')
        st = StandardScaler(with_std=True, with_mean=True)
        X_ = st.fit_transform(Data)
        y_ = np.zeros(Data.shape[0], dtype=np.int32)

    X = X_
    y = y_

    nf = f.split('.')[0]

    mdict = loadmat('./' + nf + '_km_res_tmp.mat')
    score = mdict['score']
    score_perm = mdict['null']
    kmax = mdict['kmax']

    res = []
    for ien, (s_null, s) in enumerate(zip(score_perm, score)):
        stat, pval = mannwhitneyu(s, s_null, alternative='less')
        res.append([ien+2, stat, pval])
    res = np.vstack(res)
    ix = np.argsort(res[:, 2])
    res = res[ix]
    print('Name File: ', nf)
    for ii, row in enumerate(res):
        if ii < 5:
            print('k: ', row[0], ' stat: ', row[1], ' pval: ', row[2])
    print()
    print()
# plt.show()
