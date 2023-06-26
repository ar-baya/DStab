import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

import csv
from scipy.stats.mstats import mquantiles
from scipy.io import loadmat


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

# for index in [9]:
for index in np.arange(0, 10, dtype=np.int32):
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
    _, ax = plt.subplots(1, 1, figsize=(12, 8))

    ax.set_ylabel('Linear Sum', fontsize=20)
    ax.set_xlabel('Nº clusters', fontsize=20)
    plt.subplots_adjust(left=0.075, right=.99, bottom=0.075, top=.99)
    ax.tick_params(axis='both', which='major', labelsize=16)
    ax.tick_params(axis='both', which='minor', labelsize=14)
    if 'R15' in f:
        clusts = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
        lab = [str(el) for el in clusts]
        print(clusts, lab)
        ax.set_xticks(clusts, labels=lab)
    if 's1' in f or 's2' in f or 's3' in f or 's4' in f:
        clusts = [2, 4, 6, 8, 10, 12, 15, 18, 20]
        lab = [str(el) for el in clusts]
        print(clusts, lab)
        ax.set_xticks(clusts, labels=lab)
    if 'D31' in f:
        clusts = [2, 5, 10, 15, 20, 25, 31, 35, 40]
        lab = [str(el) for el in clusts]
        print(clusts, lab)
        ax.set_xticks(clusts, labels=lab)
    if 'gauss' in f or 'unbalance' in f or 'uniform' in f:
        clusts = [2, 4, 6, 8, 10, 11]
        lab = [str(el) for el in clusts]
        print(clusts, lab)
        ax.set_xticks(clusts, labels=lab)
    ax.grid(color='gray', linestyle='--', linewidth=1)

    mdict = loadmat('./' + nf + '_km_res_tmp.mat')
    score = mdict['score']
    kmax = mdict['kmax']

    ks = np.arange(2, kmax, dtype=np.int32)
    y = np.vstack((np.median(score, axis=1), np.mean(score, axis=1))).max(axis=0).ravel()
    y_err = np.abs(np.asarray(mquantiles(score, prob=[0.25, 0.75], axis=1)).T - np.median(score, axis=1))
    ax.errorbar(ks, y, yerr=y_err, capthick=2, capsize=4,
                label='DStab', marker='o', ls='--')

    score = mdict['null']
    kmax = mdict['kmax']
    ks = np.arange(2, kmax, dtype=np.int32)
    y = np.vstack((np.median(score, axis=1), np.mean(score, axis=1))).max(axis=0).ravel()
    y_err = np.abs(np.asarray(mquantiles(score, prob=[0.25, 0.75], axis=1)).T - np.median(score, axis=1))
    ax.errorbar(ks, y, yerr=y_err, capthick=2, capsize=4,
                label='DStab Ref.', marker='o', ls='-.')

    if 'R15' in f  or 'uniform' in f:
        ax.legend(loc='best', ncol=1, fontsize=20)
    plt.savefig('./' + nf + '_res_tmp.png', dpi=150)

# plt.show()
