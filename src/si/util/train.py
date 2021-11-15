# Confirmar o balancemaneto dos datasets (negativos / positivos)
from copy import copy
import numpy as np


def trainning_test_data_split(dataset, train_split=0.8):
    from si.data import Dataset
    n = dataset.X.shape[0]
    m = int(train_split*n)
    rng = np.random.default_rng()
    idxs = np.arange(n)
    rng.shuffle(idxs)

    train = Dataset(copy(dataset.X[idxs[:m]]), copy(dataset.Y[idxs[:m]]), copy(dataset.xnames), copy(dataset.yname))
    test = Dataset(copy(dataset.X[idxs[m:]]), copy(dataset.Y[idxs[m:]]), copy(dataset.xnames), copy(dataset.yname))
    return train, test
