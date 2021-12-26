# Confirmar o balancemaneto dos datasets (negativos / positivos)
from copy import copy, deepcopy
import numpy as np


def categorical_to_numeric(dataframe, collumn):
    categories = dataframe.iloc[:, collumn].unique().tolist()
    new_df = copy(dataframe)
    cat_dict = {cat: i for i, cat in enumerate(categories)}
    new_df.iloc[:, collumn] = dataframe.iloc[:, collumn].replace(cat_dict)
    cat_dict = {i: cat_dict[i] for i in cat_dict}
    return new_df, cat_dict


def vectorize_dataset(dataset, labels=None):
    new_dataset = copy(dataset)
    if not labels:
        labels = {}
        for i, name in enumerate(dataset.xnames):
            categories = np.unique(dataset.X[:, i])
            labels[name] = {cat: i for i, cat in enumerate(categories)}
    for i, name in enumerate(dataset.xnames):
        for label, new_val in labels[name].items():
            new_dataset.X[dataset.X[:, i] == label, i] = new_val
    return new_dataset, labels


def trainning_test_data_split(dataset, train_split=0.8):
    from si.data import Dataset
    n = dataset.X.shape[0]
    m = int(train_split * n)
    rng = np.random.default_rng()
    idxs = np.arange(n)
    rng.shuffle(idxs)

    train = Dataset(copy(dataset.X[idxs[:m]]), copy(dataset.y[idxs[:m]]), copy(dataset.xnames), copy(dataset.yname))
    test = Dataset(copy(dataset.X[idxs[m:]]), copy(dataset.y[idxs[m:]]), copy(dataset.xnames), copy(dataset.yname))
    return train, test
