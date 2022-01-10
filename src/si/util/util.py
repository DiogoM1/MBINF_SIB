import itertools

# y is reserved to identify dependent variables
import numpy as np
import pandas as pd

ALPHA = 'ABCDEFGHIJKLMNOPQRSTUVWXZ'


def label_gen(n):
    """ Generates a list of n distinct labels similar to Excel"""

    def _iter_all_strings():
        size = 1
        while True:
            for s in itertools.product(ALPHA, repeat=size):
                yield "".join(s)
            size += 1

    generator = _iter_all_strings()

    def gen():
        for s in generator:
            return s

    return [gen() for _ in range(n)]


def summary(dataset, fmt='df'):
    """ Returns the statistics of a dataset(mean, std, max, min)

    :param dataset: A Dataset object
    :type dataset: si.data.Dataset
    :param fmt: Output format ('df':DataFrame, 'dict':dictionary ), defaults to 'df'
    :type fmt: str, optional
    """
    # Validation step to guarantee that the fmt the user provided is legal per the current documentation
    if fmt not in ["df", "dict"]:
        raise Exception("Ilegal fmt provided. Please choose between df and dict.")

    # Assign data and calculate the statistics for the flattened array
    if dataset.hasLabel():
        data = np.hstack([dataset.X, np.reshape(dataset.y, (-1, 1))])
        columns = dataset.xnames[:] + [dataset.yname]
    else:
        data = dataset.X
        columns = dataset.xnames[:]

    _mean = np.mean(data, axis=0)
    _std = np.std(data, axis=0)
    _var = np.var(data, axis=0)
    _max = np.max(data, axis=0)
    _min = np.min(data, axis=0)

    stats_dict = {}

    for i in range(data.shape[1]):
        stats = {
            "mean": _mean[i],
            "std": _std[i],
            "var": _var[i],
            "max": _max[i],
            "min": _min[i],
        }
        stats_dict[columns[i]] = stats

    # Return the statistics in the user defined fmt
    if fmt == 'dict':
        return stats_dict
    else:
        # because the dict values are not lists, must pass an index
        # READ MORE:
        # https://www.statology.org/valueerror-if-using-all-scalar-values-you-must-pass-an-index/
        return pd.DataFrame.from_dict(stats_dict, orient="index")


# Ver qual foi a correção de custo aplicada.
def add_intersect(X):
    return np.hstack((np.ones((X.shape[0], 1)), X))


def to_categorical(y, num_classes=None, dtype='float32'):
    y = np.array(y, dtype='int')
    input_shape = y.shape
    if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
        input_shape = tuple(input_shape[:-1])
    y = y.ravel()
    if not num_classes:
        num_classes = np.max(y) + 1
    n = y.shape[0]
    categorical = np.zeros((n, num_classes), dtype=dtype)
    categorical[np.arange(n), y] = 1
    output_shape = input_shape + (num_classes,)
    categorical = np.reshape(categorical, output_shape)
    return categorical


def minibatch(X, batchsize=256, shuffle=True):
    N = X.shape[0]
    ix = np.arange(N)
    n_batches = int(np.ceil(N / batchsize))

    if shuffle:
        np.random.shuffle(ix)

    def mb_generator():
        for i in range(n_batches):
            yield ix[i * batchsize: (i + 1) * batchsize]

    return mb_generator(),
