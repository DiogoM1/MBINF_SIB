import itertools

# Y is reserved to idenfify dependent variables
import numpy as np
import pandas as pd

ALPHA = 'ABCDEFGHIJKLMNOPQRSTUVWXZ'

__all__ = ['label_gen', 'summary']


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


def summary(dataset, format='df'):
    """ Returns the statistics of a dataset(mean, std, max, min)

    :param dataset: A Dataset object
    :type dataset: si.data.Dataset
    :param format: Output format ('df':DataFrame, 'dict':dictionary ), defaults to 'df'
    :type format: str, optional
    """
    # Validation step to guarantee that the format the user provided is legal per the current documentation
    if format not in ["df", "dict"]:
        raise Exception("Ilegal format provided. Please choose between df and dict.")

    # Assign data and calculate the statistics for the flattened array
    if dataset.hasLabel():
        data = np.hstack([dataset.X, np.reshape(dataset.Y, (-1, 1))])
        columns = dataset.xnames[:] + [dataset.yname]
    else:
        data = dataset.X
        columns = dataset.xnames[:]

    _mean = np.mean(data, axis=0)
    _std = np.std(data, axis=0)
    _max = np.max(data, axis=0)
    _min = np.min(data, axis=0)

    stats_dict = {}

    for i in range(data.shape[1]):
        stats = {
            "mean": _mean[i],
            "std": _std[i],
            "max": _max[i],
            "min": _min[i],
        }
        stats_dict[columns[i]] = stats

    # Return the statistics in the user defined format
    if format == 'dict':
        return stats_dict
    else:
        # because the dict values are not lists, must pass an index
        # READ MORE:
        # https://www.statology.org/valueerror-if-using-all-scalar-values-you-must-pass-an-index/
        a = pd.DataFrame.from_dict(stats_dict, orient="index")
        return pd.DataFrame.from_dict(stats_dict, orient="index")

