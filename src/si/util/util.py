import itertools

# Y is reserved to idenfify dependent variables
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
    data = dataset.X

    stats = {
        "mean": data.mean(),
        "std": data.std(),
        "max": data.max(),
        "min": data.min(),
    }

    # Return the statistics in the user defined format
    if format == 'dict':
        return stats
    else:
        # because the dict values are not lists, must pass an index
        # READ MORE:
        # https://www.statology.org/valueerror-if-using-all-scalar-values-you-must-pass-an-index/
        return pd.DataFrame(stats, index=[0])
