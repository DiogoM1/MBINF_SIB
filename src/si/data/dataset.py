from copy import copy

import numpy as np
import pandas as pd

from si.util.util import label_gen

__all__ = ['Dataset']


class Dataset:
    def __init__(self, X=None, y=None,
                 xnames: list = None,
                 yname: str = None):
        """ Tabular Dataset"""
        if X is None:
            raise Exception("Trying to instanciate a DataSet without any data")
        self.X = X
        self.y = y
        self.xnames = xnames if xnames else label_gen(X.shape[1])
        self.yname = yname if yname else 'y'

    @classmethod
    def from_data(cls, filename, sep=",", labeled=True):
        """Creates a DataSet from a data file.

        :param filename: The filename
        :type filename: str
        :param sep: attributes separator, defaults to ","
        :type sep: str, optional
        :param labeled: Data has labels
        :type labeled: bool
        :return: A DataSet object
        :rtype: DataSet
        """
        data = np.genfromtxt(filename, delimiter=sep)
        if labeled:
            X = data[:, 0:-1]
            y = data[:, -1]
        else:
            X = data
            y = None
        return cls(X, y)

    @classmethod
    def from_dataframe(cls, df, ylabel=None):
        """Creates a DataSet from a pandas dataframe.

        :param df: Input Dataframe
        :type df: DataFrame
        :param ylabel: [description], defaults to None
        :type ylabel: [type], optional
        :return: [description]
        :rtype: [type]
        """
        if ylabel is not None and ylabel in df.columns:
            X = df.loc[:, df.columns != ylabel].to_numpy()
            y = df.loc[:, ylabel].to_numpy()
            xname = df.columns.tolist().remove(ylabel)
            yname = ylabel
        else:
            X = df.to_numpy()
            y = None
            xname = df.columns.tolist()
            yname = None
        return cls(X, y, xnames=xname, yname=yname)

    def __len__(self):
        """Returns the number of data points."""
        return self.X.shape[0]

    def hasLabel(self):
        """Returns True if the dataset constains labels (a dependent variable)"""
        return False if isinstance(self.y, type(None)) else True

    def getNumFeatures(self):
        """Returns the number of features"""
        return self.X.shape[1]

    def getNumClasses(self):
        """Returns the number of label classes or 0 if the dataset has no dependent variable."""
        return len(np.unique(self.y)) if self.hasLabel() else 0

    def writeDataset(self, filename, sep=","):
        """Saves the dataset to a file

        :param filename: The output file path
        :type filename: str
        :param sep: The fields separator, defaults to ","
        :type sep: str, optional
        """
        if self.hasLabel():
            fullds = np.hstack((self.X, self.y.reshape(len(self.y), 1)))
        else:
            fullds = self.X
        np.savetxt(filename, fullds, delimiter=sep)

    def toDataframe(self):
        """ Converts the dataset into a pandas DataFrame"""
        collumns = copy(self.xnames)
        if self.hasLabel():
            fullds = np.hstack((self.X, self.y.reshape(len(self.y), 1)))
            collumns.append(copy(self.yname))
        else:
            fullds = self.X
        df = pd.DataFrame(fullds, columns=collumns)
        return df

    def getXy(self):
        return self.X, self.y
