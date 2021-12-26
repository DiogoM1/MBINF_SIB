import numpy as np


def accuracy(y_true, y_pred):
    correct = 0
    for true, pred in zip(y_true, y_pred):
        print(true)
        print(pred)
        if true == pred:
            correct += 1
    return correct / len(y_true)


def mse(y_true, y_pred, squared=True):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    errors = np.average((y_true - y_pred) ** 2, axis=0)
    if not squared:
        errors = np.sqrt(errors)
    return np.average(errors)


def mse_prime(y_true, y_pred):
    return 2 * (y_pred - y_true) / y_true.size


# def cross_entropy(y_true, y_pred):
#     m = len(y_true)
#     return -(1.0 / m) * np.sum(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))


def cross_entropy(y_true, y_pred):
    return -(y_true * np.log(y_pred)).sum()


def cross_entropy_prime(y_true, y_pred):
    return y_pred - y_true
