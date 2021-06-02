import numpy as np


def cos(x, y):
    x = x / np.sqrt(np.sum(x ** 2) + 1e-8)
    y = y / np.sqrt(np.sum(y ** 2) + 1e-8)
    return np.dot(x, y)


if __name__ == '__main__':
    print(cos(np.asarray([1, 1]), np.array([1, 1])))
