import numpy as np


def ppmi(C: np.ndarray, eps=1e-8):
    """
    log2 [(p(x, y) * N)/ (p(x) * p(y)]
    """
    N = C.sum()
    S = C.sum(axis=0)

    length = len(C)
    M = np.zeros_like(C, dtype=np.float32)

    for i in range(length):
        for j in range(length):
            M[i, j] = np.log2(C[i, j] * N / (S[i] * S[j]) + eps)

    return M


if __name__ == '__main__':
    print(ppmi(np.ones((5, 5))))
