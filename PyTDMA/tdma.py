import numpy as np
from ._tdma import tdma as t


def tdma(A, b):
    """Tridiagonal matrix solver.

    Parameters
    ----------
    A : M x N, array-like
    b : N, array-like

    Returns
    -------
    ret : N, array-like
    """
    lower = np.hstack((0, np.diag(A, -1)))
    middle = np.diag(A).copy()
    upper = np.hstack((np.diag(A, 1), 0))
    return np.asarray(t(lower, middle, upper,
                                 np.asarray(b).squeeze()))
