import warnings
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.linalg import sqrtm
from scipy.special import logsumexp
from scipy._lib._util import _asarray_validated





def nan_cov(X):
    """
    Parameter
    ---------
    X: np array with shape (n_sampels, dim)

    Returns
    -------
    cov: np array with shappe (dim, dim)


    Notes:
    - Contrary to numpy.cov, where rowvar is True by default, this function behaves
      like  numpy.cov(X, rowvar=False), given that this is the only case we will encounter
    - This function is expected to return a covariance with only non-NaN values, meaning
      X is expected to have at least two non-NaN couples of values for each couple of columns
      The function does not check this.
    """

    n_samples, dim = X.shape
    mean = np.nanmean(X, axis=0)

    cov = np.zeros((dim, dim))
    for d1 in range(dim):
        for d2 in range(d1+1):
            product = X[:,d1] * X[:,d2]  # shape: (n_samples)
            non_nan_samples = (~np.isnan(product)).astype(int).sum()
                # this is euivalent to   (X_isnan[:,d1] & X_isnan[:,d2]).sum()
                # but we save a bit by computing the product only once
            this_cov = np.nanmean(product) - mean[d1] * mean[d2]
            this_cov *= non_nan_samples / (non_nan_samples-1)
            cov[d1,d2] = this_cov
            cov[d2,d1] = this_cov

    return cov



    # X_isnan = np.isnan(X).astype(int)  # shape: (n_samples, n_features)
    # non_nan_samples = X_isnan.T @ X_isnan
    # print(non_nan_samples)
    # cov = np.zeros((dim, dim))
    # for d1 in range(dim):
    #     for d2 in range(d1+1):
    #         this_cov = np.nanmean(X[:,d1] * X[:,d2]) - mean[d1] * mean[d2]
    #         this_cov *= non_nan_samples[d1,d2] / (non_nan_samples[d1,d2]-1)
    #         cov[d1,d2] = this_cov
    #         cov[d2,d1] = this_cov

    # return cov
if __name__ == "__main__":
    X = np.random.randn(7,3)  # no nan values
    assert np.allclose(nan_cov(X), np.cov(X, rowvar=False))


    X = np.array([[1,     -1],
                  [-1,     1],
                  [1,      1],
                  [-1,    -1],
                  [-2,  np.nan],
                  [2,   np.nan],
                  [np.nan, 0]])

    expected_cov = np.array([[12/(6-1),      0   ],
                             [   0,       4/(5-1)]])
    assert np.allclose(nan_cov(X), expected_cov)





#%%



def nan_norm(X, axis):
    """
    Parameter
    ---------
    X: np array with shape (n_sampels, dim)

    Returns
    -------
    norm: np array with shappe (dim, dim)


    Notes:
    - Contrary to numpy.cov, where rowvar is True by default, this function behaves
      like  numpy.cov(X, rowvar=False), given that this is the only case we will encounter
    - This function is expected to return a covariance with only non-NaN values, meaning
      X is expected to have at least two non-NaN couples of values for each couple of columns
      The function does not check this.
    """



#%%







def nan_logsumexp(a, axis=None, b=None, keepdims=False, return_sign=False, all_nan_return=-np.inf):
    """Compute the log of the sum of exponentials of input elements.
    This function may or may not be a fork of scipy.special.logsumexp,
    https://github.com/scipy/scipy/blob/v1.10.0/scipy/special/_logsumexp.py

    Parameters
    ----------
    a : array_like
        Input array.
    axis : None or int or tuple of ints, optional
        Axis or axes over which the sum is taken. By default `axis` is None,
        and all elements are summed.
    b : array-like, optional
        Scaling factor for exp(`a`) must be of the same shape as `a` or
        broadcastable to `a`. These values may be negative in order to
        implement subtraction.
    keepdims : bool, optional
        If this is set to True, the axes which are reduced are left in the
        result as dimensions with size one. With this option, the result
        will broadcast correctly against the original array.
    return_sign : bool, optional
        If this is set to True, the result will be a pair containing sign
        information; if False, results that are negative will be returned
        as NaN. Default is False (no sign information).
    all_nan_return: float, optional
        Which values to return when all the values to sum are NaNs. Defaults to -np.inf.
        When one of the summed columns is entirely NaNs, np.nansum(NaN) returns 0.,
        so a consistent logsumexp would return
        np.log(np.nansum(np.exp(NaN))) = np.log(0) = -np.inf.
        Setting this value makes the behaviour of this function inconsistent ith np.nansum

    Returns
    -------
    res : ndarray
        The result, ``np.log(np.sum(np.exp(a)))`` calculated in a numerically
        more stable way. If `b` is given then ``np.log(np.sum(b*np.exp(a)))``
        is returned.
    sgn : ndarray
        If return_sign is True, this will be an array of floating-point
        numbers matching res and +1, 0, or -1 depending on the sign
        of the result. If False, only one result is returned.


    """



    a = _asarray_validated(a, check_finite=False)
    if b is not None:
        raise NotImplementedError()

    with warnings.catch_warnings(): # we deal with all NaN values later, no need to raise a warning
        warnings.simplefilter("ignore")
        finite_a = np.where(np.isfinite(a), a, -np.inf)  # same as a, except that +np.inf are replaced with -np.inf
        a_max = np.nanmax(finite_a, axis=axis, keepdims=True)

    if a_max.ndim > 0:
        a_max[~np.isfinite(a_max)] = 0
    elif not np.isfinite(a_max):
        a_max = 0

    tmp = np.exp(a - a_max)

    # suppress warnings about log of zero
    with np.errstate(divide='ignore'):
        s = np.nansum(tmp, axis=axis, keepdims=keepdims)
        if return_sign:
            sgn = np.sign(s)
            s *= sgn  # /= makes more sense but we need zero -> zero
        out = np.log(s)

    if all_nan_return != -np.inf:
        assert (type(all_nan_return) == float) or (type(all_nan_return) == int)
        all_nans = np.isnan(a).all(axis=axis, keepdims=keepdims)
        if out.ndim > 0:
            out[all_nans] = all_nan_return
        elif all_nans:
            out = all_nan_return


    if not keepdims:
        a_max = np.squeeze(a_max, axis=axis)
    out += a_max

    if return_sign:
        return out, sgn
    else:
        return out

if __name__ == "__main__":
    X = np.random.randn(100, 200) * 2000  # make sure to overflow a log(sum(exp()))
    assert np.allclose(logsumexp(X, axis=1), nan_logsumexp(X, axis=1))

    X = np.random.randn(100,200)
    X[:,-1] = -np.inf # is underflown to zero in logsumexp, i.e., does not count
    logsumexp_inf = logsumexp(X, axis=1)
    X[:,-1] =  np.nan
    logsumexp_nan = nan_logsumexp(X, axis=1)
    assert np.allclose(logsumexp_inf, logsumexp_nan)








def nan_softmax(a, axis=None):
    """ Similar to nan_logsumexp, except that some arguments are unavailable"""
    logsumexp_result =  nan_logsumexp(a, axis=axis,
                             keepdims=True, return_sign=False, b=None, all_nan_return=np.nan)

    return np.exp(a - logsumexp_result)



