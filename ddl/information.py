from typing import Dict
import numpy as np
from scipy import stats
from sklearn.base import BaseEstimator, clone


def histogram_entropy(data, base=2):
    """Calculates the histogram entropy of 1D data.
    This function uses the histogram and then calculates
    the entropy. Does the miller-maddow correction

    Parameters
    ----------
    data : np.ndarray, (n_samples,)
        the input data for the entropy

    base : int, default=2
        the log base for the calculation.

    Returns
    -------
    S : float
        the entropy"""
    # get number of samples
    n_samples = np.shape(data)[0]

    # get number of bins (default square root heuristic)
    nbins = int(np.ceil(np.sqrt(n_samples)))

    # get histogram counts and bin edges
    counts, bin_edges = np.histogram(data, bins=nbins, density=False)

    # get bin centers and sizes
    bin_centers = np.mean(np.vstack((bin_edges[0:-1], bin_edges[1:])), axis=0)

    # get difference between the bins
    delta = bin_centers[3] - bin_centers[2]

    # normalize counts (density)
    pk = 1.0 * np.array(counts) / np.sum(counts)

    # calculate the entropy
    S = stats.entropy(pk, base=base)

    # Miller Maddow Correction
    correction = 0.5 * (np.sum(counts > 0) - 1) / counts.sum()

    return S + correction + np.log2(delta)


def get_tolerance_dimensions(n_samples: int):
    xxx = np.logspace(2, 8, 7)
    yyy = np.array([0.1571, 0.0468, 0.0145, 0.0046, 0.0014, 0.0001, 0.00001])
    tol_dimensions = np.interp(n_samples, xxx, yyy)
    return tol_dimensions


def information_reduction(X: np.ndarray, Y: np.ndarray, p: float = 0.25) -> float:
    """calculates the information reduction between layers
    This function computes the multi-information (total correlation)
    reduction after a linear transformation.
    
    .. math::
        Y = XW \\
        II = I(X) - I(Y)
    
    Parameters
    ----------
    X : np.ndarray, shape (n_samples, n_features)
        Data before the transformation, where n_samples is the number
        of samples and n_features is the number of features.
    
    Y : np.ndarray, shape (n_samples, n_features)
        Data after the transformation, where n_samples is the number
        of samples and n_features is the number of features
        
    p : float, default=0.25
        Tolerance on the minimum multi-information difference
        
    Returns
    -------
    II : float
        The change in multi-information
        
    Information
    -----------
    Author: Valero Laparra
            Juan Emmanuel Johnson
    """

    # calculate the marginal entropy
    hx, hy = [], []
    for ix, iy in zip(X.T, Y.T):
        hx.append(histogram_entropy(ix))
        hy.append(histogram_entropy(iy))

    # Information content
    hx, hy = np.array(hx), np.array(hy)
    I = np.sum(hy) - np.sum(hx)
    II = np.sqrt(np.sum((hy - hx) ** 2))

    # get tolerance
    n_samples, n_dimensions = X.shape
    tol_dimensions = get_tolerance_dimensions(n_samples)

    if II < np.sqrt(n_dimensions * p * tol_dimensions ** 2) or I < 0:
        I = 0.0

    return I
    # tol_dimensions = get_tolerance_dimensions(n_samples)
    # cond = np.logical_or(
    #     tol_info < np.sqrt(n_dimensions * p * tol_dimensions ** 2), delta_info < 0
    # )
    # return np.where(cond, 0.0, delta_info)


def mutual_info(
    estimator: BaseEstimator, X: np.ndarray, Y: np.ndarray, base: int = 2
) -> Dict[str, float]:
    # ==================
    # X Data
    # ==================
    x_rbig = clone(estimator)
    # fit to data
    X_trans = x_rbig.fit_transform(X)

    # ==================
    # Y Data
    # ==================
    y_rbig = clone(estimator)
    # fit to data
    Y_trans = y_rbig.fit_transform(Y)

    # ==================
    # XY Data
    # ==================
    xy_rbig = clone(estimator)
    # fit to data
    xy_rbig.fit(np.hstack([X_trans, Y_trans]))
    return {
        "total_corr_x": x_rbig.total_corr(base),
        "total_corr_y": y_rbig.total_corr(base),
        "mutual_info": xy_rbig.total_corr(base),
    }
