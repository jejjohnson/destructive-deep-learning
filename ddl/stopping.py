import numpy as np
from scipy import stats


def condition(loss: np.ndarray, tol_layers: int, n_layers: int) -> float:
    if n_layers < tol_layers:
        return True
    else:
        layers = loss[-tol_layers:]
        diff = np.sum(np.abs(layers))
        return diff > 0.0


def condition_stop(loss: np.ndarray, tol_layers: int, n_layers: int) -> float:
    if n_layers < tol_layers:
        return True
    else:
        return False


def diff_negentropy(Z: np.ndarray, X: np.ndarray, X_slogdet: np.ndarray) -> float:
    """Difference in negentropy between invertible transformations
    Function to calculate the difference in negentropy between a variable
    X and an invertible transformation

    Parameters
    ----------
    Xtrans : np.ndarray, (n_samples, n_features)
        the transformed variable
    X : np.ndarray, (n_samples, n_features)
        the variable
    X_slogdet: np.ndarray, (n_features, n_features)
        the log det jacobian of the transformation
    Returns
    -------
    x_negent : float
        the difference in negentropy between transformations
    """
    # calculate the difference in negentropy
    delta_neg = np.mean(
        0.5 * np.linalg.norm(Z, 2) ** 2 - X_slogdet - 0.5 * np.linalg.norm(X, 2) ** 2
    )

    # return change in marginal entropy
    return np.abs(delta_neg)


def negative_log_likelihood(Z: np.ndarray, X_slogdet: np.ndarray) -> float:
    # calculate log probability in the latent space
    Z_logprob = stats.norm().logpdf(Z)

    # calculate the probability of the transform
    X_logprob = Z_logprob.sum(axis=1) + X_slogdet.sum(axis=1)

    # return the nll
    return np.mean(X_logprob)
