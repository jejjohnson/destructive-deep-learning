from typing import Union, Dict, Optional
import numpy as np
from sklearn.base import BaseEstimator
from .independent import (
    IndependentDensity,
    IndependentDestructor,
    IndependentInverseCdf,
)
from .linear import LinearProjector, RandomOrthogonalEstimator
from .deep import DeepDestructorIT
from sklearn.decomposition import PCA
from .univariate import HistogramUnivariateDensity


def get_rbig_model(
    bins: Union[str, int] = "auto",
    bounds: Union[int, float] = 0.1,
    alpha: float = 1e-10,
    rotation: str = "pca",
    rotation_kwargs: Dict = {},
    tol_layers: int = 15,
    threshhold: float = 0.25,
    random_state: Optional[int] = 123,
) -> BaseEstimator:
    """Rotation-Based Iterative Gaussianization (RBIG).
    This algorithm learns the function transforms any multidimensional data into
    a Gaussian distribution. Just like other density destructors here, once the
    transformation is learned, you can sample and calculate probabilities. Due to
    the formulation of RBIG, we have access to some information theoretic
    measures such as total correlation, entropy and multivariate mutual information.

    Parameters
    ----------
    bins : int or sequence of scalars or str, (default='auto')
        Same ase the parameter of :func:`numpy.histogram`.



    bounds : float or array-like of shape (2,), (default=0.1)
        Specification for the finite bounds of the histogram. Bounds can be
        percentage extension or a specified interval [a,b].

    alpha : float
        Regularization parameter corresponding to the number of
        pseudo-counts to add to each bin of the histogram. This can be seen
        as putting a Dirichlet prior on the empirical bin counts with
        Dirichlet parameter alpha.

    rotation_type : str, (default='pca')
        The rotation applied to the marginally Gaussian-ized data at each iteration.
        - 'pca'     : a principal components analysis rotation (PCA)
        - 'random'  : random rotations

    rotation_kwargs : dict, optional (default=None)
        Any extra keyword arguments that you want to pass into the rotation
        algorithms (i.e. ICA or PCA). See the respective algorithms on
        scikit-learn for more details.

    tol_layers : int, (default=15)
        the number of layers allowed to be zero representing the change in marginal
        entropy with each iteration. A higher tolerance will result in a more
        Gaussian latent space.

    threshhold : float, (default=0.25)
        the p-value used to assess if the marginal information between
        layers is 0. A lower tolerance will result in more layers as it
        allows much smaller differences of marginal entropy whereas a higher
        p-value will result in less layers as it will ignore bigger
        differences between marginal entropy.

    random_state : int, optional (default=123)
        Control the seed for any randomization that occurs in this algorithm.
    """
    # ==================================
    # Step I - Marginal Uniformization
    # ==================================
    # Choose the Histogram estimator that converts the data X to uniform U(0,1)
    univariate_estimator = HistogramUnivariateDensity(
        bounds=bounds, bins=bins, alpha=alpha
    )

    # Marginally uses histogram estimator
    marginal_uniformization = IndependentDensity(
        univariate_estimators=univariate_estimator
    )

    # Creates "Destructor" D_theta_1
    uniform_density = IndependentDestructor(marginal_uniformization)

    # ===================================
    # Step II - Marginal Gaussianization
    # ===================================

    # Choose destructor D_theta_2 that converts data
    marginal_gaussianization = IndependentInverseCdf()

    # ======================================
    # Step III - Rotation
    # ======================================

    # Choose a linear projection to rotate the features
    if rotation.lower() == "pca":
        linear_estimator = PCA(random_state=random_state, **rotation_kwargs)
    elif rotation.lower() == "random":
        linear_estimator = RandomOrthogonalEstimator(
            random_state=random_state, **rotation_kwargs
        )
    else:
        raise ValueError(f"Unrecognized linear estimator: {rotation}")
    rotation = LinearProjector(linear_estimator=linear_estimator)

    # ======================================
    # All Steps - Deep Density Destructor
    # ======================================
    # rbig_block = CompositeDestructor(
    #     destructors=
    # )
    rbig_flow = DeepDestructorIT(
        canonical_destructor=[uniform_density, marginal_gaussianization, rotation],
        base_dist="gaussian",
        tol_layers=tol_layers,
        threshhold=threshhold,
    )
    return rbig_flow
