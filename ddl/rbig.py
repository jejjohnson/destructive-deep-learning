import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin, clone
from ddl.independent import (
    IndependentDensity,
    IndependentDestructor,
    IndependentInverseCdf,
)
from ddl.linear import LinearProjector
from ddl.base import CompositeDestructor
from ddl.deep import DeepDestructor, DeepDestructorCV
from sklearn.decomposition import PCA
from ddl.univariate import HistogramUnivariateDensity


def get_rbig_model(bins="auto", bounds=0.1, alpha=1e-10):

    # ================================ #
    # Step I - Marginal Uniformization
    # ================================ #

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

    # ================================== #
    # Step II - Marginal Gaussianization
    # ================================== #

    # Choose destructor D_theta_2 that converts data
    marginal_gaussianization = IndependentInverseCdf()

    # =================== #
    # Step III - Rotation
    # =================== #

    # Choose a linear projection to rotate the features (PCA) "D_theta_3"
    rotation = LinearProjector(linear_estimator=PCA())

    # ==================== #
    # Composite Destructor
    # ==================== #

    # Composite Destructor
    rbig_model = CompositeDestructor(
        [
            clone(uniform_density),  # Marginal Uniformization
            clone(marginal_gaussianization),  # Marginal Gaussianization
            clone(rotation),  # Rotation (PCA)
        ]
    )

    return rbig_model


def get_deep_rbig_model(
    n_layers=100, bins="auto", bounds=0.1, alpha=1e-10, random_state=123
):

    rbig_model = get_rbig_model(bins=bins, bounds=bounds, alpha=alpha)

    # Initialize Deep RBIG model
    deep_rbig_model = DeepDestructor(
        canonical_destructor=rbig_model,
        n_canonical_destructors=n_layers,
        random_state=random_state,
    )

    return deep_rbig_model


def get_rbig_cvmodel(
    n_layers=50,
    bins="auto",
    bounds=0.1,
    alpha=1e-10,
    random_state=123,
    init=None,
    **kwargs
):
    if init is not None:
        # Choose the Histogram estimator that converts the data X to uniform U(0,1)
        univariate_estimator = HistogramUnivariateDensity(
            bounds=bounds, bins=bins, alpha=alpha
        )

        # Marginally uses histogram estimator
        marginal_uniformization = IndependentDensity(
            univariate_estimators=univariate_estimator
        )

        # Creates "Destructor" D_theta_1
        init = IndependentDestructor(marginal_uniformization)

    deep_rbig_model = get_deep_rbig_model(
        n_layers=n_layers,
        bins=bins,
        bounds=bounds,
        alpha=alpha,
        random_state=random_state,
    )

    # Initialize Deep RBIG CV Model
    deep_rbig_cvmodel = DeepDestructorCV(
        init_destructor=clone(init), canonical_destructor=deep_rbig_model, **kwargs
    )
    return deep_rbig_cvmodel


def get_rbig_cmodel(bins="auto", bounds=0.1, alpha=1e-10):
    # ================================ #
    # Step I - Marginal Uniformization
    # ================================ #

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

    # ================================== #
    # Step II - Marginal Gaussianization
    # ================================== #

    # Choose destructor D_theta_2 that converts data
    marginal_gaussianization = IndependentInverseCdf()

    # =================== #
    # Step III - Rotation
    # =================== #

    # Choose a linear projection to rotate the features (PCA) "D_theta_3"
    rotation = LinearProjector(linear_estimator=PCA())

    # ==================== #
    # Composite Destructor
    # ==================== #

    # Composite Destructor
    rbig_model = CompositeDestructor(
        [
            clone(marginal_gaussianization),  # Marginal Gaussianization
            clone(rotation),  # Rotation (PCA)
            clone(uniform_density),  # Marginal Uniformization
        ]
    )

    return rbig_model


def get_deep_rbig_cmodel(
    n_layers=100, bins="auto", bounds=0.1, alpha=1e-10, random_state=123
):

    rbig_cmodel = get_rbig_cmodel(bins=bins, bounds=bounds, alpha=alpha)

    # Initialize Deep RBIG model
    deep_rbig_model = DeepDestructor(
        canonical_destructor=rbig_cmodel,
        n_canonical_destructors=n_layers,
        random_state=random_state,
    )

    return deep_rbig_model
