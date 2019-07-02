import sys

sys.path.append("/home/emmanuel/code/destructive-deep-learning")


from data import get_toy_data
from visualize import plot_histograms, plot_layer_scores, plot_scatter
from ddl.univariate import HistogramUnivariateDensity
from ddl.independent import (
    IndependentInverseCdf,
    IndependentDensity,
    IndependentDestructor,
)
from ddl.linear import LinearProjector
from ddl.base import CompositeDestructor
from ddl.deep import DeepDestructor, DeepDestructorCV
from sklearn.decomposition import PCA


# ========================================
# MODEL I - Randomized Linear Projections
# ========================================

# ========================================
# MODEL II - RBIG
# ========================================
