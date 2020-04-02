import sys

sys.path.append("/home/emmanuel/code/destructive-deep-learning")
sys.path.insert(0, "/Users/eman/Documents/code_projects/destructive-deep-learning")

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


n_samples = 10000
dataset = "classic"
data = get_toy_data(dataset=dataset, n_samples=n_samples)

save_path = "/home/emmanuel/code/destructive-deep-learning/pics/rbig/"
save_path = "/Users/eman/Documents/code_projects/destructive-deep-learning/pics/rbig/"
plot_scatter(data, name=save_path + "raw.png")
plot_histograms(data, name=save_path + "raw_hist.png")

# Parameters
bins = "auto"  # bin estimation (uses automatric method)
bounds = 0.1  # percentage extension of the support
alpha = 1e-10  # regularization parameter for hist (not needed for RBIG)


# ==================================
# Step I - Marginal Uniformization
# ==================================
# Choose the Histogram estimator that converts the data X to uniform U(0,1)
univariate_estimator = HistogramUnivariateDensity(bounds=bounds, bins=bins, alpha=alpha)

# Marginally uses histogram estimator
marginal_uniformization = IndependentDensity(univariate_estimators=univariate_estimator)

# Creates "Destructor" D_theta_1
uniform_density = IndependentDestructor(marginal_uniformization)

# Fit and transform data P_x ---> U_z
mU_approx = uniform_density.fit_transform(data)

# Sampling

# Data Together
plot_scatter(mU_approx, name=save_path + "1_mu.png")

# Histograms
plot_histograms(mU_approx, name=save_path + "1_mu_hist.png")

# ===================================
# Step II - Marginal Gaussianization
# ===================================

# Choose destructor D_theta_2 that converts data
marginal_gaussianization = IndependentInverseCdf()


#  U_z ---> G_z
mG_approx = marginal_gaussianization.fit_transform(mU_approx)

# Fit and transform data Data Together
plot_scatter(mG_approx, name=save_path + "2_mg.png")

# Histograms
plot_histograms(mG_approx, name=save_path + "2_mg_hist.png")

# ======================================
# Step III - Rotation
# ======================================

# Choose a linear projection to rotate the features (PCA) "D_theta_3"
rotation = LinearProjector(linear_estimator=PCA())


# Fit and transform data G_z ----> R (G_z)
G_approx = rotation.fit_transform(mG_approx)

# Data Together
plot_scatter(G_approx, name=save_path + "3_rot.png")

# Histograms
plot_histograms(G_approx, name=save_path + "3_rot_hist.png")
