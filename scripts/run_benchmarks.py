# import numpy as np
# import tqdm
# import time

# # Choose the Histogram estimator that converts the data X to uniform U(0,1)
# univariate_estimator = HistogramUnivariateDensity(
#     bounds=wandb.config.bounds, bins=wandb.config.bins, alpha=wandb.config.alpha
# )

# # Marginally uses histogram estimator
# marginal_uniformization = IndependentDensity(univariate_estimators=univariate_estimator)

# # Creates "Destructor" D_theta_1
# uniform_density = IndependentDestructor(marginal_uniformization)

# # Choose destructor D_theta_2 that converts data
# marginal_gaussianization = IndependentInverseCdf()


# # Choose a linear projection to rotate the features (PCA) "D_theta_3"
# rotation = LinearProjector(linear_estimator=PCA(svd_solver="randomized"))

# n_layers = 50
# rbig_flow = DeepDestructor(
#     canonical_destructor=[uniform_density, marginal_gaussianization, rotation],
#     n_canonical_destructors=n_layers,
#     base_dist="gaussian",
# )

# n_samples = [100, 1_000, 10_000, 100_000, 1_000_000]
# n_features = 10

# times = []

# pbar = tqdm.tqdm(n_samples, desc="Samples")
# with pbar:
#     for isamples in pbar:

#         # generate grid
#         x_eval = np.random.randn(isamples, n_features)

#         # ==================
#         # RBIG Implementation
#         # ==================

#         # evaluate timings
#         t0 = time.time()
#         _ = rbig_flow.fit_transform(x_eval)
#         t1 = time.time() - t0

#         # append times
#         times.append(t1)
