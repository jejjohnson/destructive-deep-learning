# Setup imports
import sys, os, warnings
import numpy as np
import matplotlib.pyplot as plt
import numpy as np

plt.style.use("ggplot")
sys.path.append("..")  # Enable importing from package ddl without installing ddl


def get_toy_data(dataset="classic", n_samples=1000, seed=123):
    rng = np.random.RandomState(seed=seed)

    x = np.abs(2 * rng.randn(n_samples, 1))
    y = np.sin(x) + 0.25 * rng.randn(n_samples, 1)
    data = np.hstack((x, y))

    return data
