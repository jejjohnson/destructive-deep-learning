import matplotlib.pyplot as plt

plt.style.use("ggplot")


def plot_scatter(data, probs=None, name=None):
    fig, ax = plt.subplots()
    if probs is not None:
        h = ax.scatter(data[:, 0], data[:, 1], s=1, c=probs)
        cbar = plt.colorbar(h)

    else:
        ax.scatter(data[:, 0], data[:, 1], s=1)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title("Data")
    plt.tight_layout()

    if name is not None:
        fig.savefig(name)
    else:
        plt.show()
    return None


def plot_histograms(data, bins=50, name=None):

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 4))
    ax[0].set_title(r"$X$")
    ax[0].hist(data[:, 0], bins=50)
    ax[1].set_title(r"$Y$")
    ax[1].hist(data[:, 1], bins=50)
    plt.tight_layout()
    if name is not None:
        fig.savefig(name)
    else:
        plt.show()

    return None


def plot_layer_scores(scores, name=None):

    # Score Layers
    fig, ax = plt.subplots()

    ax.plot(scores.cumsum())
    if name is not None:
        fig.savefig(name)
    else:
        plt.show()

    return None
