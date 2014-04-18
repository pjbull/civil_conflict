from matplotlib import pyplot as plt


def plot_locations(configuration, ax):
    """ plot cities and connections """
    # plot the cities
    ax.scatter(configuration.cities[:, 0],
               configuration.cities[:, 1],
               c=configuration.sizes, cmap='Reds_r',
               s=configuration.sizes * 10, zorder=2)

    # set sensible plotting limits
    mind, maxd = configuration.domain
    delta = (maxd - mind) * 0.05
    ax.set_xlim(mind - delta, maxd + delta)
    ax.set_ylim(mind - delta, maxd + delta)


def plot_route(configuration, perm, ax):
    """ plot a traveling salesman route """
    plot_locations(configuration, ax)

    # plot the starting point as a green circle
    ax.plot(*configuration.cities[perm[0], :], c='g', marker='.', label='start',
            markersize=40.0, alpha=0.5, zorder=1)

    for i, j in configuration.itercities(perm):
        dd = configuration.cities[j, :] - configuration.cities[i, :]
        ax.arrow(configuration.cities[i, 0],
                 configuration.cities[i, 1], dd[0], dd[1],
                 length_includes_head=True, width=0.05, color='k', zorder=3)

    plt.legend(loc='best')