from matplotlib import pyplot as plt
import seaborn as sns


def plot_locations(configuration, ax, palette='Set1'):
    """ plot cities and connections """
    # plot the cities

    cs = sns.color_palette(palette)

    ax.scatter(configuration.cities[0, 0],
               configuration.cities[0, 1],
               c=cs[0], s=100, marker='s')

    ax.scatter(configuration.cities[1:, 0],
               configuration.cities[1:, 1],
               s=configuration.sizes * 10,
               c=cs[1], zorder=2)

    # set sensible plotting limits
    mind, maxd = configuration.domain
    delta = (maxd - mind) * 0.05
    ax.set_xlim(mind - delta, maxd + delta)
    ax.set_ylim(mind - delta, maxd + delta)

    return ax


def plot_route(configuration, perm, ax, palette='Set1'):
    """ plot a traveling salesman route """
    plot_locations(configuration, ax)

    cs = sns.color_palette(palette)

    # plot the starting point as a green circle
    ax.scatter(*configuration.cities[perm[0], :],
               c=cs[0], marker='.', label='start',
               markersize=40.0, alpha=0.5, zorder=1)

    for i, j in configuration.itercities(perm):
        dd = configuration.cities[j, :] - configuration.cities[i, :]
        ax.arrow(configuration.cities[i, 0],
                 configuration.cities[i, 1], dd[0], dd[1],
                 length_includes_head=True, width=0.05, color='k', zorder=3)

    return ax

def plot_route_with_colors(configuration, perm, ax, palette='Set2'):
    """ plot a traveling salesman route """
    plot_locations(configuration, ax)

    # plot the starting point as a green circle
    ax.plot(*configuration.cities[perm[0], :],
            c='g', marker='.', label='start',
            markersize=40.0, alpha=0.5, zorder=1)

    sortie = 0
    color_palette = sns.color_palette(palette, 12)
    for i, j in configuration.itercities(perm):

        label = None
        
        # if at HQ, get the next color for plotting a new sortie
        if i == 0:
            c = color_palette[sortie]
            sortie += 1
            label = 'sortie %d' % sortie
            ax.plot(None, None, ls='-', c=c, label=label)

        dd = configuration.cities[j, :] - configuration.cities[i, :]
        ax.arrow(configuration.cities[i, 0],
                 configuration.cities[i, 1], dd[0], dd[1],
                 length_includes_head=True, width=0.05,
                 color=c, zorder=3)

    return ax