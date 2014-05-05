import matplotlib.pyplot as plt
import prettyplotlib as pplt
from mpl_toolkits.basemap import Basemap

class ConflictMap(object):
    def __init__(self, uganda_data):
        self.uganda_data = uganda_data

        # get colors for map
        cmap = pplt.brewer2mpl.get_map("Set3", "Qualitative", 7)
        self.land = cmap.hex_colors[6]
        self.roads = cmap.hex_colors[2]
        self.events = cmap.hex_colors[3]
        self.water = cmap.hex_colors[4]

        # get base for map
        margin = 0.5
        self.m = Basemap(llcrnrlon=uganda_data.LONGITUDE.min() - margin,
            llcrnrlat=uganda_data.LATITUDE.min() - margin,
            urcrnrlon=uganda_data.LONGITUDE.max() + margin,
            urcrnrlat=uganda_data.LATITUDE.max() + margin,
            resolution='l',
            area_thresh=10000)

        self.m.readshapefile("data/regions/UGA_adm0", "regions0", drawbounds=True)
        self.m.readshapefile("data/regions/UGA_adm1", "regions1", drawbounds=True)
        self.m.readshapefile("data/water/UGA_water_areas_dcw", "water", drawbounds=True)
        
    def plot_map(self, outline_only=False, fig=None, ax=None):
        if not fig:
            fig = plt.gcf()
            fig.set_size_inches(12, 12)
        if not ax:
            ax = plt.gca()

        if outline_only:
            for xy, info in zip(self.m.regions0, self.m.regions0):
                poly = plt.Polygon(xy, facecolor=self.land, alpha=0.7, zorder=10)
                ax.add_patch(poly)

        else:
            # roads as a one-off file load
            self.m.readshapefile("data/roads/Uganda_Roads", "roads", drawbounds=True, linewidth=2, color=self.roads, zorder=12)

            for xy, info in zip(self.m.regions1, self.m.regions1):
                poly = plt.Polygon(xy, facecolor=self.land, alpha=0.7, zorder=10)
                ax.add_patch(poly)

            for xy, info in zip(self.m.water, self.m.water):
                poly = plt.Polygon(xy, facecolor=self.water, alpha=1, zorder=11)
                ax.add_patch(poly)


            ax.scatter(self.uganda_data.LONGITUDE, self.uganda_data.LATITUDE, color=self.events, alpha=0.7, zorder=15)

        return fig, ax
