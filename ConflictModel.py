import statsmodels.api as sm
import numpy as np
import prettyplotlib as pplt
from matplotlib import path
import SliceSampler
from mpl_toolkits.basemap import Basemap

class ConflictModel(object):
    def __init__(self, uganda_data):
        self.uganda_data = uganda_data
        self.kde_model = None

    def fit_kde(self):
        self.kde_model = sm.nonparametric.KDEMultivariate(self.uganda_data[['LATITUDE', 'LONGITUDE']], 'cc')

    def plot_kde(self, fig=None, ax=None):
        if not self.kde_model:
            self.fit_kde()

        if not fig:
            fig = plt.gcf()
            fig.set_size_inches(12, 12)
        if not ax:
            ax = plt.gca()

        # set up the domain
        margin = 0.5
        lats = np.linspace(self.uganda_data.LATITUDE.min() - margin, self.uganda_data.LATITUDE.max() + margin, 400)
        lons = np.linspace(self.uganda_data.LONGITUDE.min() - margin, self.uganda_data.LONGITUDE.max() + margin, 400)
        LA, LO = np.meshgrid(lats, lons)

        # get alpha-gradient co
        reds = pplt.brewer2mpl.get_map('YlOrRd', 'Sequential', 9, reverse=False).mpl_colormap
        reds._init()
        reds._lut[:,-1] = np.linspace(0.1, 0.8, reds.N+3)

        # plot the contour of the kde model. 
        ax.contourf(LO, LA, self.kde_model.pdf([LA.ravel(), LO.ravel()]).reshape(400, 400), 15, zorder=20, cmap=reds)
        return fig, ax

    def fit_ar_poisson_glm(self):
        pass

    def draw_samples(self, n_samples, burnin=100, thin=10):
        if not self.kde_model:
            self.fit_kde()

        # pick starting point
        start_x = np.array([[0., 31.]])

        # get region for restricting samples
        margin = 0.5
        m = Basemap(llcrnrlon=self.uganda_data.LONGITUDE.min() - margin,
            llcrnrlat=self.uganda_data.LATITUDE.min() - margin,
            urcrnrlon=self.uganda_data.LONGITUDE.max() + margin,
            urcrnrlat=self.uganda_data.LATITUDE.max() + margin,
            resolution='l',
            area_thresh=10000)
        m.readshapefile("data/regions/UGA_adm0", "regions", drawbounds=True)
        for xy, info in zip(m.regions, m.regions):
            p = path.Path(xy)

        slice_samples, _ = SliceSampler.SliceSampler.mvslice(self.kde_model.pdf, 
            start_x,
            sample_size=n_samples, 
            burnin=burnin, 
            thin=thin, 
            confirm_region=p)
        
        # we get lat, long; we want x, y
        return np.fliplr(slice_samples)

    def sample_fake_uniform(self, x, y, size):
        """ use this for the same interface as np.random.uniform
        """
        n, _ = size
        return self.draw_samples(n)
    
    