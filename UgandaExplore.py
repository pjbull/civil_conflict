
# coding: utf-8

# In[1]:

get_ipython().magic(u'matplotlib inline')

# data processing
import numpy as np
import pandas as pd

# statistics
import statsmodels.api as sm

# plotting and viz
import prettyplotlib as pplt
import matplotlib.pyplot as plt
import scipy.spatial as space

import seaborn as sns

# colors
cmap = pplt.brewer2mpl.get_map("Set3", "Qualitative", 7)
land = cmap.hex_colors[6]
roads = cmap.hex_colors[2]
events = cmap.hex_colors[3]
water = cmap.hex_colors[4]

# maps
from mpl_toolkits.basemap import Basemap


# In[2]:

with open("data/ACLED-Uganda_19970101-to-20131231_final.xlsx") as f:
    uganda_file = pd.ExcelFile(f)
    
uganda_data = uganda_file.parse('Sheet1')


# In[7]:

print uganda_data.head().to_latex(columns=[u'GWNO', u'EVENT_DATE', u'TIME_PRECISION', u'EVENT_TYPE', u'LATITUDE', u'LONGITUDE', u'GEO_PRECIS', u'FATALITIES'])


# In[67]:

# setup the figure

# intitialize the map
def plot_map(outline_only=False):
    fig = plt.gcf()
    fig.set_size_inches(12, 12)
    ax = plt.gca()
    
    margin = 0.5
    m = Basemap(llcrnrlon=uganda_data.LONGITUDE.min() - margin,
                llcrnrlat=uganda_data.LATITUDE.min() - margin,
                urcrnrlon=uganda_data.LONGITUDE.max() + margin,
                urcrnrlat=uganda_data.LATITUDE.max() + margin,
                resolution='l',
                area_thresh=10000)

    # add the land
    if outline_only:
        m.readshapefile("data/regions/UGA_adm0", "regions", drawbounds=True)
        for xy, info in zip(m.regions, m.regions):
            poly = plt.Polygon(xy, facecolor=land, alpha=0.7, zorder=10)
            ax.add_patch(poly)

    else:
        m.readshapefile("data/regions/UGA_adm1", "regions", drawbounds=True)
        for xy, info in zip(m.regions, m.regions):
            poly = plt.Polygon(xy, facecolor=land, alpha=0.7, zorder=10)
            ax.add_patch(poly)

        # add the water
        m.readshapefile("data/water/UGA_water_areas_dcw", "water", drawbounds=True)
        for xy, info in zip(m.water, m.water):
            poly = plt.Polygon(xy, facecolor=water, alpha=1, zorder=11)
            ax.add_patch(poly)

        # add the roads
        m.readshapefile("data/roads/Uganda_Roads", "roads", drawbounds=True, linewidth=2, color=roads, zorder=12)

        # add the roads
        #m.readshapefile("data/refugee/North_Uganda_IDP_Camp_July2009", "refugee", color=roads, zorder=13)

        # add the events
        ax.scatter(uganda_data.LONGITUDE, uganda_data.LATITUDE, color=events, alpha=0.7, zorder=15)

    return fig, ax

fig, ax = plot_map()
    
def get_kde(uganda_data):
    k = sm.nonparametric.KDEMultivariate(uganda_data[['LATITUDE', 'LONGITUDE']], 'cc')
    return k

k = get_kde(uganda_data)

margin = 0.5
lats = np.linspace(uganda_data.LATITUDE.min() - margin, uganda_data.LATITUDE.max() + margin, 400)
lons = np.linspace(uganda_data.LONGITUDE.min() - margin, uganda_data.LONGITUDE.max() + margin, 400)

LA, LO = np.meshgrid(lats, lons)

reds = pplt.brewer2mpl.get_map('YlOrRd', 'Sequential', 9, reverse=False).mpl_colormap
reds._init()
reds._lut[:,-1] = np.linspace(0.1, 0.8, reds.N+3)

ax.contourf(LO, LA, k.pdf([LA.ravel(), LO.ravel()]).reshape(400, 400), 15, zorder=20, cmap=reds)
fig.show()


# In[157]:

def mvslice(pdf, x0, widths, sampleSize=1000, dims=2, burnin=0, thin=1, confirm_region=False, **kwargs):

    """
    :param pdf:  function we're trying to sample
    :param x0:   inital point
    :param widths:  prior for widths of our hyperrectangle
    :param sampleSize:  number of samples to generate
    :param dims:  dimension of our multivariate space
    :param burnin:  number of samples to get rid of at beginning
    :param thin:   number of samples to keep
    """
    
    x = x0.copy()
    y = np.random.uniform(low=0, high=pdf(x, **kwargs))
    samples = []
    pdf_values = []
    
    if confirm_region:
        from matplotlib import path
        m = Basemap(llcrnrlon=uganda_data.LONGITUDE.min() - margin,
            llcrnrlat=uganda_data.LATITUDE.min() - margin,
            urcrnrlon=uganda_data.LONGITUDE.max() + margin,
            urcrnrlat=uganda_data.LATITUDE.max() + margin,
            resolution='l',
            area_thresh=10000)
        m.readshapefile("data/regions/UGA_adm0", "regions", drawbounds=True)
        for xy, info in zip(m.regions, m.regions):
            p = path.Path(xy)

    # get hyperrectangle
    rectUnifs = np.random.uniform(size=dims)
    rectLefts = x - widths * rectUnifs
    rectRights = rectLefts + widths

    # Get our samples
    for i in np.arange((sampleSize + burnin)*thin):

        tr_cnt = 0
        while True:

            # new proposal
            xstarUnifs = np.random.uniform(size=dims)
            xstar = rectLefts + xstarUnifs*(rectRights - rectLefts)

            if tr_cnt % 1000 == 0 and tr_cnt > 0:
                print "Still searching at ", tr_cnt
                print "y:", y
                print "pdf:", pdf(xstar, **kwargs)
                print "xstar:", xstar
                
            if y < pdf(xstar, **kwargs):
                break
            else:

                # update rectangle
                for j in range(dims):
                    if xstar[:,j] < x[:,j]:
                        rectLefts[:,j] = xstar[:,j]
                    else:
                        rectRights[:,j] = xstar[:,j]
                        
            tr_cnt+=1


        # save our sample
        if confirm_region:
            if p.contains_points(np.fliplr(xstar)):
                samples.append(xstar[0])
                pdf_values.append(pdf(xstar, **kwargs))
            
        else:
            samples.append(xstar[0])
            pdf_values.append(pdf(xstar, **kwargs))

        # Our last sample x0 is now our proposal
        x = xstar.copy()

        # Get our new y0 for next step
        y = np.random.uniform(low=0, high=pdf(x, **kwargs))

        # reset our rectangle
        rectUnifs = np.random.uniform(size=dims)
        rectLefts = x - widths*rectUnifs
        rectRights = rectLefts + widths

    return np.array(samples)[burnin::thin], np.array(pdf_values)[burnin::thin]



# In[179]:

get_ipython().magic(u'time')
start_x = np.array([[0., 31.]])
rect_widths = np.array([10., 10.])
slice_samples, _ = mvslice(k.pdf, start_x, rect_widths, sampleSize=100, burnin=10, thin=10, 
                           confirm_region=True)


# In[180]:

fig, ax = plot_map(outline_only=True)
#pplt.scatter(ax, slice_samples[:1000,1], slice_samples[:1000,0], zorder=20, s=20, color='b', alpha=0.5) #bins=50, cmap=reds, zorder=17)
ax.hist2d(slice_samples[:,1], slice_samples[:,0], cmap=reds, bins=25)
fig.show()


# In[181]:

def run_diagnostics(samples, function=None, plots=True):
    if plots:
        xlim = (-0.5, 1.5)
        ylim = (-1.5, 1.)
        
        # plot the sample distribution
        f = plt.gcf()
        f.set_size_inches(8, 8)
        plt.hist2d(samples[:,1], samples[:,0], bins=50, cmap=reds, zorder=100)
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        
        # overlay the true function
        if function:
            plot_true_function(function, xlim, ylim)
        
        plt.show()
        
        plot_diagnostics(samples)
    
    gelman_rubin(samples)

    # gewecke
    #geweke_val = pymc.diagnostics.geweke(samples, intervals=1)[0][0][1]
    Geweke(samples)
    
    
def gelman_rubin(samples):
    # g-r conventionally uses 10 chains
    # we'll assume an appropriate burnin
    # so we can divide our chains into 10
    # seaprate ones
    m_chains = 10
    length, dims = samples.shape
    n_draws = length//m_chains
    
    # split the chain into 10 subchains
    total_length = n_draws * m_chains
    chain_draws = samples[:total_length,:].reshape(n_draws, m_chains, dims)
        
    # calculate within chain variance for each dimension
    var_j = np.var(chain_draws, axis=1)
    var_wc = np.mean(var_j, axis=0)
    
    # calculate between chain variance for each dimension
    mu_j = np.mean(chain_draws, axis=1)
    var_bc = np.var(mu_j, axis=0) * n_draws
    
    # calculate the estimated variance per dimension
    var = (1 - (1/n_draws))*var_wc + (1/n_draws)*var_bc
    
    # calculate potential scale reduction factor
    R = np.sqrt(var/var_wc)
    
    print "The Gelman-Rubin potential scale reduction factor is: ", R, " (< 1.1 indicates good mixing)"

def Geweke(trace, intervals=1, length=200, first=0.1):
    first*=len(trace)
    # take two parts of the chain. 
    # subsample lenght 
    nsl=length
    
    z =np.empty(intervals)
    for k in np.arange(0, intervals):
        # beg of each sub samples
        bega=first+k*length
        begb = len(trace)/2 + k*length
        
        sub_trace_a = trace[bega:bega+nsl]
        sub_trace_b = trace[begb:begb+nsl]
        
        theta_a = np.mean(sub_trace_a)
        theta_b = np.mean(sub_trace_b)
        var_a  = np.var(sub_trace_a)
        var_b  = np.var(sub_trace_b)
        
        z[k] = (theta_a-theta_b)/np.sqrt( var_a + var_b)
    
    print "The Geweke Diagnostic Value is: ", np.abs(z), "(< 1.96 indicates convergence)"


def plot_diagnostics(samples):
    # Samples Trace
    plot_traces(samples)
    
    # Samples Autocorrelation
    plot_acorr(samples)
    
def plot_traces(samples, sample_lim=50):
    lens, dims = samples.shape
    figs, axes = plt.subplots(dims,1)
    
    for d in range(dims):
        pplt.plot(axes[d], np.arange(sample_lim), samples[:sample_lim,d])
    

def plot_acorr(x_vals, maxlags=3):
    figs, axes = plt.subplots(1,2)
    
    # plot x autocorrelation
    axes[0].acorr(x_vals[:,0]-np.mean(x_vals[:,0]), 
                  normed=True, 
                  usevlines=False,
                  maxlags=maxlags,
                  color=c1,
                  alpha=0.8)
    axes[0].set_xlim((0, maxlags))
    axes[0].set_title(r"Autocorrelation of $longitude$")

    # plot y autocorrelation
    axes[1].acorr(x_vals[:,1]-np.mean(x_vals[:,1]),
                  normed=True,
                  usevlines=False,
                  maxlags=maxlags,
                  color=c2,
                  alpha=0.8)
    
    axes[1].set_xlim((0, maxlags))
    axes[1].set_title(r"Autocorrelation of $latitude$")
    plt.show()


# In[182]:

# colors for run_diagnostics
blues_rev = pplt.brewer2mpl.get_map('Reds', 'Sequential', 9, reverse=False).mpl_colormap
c1, c2 = pplt.brewer2mpl.get_map('Dark2', 'Qualitative', 8).mpl_colors[3:5]

plot_map(outline_only=True)
run_diagnostics(slice_samples)


# In[171]:

def getDistanceByHaversine(latitudes, longitudes):
    '''Haversine formula - give coordinates as a 2D numpy array of
    (lat_decimal,lon_decimal) pairs'''
    
    # earth's mean radius = 6,371km
    EARTHRADIUS = 6371.0
    
    # create meshgrid:
    lat, lon = np.meshgrid(latitudes, longitudes)
    
    # convert to radians
    lat *= np.pi / 180.0
    lon *= np.pi / 180.0
    
    # get transposed meshgrids for distances
    lat_T = lat.T.copy()
    lon_T = lon.T.copy()

    dlon = lon_T - lon
    dlat = lat_T - lat
    
    a = (np.sin(dlat/2))**2 + np.cos(lat) * np.cos(lat_T) * (np.sin(dlon/2.0))**2
    c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0-a))
    km = EARTHRADIUS * c
    return km


# In[172]:

# gets all pairwise distances
w = getDistanceByHaversine(uganda_data.LATITUDE, uganda_data.LONGITUDE)


# In[173]:

# add a column that has a number of days from the
# first event in our dataset to make the math easier
start_day = uganda_data.EVENT_DATE[0]

day_diff = lambda x: (x - start_day).days
uganda_data['DAYS_FROM_START'] = map(day_diff, uganda_data.EVENT_DATE)


# In[173]:




# In[174]:

# k = sm.nonparametric.KDEMultivariate(uganda_data[['LATITUDE', 'LONGITUDE']], 'cc')

# lats = np.linspace(uganda_data.LATITUDE.min(), uganda_data.LATITUDE.max(), 200)
# lons = np.linspace(uganda_data.LONGITUDE.min(), uganda_data.LONGITUDE.max(), 200)

# LA, LO = np.meshgrid(lats, lons)

# plt.contourf(LA, LO, k.pdf([LA.ravel(), LO.ravel()]).reshape(200, 200).T)


# In[176]:

dt = uganda_data[['EVENT_DATE', 'FATALITIES']]

by = lambda x: lambda y: getattr(y, x)
dri = dt.set_index('EVENT_DATE', inplace=False)
fatality_df = dri.groupby([by('year'), by('month')]).count()
fatality_df['sum'] = dri.groupby([by('year'), by('month')]).sum()

def add_lags(orig_df, num_lags):
    df = orig_df.copy()
    
    for i in range(1, num_lags+1):
        col_name = 'prev{}sum'.format(i)
        fat_name = 'prev{}fat'.format(i)
        
#         df[col_name] = np.zeros(df.shape[0])
#         df[col_name][i:] = df['sum'][:-i]
        
        df[fat_name] = np.zeros(df.shape[0])
        df[fat_name][i:] = df['FATALITIES'][:-i]
        
    return df.iloc[i:,:]


fatality_df = add_lags(fatality_df, 5)
fatality_df


# In[175]:




# In[175]:




# In[ ]:



