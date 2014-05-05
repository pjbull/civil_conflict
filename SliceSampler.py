import numpy as np
import prettyplotlib as pplt
import matplotlib.pyplot as plt
import seaborn as sns

class SliceSampler(object):
    @classmethod
    def mvslice(cls, pdf, x0, sample_size=1000, widths=np.array([10., 10.]), dims=2, burnin=0, thin=1, confirm_region=False):
            """
            :param pdf:  function we're trying to sample
            :param x0:   inital point
            :param widths:  prior for widths of our hyperrectangle
            :param sample_size:  number of samples to generate
            :param dims:  dimension of our multivariate space
            :param burnin:  number of samples to get rid of at beginning
            :param thin:   number of samples to keep
            """
            
            x = x0.copy()
            y = np.random.uniform(low=0, high=pdf(x))
            samples = []
            pdf_values = []
            
            if confirm_region:
                p = confirm_region

            # get hyperrectangle
            rectUnifs = np.random.uniform(size=dims)
            rectLefts = x - widths * rectUnifs
            rectRights = rectLefts + widths

            # Get our samples
            #for i in np.arange((sample_size + burnin)*thin):
            while len(samples) < (sample_size*thin)+burnin:
                tr_cnt = 0
                while True:
                    # new proposal
                    xstarUnifs = np.random.uniform(size=dims)
                    xstar = rectLefts + xstarUnifs*(rectRights - rectLefts)

                    if tr_cnt % 1000 == 0 and tr_cnt > 0:
                        print "Still searching at ", tr_cnt
                        print "y:", y
                        print "pdf:", pdf(xstar)
                        print "xstar:", xstar
                        
                    if y < pdf(xstar):
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
                        pdf_values.append(pdf(xstar))
                    
                else:
                    samples.append(xstar[0])
                    pdf_values.append(pdf(xstar))

                # Our last sample x0 is now our proposal
                x = xstar.copy()

                # Get our new y0 for next step
                y = np.random.uniform(low=0, high=pdf(x))

                # reset our rectangle
                rectUnifs = np.random.uniform(size=dims)
                rectLefts = x - widths*rectUnifs
                rectRights = rectLefts + widths

            return np.array(samples)[burnin::thin], np.array(pdf_values)[burnin::thin]

    @classmethod    
    def run_diagnostics(cls, samples, function=None, plots=True):
        if plots:
            reds = pplt.brewer2mpl.get_map('YlOrRd', 'Sequential', 9, reverse=False).mpl_colormap
            # plot the sample distribution
            f = plt.gcf()
            f.set_size_inches(8, 8)
            plt.hist2d(samples[:,0], samples[:,1], bins=50, cmap=reds, zorder=100)
            plt.xlabel("Longitude")
            plt.ylabel("Latitude")
            
            # overlay the true function
            if function:
                xlim = (-0.5, 1.5)
                ylim = (-1.5, 1.)
                cls.plot_true_function(function, xlim, ylim)
            
            plt.show()
            
            cls.plot_diagnostics(samples)
        
        cls.gelman_rubin(samples)

        # gewecke
        cls.geweke(samples)
        
    @classmethod       
    def gelman_rubin(cls, samples):
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

    @classmethod
    def geweke(cls, trace, intervals=1, length=200, first=0.1):
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

    @classmethod
    def plot_diagnostics(cls, samples):
        # Samples Trace
        cls.plot_traces(samples)
        
        # Samples Autocorrelation
        cls.plot_acorr(samples)
    
    @classmethod       
    def plot_traces(cls, samples, sample_lim=250):
        lens, dims = samples.shape
        figs, axes = plt.subplots(dims,1)
        
        for d in range(dims):
            pplt.plot(axes[d], np.arange(sample_lim), samples[:sample_lim,d])
        
    @classmethod
    def plot_acorr(cls, x_vals, maxlags=5):
        c1, c2 = pplt.brewer2mpl.get_map('Dark2', 'Qualitative', 8).mpl_colors[3:5]

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
