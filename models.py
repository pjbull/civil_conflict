import numpy as np
from scipy.spatial.distance import euclidean


class Configuration(object):
    """ class to hold state and utility functions for cities TSP """

    def __init__(self, n=101, domain=(0, 50), time_per_dist=1, default_needs=(10, 5, 2)):
        """ set contants, create random cities, and precompute lookup matrices """
        ind_hq = 0
        
        # set constants
        self.n = n
        self.domain = domain
        self.time_per_dist = time_per_dist

        # set the headquarters in the middle of the map
        self.hq = np.array([np.mean(domain)] * 2, dtype=np.float)

        # randomly seed a configuration of (n + 1) locations, cities[0, 0] is HQ
        self.cities = np.random.uniform(*self.domain, size=(self.n, 2))

        # center the headquarters
        self.cities[ind_hq, :] = self.hq

        # set the sizes of each location
        self.sizes = np.random.randint(low=1, high=11, size=self.n)
        self.sizes[ind_hq] = 1

        # randomly set the proportional needs of each of location
        self.needs = np.random.dirichlet(default_needs, self.n)
        self.needs[ind_hq] *= 0
        
        # set a placeholder for the scaled (absolute need) of each location
        self.scaled_needs = np.empty_like(self.needs)

        # compute the distances between locations
        self.distances = np.zeros((n, n), dtype=np.float)
        self.times = np.zeros((n, n), dtype=np.float)
        self.precompute()

    def precompute(self):
        """ precompute distance, flight connection, and time matrices """
        for i in xrange(self.n):
            for j in xrange(self.n):
                self.distances[i, j] = euclidean(self.cities[i, :], self.cities[j, :])

        # use the distances to set up the time matrix
        self.times = self.distances * self.time_per_dist
        
        # scale the relative need vectors by the size of the location to
        # get the absolute number of each needed type of supply
        self.scaled_needs = np.array([(self.needs[i] * s) for i, s in enumerate(self.sizes)])

    def itercities(self, perm):
        """ iterate through i, j indices of cities traveled to """
        for k in xrange(perm.size - 1):
            yield perm[k], perm[k + 1]


class Route(object):
    
    def __init__(self, configuration, load_at_hq):
        self.configuration = configuration
        self.load_at_hq = load_at_hq
        self._current_load = load_at_hq.copy()
    
    def has_repeats(self, perm):
        """ determine if a permutation visits any location consecutively """
        return np.where(perm[:-1]==perm[1:])[0].size > 0
    
    def refill_truck(self):
        """ reload the truck based on reload settings """
        self._current_load = self.load_at_hq.copy()
        
    def unmet_needs(self, perm):
        """ determine how many aid shortfalls occur given truck's load """
        # initialize the array of unmet needs
        unmet_needs = np.array([0, 0, 0], dtype=np.float)
        
        # iterate through the cities
        for ind in perm:
            
            if ind == 0:  # at HQ, reload
                self.refill_truck()
            
            # subtract whatever the current location consumed
            self._current_load -= self.configuration.scaled_needs[ind]
            
            # record the shortfalls
            unmet_needs += (self._current_load < 0).astype(int) * self._current_load

            # we can't have negative load on the truck, so set equal to 0
            self._current_load = self._current_load.clip(0, np.inf)
            
        return 2 * unmet_needs
            
    def dist(self, perm):
        """ return the total distance traveled """
        return np.sum(self.configuration.distances[i, j] for i, j in 
                      self.configuration.itercities(perm))
        
    def loss(self, perm):
        """ calculate the energy of a given order of cities """
        return self.dist(perm) + self.unmet_needs(perm).sum() ** 2
    
    def perturb(self, perm, max_hq_stops=12, p_add_hq_stop=0.05, p_remove_hq_stop=0.1):
        """ propose a new solution by perturbing current arrangement """

        assert perm[0] == 0  # make sure we're starting at HQ
        new = perm.copy()
        
        working_inds = np.arange(1, perm.size)

        # choose two random indices to switch and switch them
        inds = np.random.choice(working_inds, size=2, replace=False)
        new[inds] = new[inds][::-1]
        
        # maybe add a zero (visit to HQ to reload)
        if (new == 0).sum() <= max_hq_stops and np.random.rand() < p_add_hq_stop:
            j = np.random.choice(working_inds)
            new = np.concatenate((new[:j], np.array([0.]), new[j:]))
            
        # maybe remove a zero (visit to HQ to reload)
        if (new == 0).sum() > 1 and np.random.rand() < p_remove_hq_stop:
            extra_zero_indices = np.where(new==0)[0][1:]
            j = np.random.choice(extra_zero_indices)
            new = np.concatenate((new[:j], new[j+1:]))
            
        # remove any repeats (deterministic -- must happen)
        while(self.has_repeats(new)):
            j = np.where(new[:-1]==new[1:])[0][0]
            new = np.concatenate((new[:j], new[j+1:]))

        return new