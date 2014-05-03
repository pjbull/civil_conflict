import numpy as np
from scipy.spatial.distance import euclidean
from haversine import haversine


class Configuration(object):
    """ class to hold state and utility functions for cities TSP """

    def __init__(self, 
                 n=21, 
                 domain=(0, 50), 
                 time_per_dist=1, 
                 default_needs=(10, 5, 2), 
                 sample_method=np.random.uniform):
        """ set contants, create random cities, and precompute lookup matrices """

        # set constants
        self.n = n
        self.domain = domain
        self.time_per_dist = time_per_dist

        # randomly seed a configuration of (n + 1) locations, cities[0, 0] is HQ
        self.cities = sample_method(*self.domain, size=(self.n, 2))

        # set the sizes of each location
        self.sizes = np.random.randint(low=1, high=11, size=self.n)

        # randomly set the proportional needs of each of location
        self.needs = np.random.dirichlet(default_needs, self.n)

        # set a placeholder for the scaled (absolute need) of each location
        self.scaled_needs = np.empty_like(self.needs)

        # put HQ at the center of the map
        center = np.array([np.mean(self.domain)] * 2, dtype=np.float)
        self.set_hq(center)

        # compute the distances between locations
        self.distances = np.zeros((n, n), dtype=np.float)
        self.times = np.zeros((n, n), dtype=np.float)
        self.compute_matrices()

    def set_hq(self, location):
        """ set the location of headquarters """

        # center the headquarters
        self.cities[0, :] = location
        self.sizes[0] = 1
        self.needs[0] *= 0

    def compute_matrices(self):
        """ precompute distance, flight connection, and time matrices """
        for i in xrange(self.n):
            for j in xrange(self.n):
                # use the haversine (Earth-surface) distance (in km)
                self.distances[i, j] = haversine(self.cities[i, :], self.cities[j, :])

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
        self._load_at_hq = load_at_hq
        self._current_load = load_at_hq.copy()

    @staticmethod
    def has_repeats(perm):
        """ determine if a permutation visits any location consecutively """
        return np.where(perm[:-1] == perm[1:])[0].size > 0
    
    def refill_truck(self):
        """ reload the truck based on reload settings """
        self._current_load = self._load_at_hq.copy()
        
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
        
        non_hq_indices = np.arange(1, perm.size)

        # choose two random indices to switch and switch them
        random_indices = np.random.choice(non_hq_indices, size=2, replace=False)
        new[random_indices] = new[random_indices][::-1]
        
        # maybe add a zero (visit to HQ to reload)
        if (new == 0).sum() <= max_hq_stops and np.random.rand() < p_add_hq_stop:
            j = np.random.choice(non_hq_indices)
            new = np.concatenate((new[:j], np.array([0.]), new[j:]))
            
        # maybe remove a zero (visit to HQ to reload)
        if (new == 0).sum() > 1 and np.random.rand() < p_remove_hq_stop:
            extra_zero_indices = np.where(new==0)[0][1:]
            j = np.random.choice(extra_zero_indices)
            new = np.concatenate((new[:j], new[j+1:]))
            
        # remove any repeats (deterministic -- must happen)
        while Route.has_repeats(new):
            j = np.where(new[:-1] == new[1:])[0][0]
            new = np.concatenate((new[:j], new[j+1:]))

        return new


class TSPRoute(Route):
    """ don't revisit HQ at all for reloading """
    
    def perturb(self, perm, max_hq_stops=12, p_add_hq_stop=0.05, p_remove_hq_stop=0.1):
        """ propose a new solution by perturbing current arrangement """

        assert perm[0] == 0  # make sure we're starting at HQ
        new = perm.copy()
        
        non_hq_indices = np.arange(1, perm.size)

        # choose two random indices to switch and switch them
        random_indices = np.random.choice(non_hq_indices, size=2, replace=False)
        new[random_indices] = new[random_indices][::-1]
        
        return new


class HqLocator(object):
    """ class which also moves HQ around """

    def __init__(self, n=100, x_domain=(29., 36.), y_domain=(-1., 4), proposed_location=None):

        self.n = n
        self.x_domain = x_domain
        self.y_domain = y_domain
        self.draw_new_random_cities()

        self._proposed_location = proposed_location
        if proposed_location is None:
            self._proposed_location = np.array([5, 5])

    def draw_new_random_cities(self):
        """ draw new configuration of cities from our distribution """
        xs = np.random.uniform(*self.x_domain, size=self.n)
        ys = np.random.uniform(*self.y_domain, size=self.n)
        self.cities = np.hstack((xs, ys))

    def loss(self, hq_location):
        dists = np.array([haversine(self.cities[i, :], hq_location)
                          for i in xrange(self.n)])
        return (dists ** 2).sum()

    def proposed_location(self, *args):
        """ utility method to keep track of HQ location from the SA algorithm """
        return self._proposed_location

    def perturb(self, hq_location, **kwargs):
        """ this time we'll switch the headquarters around location too """

        self._proposed_location = hq_location.copy()

        if np.random.rand() < 0.2:
            self.draw_new_random_cities()

        dmin, dmax = self.x_domain
        dist = (self.x_domain[1] - self.x_domain[0])/20
        self._proposed_location[0] += np.random.uniform(-dist, dist)
        self._proposed_location[0] = np.clip(self._proposed_location[0], dmin, dmax)
        
        dmin, dmax = self.y_domain
        dist = (self.y_domain[1] - self.y_domain[0])/20
        self._proposed_location[1] += np.random.uniform(-dist, dist)
        self._proposed_location[1] = np.clip(self._proposed_location[1], dmin, dmax)

        return self._proposed_location

    
class HqLocatorUganda(HqLocator):
    """ sample cities from the Uganda distribution """

    def draw_new_random_cities(self):
        self.cities = slicer_api(*self.x_domain, size=(self.n, 2))