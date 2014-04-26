import numpy as np
from scipy.spatial.distance import euclidean


class Configuration(object):
    """ class to hold state and utility functions for cities TSP """

    def __init__(self, n=100, domain=(0, 50), time_per_dist=1, default_needs=(10, 5, 2)):
        """ set contants, create random cities, and precompute lookup matrices """
        # set constants
        self.n = n
        self.domain = domain
        self.time_per_dist = time_per_dist

        # randomly seed a configuration
        self.cities = np.random.uniform(*self.domain, size=(self.n, 2))
        self.sizes = np.random.randint(low=1, high=11, size=self.n)
        self.needs = np.random.dirichlet(default_needs, self.n)

        # compute the distances between locations
        self.distances = np.zeros((n, n), dtype=np.float)
        self.precompute()

    def precompute(self):
        """ precompute distance, flight connection, and time matrices """
        for i in xrange(self.n):
            for j in xrange(self.n):
                self.distances[i, j] = euclidean(self.cities[i, :], self.cities[j, :])

        # create the time matrix
        self.times = self.distances * self.time_per_dist

    def itercities(self, perm):
        """ iterate through i, j indices of cities traveled to """
        for k in xrange(perm.size - 1):
            yield perm[k], perm[k + 1]

    def time(self, perm):
        """ calculate the amount of time that the trip took """
        return np.sum(self.times[i, j] for i, j in self.itercities(perm))

    def dist(self, perm):
        """ calculate the energy of a given order of cities """
        return np.sum(self.distances[i, j] for i, j in self.itercities(perm))

    def perturb(self, perm):
        """ propose a new solution by perturbing current arrangement """
        new = perm.copy()

        # choose two random indices to switch and switch them
        inds = np.random.choice(np.arange(perm.size), size=2, replace=False)
        new[inds] = new[inds][::-1]

        return new