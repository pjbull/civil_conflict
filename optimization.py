import numpy as np


def sim_anneal(energy, perturb, n0, ntrial=100000, t0=2.0, thermo=0.9,
               reanneal=1000, verbose=True, other_func=None):
    print_str = 'reannealing; i[{}] exp(dE/t)[{}] eprev[{}], enew[{}]'

    temp = t0
    n = n0
    e_prev = energy(n)

    # initialize our value holders
    energies = [e_prev]
    other = []
    if other_func:
        other = [other_func(n)]

    for i in xrange(ntrial):

        # get proposal and calculate energy
        propose_n = perturb(n)
        e_new = energy(propose_n)
        deltaE = e_prev - e_new

        # decide whether to accept the proposal
        if e_new < e_prev or np.random.rand() < np.exp(deltaE / temp):
            e_prev = e_new
            n = propose_n
            energies.append(e_new)

            if other_func:
                other.append(other_func(n))

        # stop computing if the solution is found
        if e_prev == 0:
            break

        # reanneal if necessary
        if (i % reanneal) == 0:

            if verbose:
                print print_str.format(i, np.exp(deltaE / temp), e_prev, e_new)

            # re-anneal up to fraction of temperature
            temp = temp * thermo

            # if temp falls below minimum, bump back up
            if temp < 0.1:
                temp = 0.5

    return n, np.array(energies), np.array(other)