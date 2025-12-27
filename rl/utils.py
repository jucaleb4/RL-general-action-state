import warnings 

import itertools

import numpy as np
import numpy.linalg as la

import gymnasium as gym

def safe_mean(arr):
    return -1 if len(arr) == 0 else np.mean(arr)

def safe_std(arr):
    return -1 if len(arr) == 0 else np.std(arr)

def safe_normalize_row(arr, tol=1e-32):
    """ 
    Normalizes rows (in place) so rows sum to 1
    
    :param tol: cut off for smallest value for row sum
    """
    assert len(arr.shape) == 2, f"dim(arr)={len(arr.shape)} != 2"
    assert tol <= 1e-16, f"tol={tol} too large (<= 1e-16)"
    assert np.min(arr) >= 0, "Given arr with negative value"

    policy_row_sum = np.atleast_2d(np.sum(arr, axis=1)).T
    # TODO: Does this cause any breaks?
    rows_whose_sum_is_near_zero = np.where(policy_row_sum < tol)[0]
    arr[rows_whose_sum_is_near_zero] += tol
    np.divide(arr, np.atleast_2d(np.sum(arr, axis=1)).T, out=arr)

    assert not np.any(np.isnan(arr)), "inf in arr"
    assert np.min(arr) >= 0, f"min(arr)={np.min(arr)} < 0"
    assert np.allclose(np.sum(arr, axis=1), 1.), "rows not sum to 1"

def get_space_property(space):
    """ Retruns if space is finite and its dimensionality

    :return is_finite: 
    :return dim: dimension as tuple
    :return dtype:
    """
    if isinstance(space, gym.spaces.discrete.Discrete):
        return (True, (1,), int)
    elif isinstance(space, gym.spaces.box.Box):
        is_finite = space.dtype == int
        return (is_finite, space.shape, space.dtype)
    else:
        raise Exception(f"Unsupported type {space} (only support Discrete and Box)")

def get_space_cardinality(space):
    """ If finite, return space cardinality """
    if isinstance(space, gym.spaces.discrete.Discrete):
        return space.n
    elif isinstance(space, gym.spaces.box.Box):
        assert space.dtype == int, f"Unsupported box type {space.dtype} (must be int/in64)"
        assert np.all(space.high-space.low >= 0)
        return np.prod(space.high-space.low+1)
    else:
        raise Exception(f"Space {space} is not Discrete or Box")

def vec_to_int(obs, space):
    """ 
    Maps discrete values (from Discrete or Box) to an integer with origin 0
    """
    if isinstance(space, gym.spaces.discrete.Discrete):
        return obs[0]
    elif isinstance(space, gym.spaces.box.Box):
        low  = space.low
        high = space.high
        diff = high-low+1
        multiplier = np.cumprod(diff)[:len(obs)-1]
        obs_as_int = (obs[0]-low[0]) + np.dot((obs-low)[1:], multiplier)
        return obs_as_int
    else:
        raise Exception(f"Unsupported space {type(space)} for finite spaces")

def int_to_vec(i, space):
    """ 
    Maps integers (origin 0) to discrete values (from Discrete or Box)
    """
    if isinstance(space, gym.spaces.discrete.Discrete):
        return np.array([i])
    elif isinstance(space, gym.spaces.box.Box):
        low  = space.low
        high = space.high
        diff = high-low+1
        dim = len(diff)
        obs = np.zeros(dim, dtype=int)
        obs[0] = i % diff[0]
        multiplier = np.cumprod(diff)[:len(obs)-1]
        obs[1:] = np.mod(np.floor(np.divide(i, multiplier)), diff[1:])
        return obs
    else:
        raise Exception(f"Unsupported space {type(space)} for finite spaces")

def pretty_print_gridworld(arr, space):
    for i,a in enumerate(arr):
        print(f"  {int_to_vec(i, space)} : {a}")
    
def rbf_kernel(n, dim, rng):
    """ 
    Creates random affine transformation via random Fourier features. 
    Link: https://people.eecs.berkeley.edu/~brecht/papers/07.rah.rec.nips.pdf
    """
    W = 2**0.5 * rng.normal(size=(dim, n))
    b = 2*np.pi*rng.random(size=dim)
    return (W,b)

def get_rbf_features(W, b, X, sigma):
    """ Construct feature map from X (i-th rows is x_i) 
    - W is dxn matrix (d is feature dimension, n is input dim)
    - X is mxn matrix (m is number of input points)
    """
    B = np.repeat(b[:, np.newaxis], X.shape[0], axis=1)
    return np.sqrt(2./W.shape[0]) * np.cos(sigma * W @ X.T + B)

class RunningStat():
    """ From: https://www.johndcook.com/blog/standard_deviation/ """
    def __init__(self, n):
        # online dynamic (mean and sample variance) accumulators
        self._M = np.zeros(n, dtype=float)
        self._S = np.zeros(n, dtype=float)
        self._k = 0

        # non-dynamic accumualtors; `update()` sets to most recent _M and _S
        self.M = np.copy(self._M)
        self.S = np.copy(self._S)
        self.k = 0

        # If we have not updated, pass back variance of 1
        self.S_0 = np.ones(n, dtype=float)

    def push(self, x):
        self._k += 1
        old_diff = x-self._M
        self._M += (old_diff)/self._k
        if self._k > 0:
            self._S += np.multiply(old_diff, x-self._M)

    def clear(self):
        self.M[:] = 0
        self.S[:] = 0
        self.k = 0
        self._M[:] = 0
        self._S[:] = 0
        self._k = 0

    def update(self):
        self.M = self._M
        self.S = self._S
        self.k = self._k

    @property
    def mean(self):
        return self.M

    @property
    def var(self, tol=1e-3):
        """ For vars which are very small, do not change variance """
        if self.k < 1:
            return self.S_0

        S = np.copy(self.S)
        idx_where_var_small = np.where(self.S < tol)[0]
        S[idx_where_var_small] = 1
        return S/(self.k-1.)

def tsallis_policy_update(cum_grad, eta_t, lam=None):
    """ Policy update for Tsallis Inf. Solves up to accuracy 1e-6 if warm-start
    and 1e-8 for cold start.

    We want to solve
    $$
        min_{x in Delta_n} { <cum_grad, x> + eta_t^{-1} Psi(x) }
    $$
    where (with a=0.5)
    $$
        Psi(x) := sumlimits_{i=1}^n -(x_i^a)/a
    $$
    By KKT conditions, we want to find ({x_i}_i, y=lam) such 
    $$
        cum_grad_i - eta_t^{-1}/(1-a) * x_i^(a-1) + y = 0, for all i
    $$
    and x is probability simplex. Solving for x,
    $$
        x_i = [ (1-a) * eta_t * (cum_grad_i + y) ]^(1/(a-1))
    $$
    So we compute this x_i and do binary search on y until this quantity x_i is a probability simplex

    :param cum_grad: cumulative gradient
    :param eta_t: step size
    :param lam: lam to use for warm start
    """
    tol = 1e-6
    get_x_star = lambda y : np.power(np.maximum(tol, 0.5 * eta_t * (cum_grad + y)), -2)
    max_num_newton_steps = 12

    # try warm-star with Newton's method
    if lam is not None:
        for t in range(max_num_newton_steps):
            prev_lam = lam
            w = get_x_star(lam)
            f = np.sum(w)-1
            df = -2.*0.5*eta_t*np.sum(np.power(w, 1.5))
            lam = lam - f/df
            if la.norm(lam-prev_lam) <= tol:
                u = get_x_star(lam)
                x_star = u/np.sum(u)
                return (x_star, lam)
        
    # cold start (if no warm_start flag or warm_start failed after 12 iterations)
    direction = 0

    # find whether lam should be positive or negative
    u = get_x_star(0)
    if abs(np.sum(u) - 1) <= tol:
        x_star = u/np.sum(u)
        return x_star
    elif np.sum(u) > 1:
        direction = 1
    else:
        direction = -1
            
    # exponential search
    lam = direction
    no_exp_search = True
    u = get_x_star(lam)
    while (direction == 1 and np.sum(u) > 1) or (direction == -1 and np.sum(u) < 1):
        no_exp_search = False
        lam *= 2
        u = get_x_star(lam)

    # binary search 
    if direction == 1:
        lo = 0 if no_exp_search else lam/(2.) 
        hi = lam # value that made sum(u) <= 1
        while hi-lo > tol:
            lam = (hi+lo)/2
            u = get_x_star(lam)
            if abs(np.sum(u) - 1)<=tol: break
            elif np.sum(u) < 1: hi = lam # makes the sum larger
            else: lo = lam
        best_lam = hi
    else:
        lo = lam # value that made sum(u) >= 1
        hi = 0 if no_exp_search else lam/(2.)
        while hi-lo > tol:
            lam = (hi+lo)/2
            u = get_x_star(lam)
            if abs(np.sum(u) - 1)<=tol: break
            elif np.sum(u) < 1: lo = lam
            else: hi = lam
        best_lam = lo

    u = get_x_star(best_lam)
    x_star = u/np.sum(u)
    # if la.norm(u-x_star) >= 1e-1:
    #     import ipdb; ipdb.set_trace()
    #     pass
    return (x_star, best_lam)

def get_all_grid_pts(lows, highs):
    """ Given n-dimensional grid (where n=len(lows)), return grid points of the form

        (i_1,...,i_n) 

    for all permutations of (i_1,...,i_n) where i_j = lows[j],...,lows[j]-1.
    For example if lows=[0,-1,1] and highs[1,1,3], then we return

        [
            [0,-1, 1],
            [0,-1, 2],
            [0, 0, 1],
            [0, 0, 2],
        ]

    :params lows: numpy array of min values along each axis
    :params highs: numpy array of max values (inclusive) along each axis (match length of lows)
    :return: 2d array of all grid points
    """

    if len(lows) != len(highs):
        return None
    # returns meshgrid in dimension of len(lows)x(highs-lows)
    # asterisk is to convert the list to multiple inputs to meshgrid (i.e. unpacking)
    meshgrid_separate_dim = np.meshgrid(*[np.arange(lows[i], highs[i]+1) for i in range(len(lows))])
    # concatenates all dimensions
    # see: https://numpy.org/doc/stable/reference/generated/numpy.c_.html
    meshgrid_arr = np.c_[meshgrid_separate_dim]
    # reshape so the dimension len(lows) is 2nd dim, and collapse other axes to 1st dim
    return np.reshape(meshgrid_arr, newshape=(len(lows),-1)).T
