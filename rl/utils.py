import warnings 

import numpy as np

import gymnasium as gym

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
        assert space.dtype == int, "Unsupported box type {space.dtype} (must be int/in64)"
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
        diff = x-self.M
        self._M += (diff)/self._k
        self._S += np.dot(diff, diff)

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
