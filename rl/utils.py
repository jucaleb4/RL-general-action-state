import warnings 

import numpy as np

import gymnasium as gym

def safe_normalize_row(arr, tol=1e-16):
    """ 
    Normalizes rows (in place) so rows sum to 1, then round small numbers to 0
    
    :param tol: cut off for smallest value for row sum
    """
    assert tol <= 1e-8, f"tol={tol} too large (<= 1e-8)"
    assert np.min(arr) >= 0, "Given arr with negative value"
    policy_row_sum = np.atleast_2d(np.sum(arr, axis=1)).T
    rows_whose_sum_is_near_zero = np.where(policy_row_sum < tol)
    arr[rows_whose_sum_is_near_zero] += tol
    np.divide(arr, np.atleast_2d(np.sum(arr, axis=1)).T, out=arr)
    np.round(arr, decimals=max(32, -np.log(tol)/np.log(10)), out=arr)

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
        raise Exception("Unsupported type {space} (only support Discrete and Box)")

def get_space_cardinality(space):
    """ If finite, return space cardinality """
    if isinstance(space, gym.spaces.discrete.Discrete):
        return space.n
    elif isinstance(space, gym.spaces.box.Box):
        assert space.dtype == int, "Unsupported box type {space.dtype} (must be int/in64)"
        assert np.all(space.high-space.low >= 0)
        return np.prod(space.high-space.low+1)
    else:
        raise Exception("Space {space} is not Discrete or Box")

def remap_vec_to_int(obs, space):
    """ 
    Map discrete spaces (from Discrete or Box) to a single integer with origin
    0
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
        raise Exception("Unsupported space {type(space)} for finite spaces")
