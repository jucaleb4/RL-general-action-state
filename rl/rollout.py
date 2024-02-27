""" 
Rollout object 
- Stores most recent rollout of (s, a, r, s', term, trnc)
- Computes basic statistics, simple evaluations, operators, etc.
"""
import warnings 

import numpy as np

import gymnasium as gym

def get_space_dimension(space):
    """ Get dimension and data type of the space. We currently only support Box and Dicrete """
    space_is_box = isinstance(space, gym.spaces.box.Box)
    space_is_dis = isinstance(space, gym.spaces.discrete.Discrete)
    assert space_is_box or space_is_dis, f"Space type {type(space)} unsupported (must be Box or Discrete)"

    assert hasattr(space, "_shape"), "Missing `_shape` attribute"
    assert hasattr(space, "dtype"), "Missing `dtype` attribute"

    dim = space._shape[0] if len(space._shape) > 0 else 1
    data_type = space.dtype

    return (dim, data_type)

def get_enlarged_array(arr, added_elems):
    """ Enlarges array by adding `added_elems` new elements """
    if len(arr.shape) > 1:
        return np.vstack((arr, np.zeros((added_elems, arr.shape[1]), dtype=arr.dtype)))
    return np.append(arr, np.zeros((added_elems, arr.shape[1]), dtype=arr.dtype))

class Rollout:
    def __init__(self, obs_space, action_space):
        # Process the environment's state and action space
        (obs_dim, obs_type) = get_space_dimension(obs_space)
        (action_dim, action_type) = get_space_dimension(action_space)

        # Create arrays to store the batch information (row by row)
        self.iter_ct = 0
        self.capacity = 128
        self.s_batch = np.zeros((self.capacity+1, obs_dim), dtype=obs_type) # +1 to store the initial stage
        self.a_batch = np.zeros((self.capacity, action_dim), dtype=action_type)
        self.r_batch = np.zeros((self.capacity, 1), dtype=float)
        self.term_batch = np.zeros((self.capacity, 1), dtype=bool)
        self.trunc_batch = np.zeros((self.capacity, 1), dtype=bool)

    def add_reset_data(self, s):
        """ Adds data returned from reset 
        :param s: current state
        """
        if self.iter_ct > 0:
            warnings.warn("Rollout did not call `clear_batch` before adding reset data")
            
        self.s_batch[0] = s

    def add_step_data(self, s, a, r, term, trunc):
        """ Adds data returned from step 
        :param s: current state after step
        :param a: action
        :param r: reward
        :param term: environment terminated
        :param trunc: environment truncated
        """
        self.s_batch[self.iter_ct+1] = s
        self.a_batch[self.iter_ct] = a
        self.r_batch[self.iter_ct] = r
        self.term_batch[self.iter_ct] = term
        self.trunc_batch[self.iter_ct] = trunc
        self.iter_ct += 1

        # enlarge
        if self.iter_ct == self.capacity:
            self.s_batch = get_enlarged_array(self.s_batch, self.capacity)
            self.a_batch = get_enlarged_array(self.a_batch, self.capacity)
            self.r_batch = get_enlarged_array(self.r_batch, self.capacity)
            self.term_batch = get_enlarged_array(self.term_batch, self.capacity)
            self.trunc_batch = get_enlarged_array(self.trunc_batch, self.capacity)
            self.capacity *= 2

    def clear_batch(self):
        """ 'Clears' batch by setting `iter_ct` to zero and resetting the first
        point. Easier than zeroing out or re-allocated space 
        """
        self.iter_ct = 0
        self.s_batch[0] = self.s_batch[self.iter_ct]

    def get_state(self, t: int):
        assert 0 <= t <= self.iter_ct, f"Invalid index {t} with only {self.iter_ct} data collected"
        return self.s_batch[t]
