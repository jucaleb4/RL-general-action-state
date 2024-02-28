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
    assert hasattr(space, "_shape"), "Missing `_shape` attribute"
    assert hasattr(space, "dtype"), "Missing `dtype` attribute"

    space_is_box = isinstance(space, gym.spaces.box.Box)
    space_is_dis = isinstance(space, gym.spaces.discrete.Discrete)
    assert space_is_box or space_is_dis, f"Space type {type(space)} unsupported (must be Box or Discrete)"

    dim = space._shape[0] if len(space._shape) > 0 else 1
    data_type = space.dtype
    n_vals = space.n if space_is_dis else -1

    return (dim, data_type, n_vals)

def add_rows_array(arr):
    """ 
    Returns copy of `arr` with doubled number of rows to (1d or 2d) array
    """
    assert len(arr.shape) <= 2, "Currently only support at most 2 axes (given {len(arr.shape)})"
    if len(arr.shape) == 1:
        return np.pad(arr, (0, len(arr)))
    return np.pad(arr, ((0, arr.shape[0]), (0,0)))

def native_type(x):
    """ Returns native python type. See:
        - https://github.com/numpy/numpy/issues/2951
        - https://stackoverflow.com/questions/9452775/converting-numpy-dtypes-to-native-python-types
    """
    # trick: https://stackoverflow.com/questions/12569452/how-to-identify-numpy-types-in-python
    if type(x).__module__ == np.__name__:
        return type(x.item())
    return type(x)

class Rollout:
    def __init__(self, env, gamma):
        assert 0 <= gamma <= 1, f"gamma={gamma} not in [0,1]"
        
        # discount factor
        self.gamma = gamma 

        # Process the environment's state and action space
        obs_space = env.observation_space
        action_space = env.action_space
        (obs_dim, obs_type, n_states) = get_space_dimension(obs_space)
        (action_dim, action_type, n_actions) = get_space_dimension(action_space)

        self.n_states = n_states
        self.n_actions = n_actions

        # Create arrays to store the batch information (row by row)
        capacity = 32
        self.iter_ct = 0
        self.reset_iter_ct = 0
        self.s_batch = np.zeros((capacity, obs_dim), dtype=obs_type)  
        self.a_batch = np.zeros((capacity, action_dim), dtype=action_type)
        self.r_batch = np.zeros(capacity, dtype=float)
        self.terminate_batch = np.zeros(capacity, dtype=bool)
        self.truncate_batch = np.zeros(capacity, dtype=bool)
        self.reset_steps = np.zeros(capacity, dtype=int)

    def set_gamma(self, gamma):
        self.gamma = gamma

    def add_step_data(self, state, action, reward=0, terminate=False, truncate=False):
        """ Adds data returned from step 
        :param s: current state after step
        :param a: action
        :param r: reward
        :param terminate: environment terminated
        :param truncate: environment truncated
        """
        assert native_type(state) == native_type(self.s_batch.flat[0]), \
            f"type(state)={native_type(state)} does not match {native_type(self.s_batch[0])}"
        assert native_type(action) == native_type(self.a_batch.flat[0]), \
            f"type(action)={native_type(action)} does not match {native_type(self.a_batch[0])}"
        assert native_type(reward) in [int, float], \
            f"type(reward)={native_type(reward)} not numerical"
        assert native_type(terminate) == bool, \
            f"type(terminate)={native_type(terminate)} not bool"
        assert native_type(truncate) == bool, \
            f"type(truncate)={native_type(truncate)} not bool"

        self.s_batch[self.iter_ct] = state
        self.a_batch[self.iter_ct] = action
        self.r_batch[self.iter_ct] = reward
        self.terminate_batch[self.iter_ct] = terminate
        self.truncate_batch[self.iter_ct] = truncate
        self.iter_ct += 1

        if self.iter_ct == len(self.s_batch):
            self.s_batch = add_rows_array(self.s_batch)
            self.a_batch = add_rows_array(self.a_batch)
            self.r_batch = add_rows_array(self.r_batch)
            self.terminate_batch = add_rows_array(self.terminate_batch)
            self.truncate_batch = add_rows_array(self.truncate_batch)

        if terminate or truncate:
            self.reset_steps[self.reset_iter_ct] = self.iter_ct
            self.reset_iter_ct += 1
            if self.reset_iter_ct == len(self.reset_steps):
                self.reset_steps = add_rows_arr(self.reset_steps)

    def clear_batch(self):
        """ 
        Artifically clears the batch by resetting iteration counter. We do not
        remove any data point. 
        """
        self.iter_ct = 0
        self.reset_iter_ct = 0

    def get_state(self, t):
        return self.s_batch[t]

    def compute_enumerated_stateaction_value(self):
        """ Compute advatange function 
        # TODO: Compute GAE (https://arxiv.org/pdf/1506.02438.pdf)

        :return Q: 2d array of estimated Q function (via Monte Carlo)
        :return Ind: 2d array of booleans whether we visited (s,a0
        """
        Q = np.zeros((self.n_states, self.n_actions), dtype=float)
        total_num_first_visits = np.zeros((self.n_states, self.n_actions), dtype=int)

        if self.iter_ct == 0:
            warnings.warn("Have not run rollout, returning null values..")
            return (Q, total_num_first_visits > 0)

        for i in range(self.reset_iter_ct+1):
            # episode duration = [episode_start, episode_end)
            if i == 0:
                episode_start = 0
            else:
                episode_start = self.reset_steps[i-1]
            if i < self.reset_iter_ct:
                episode_end = self.reset_steps[i]
            else:
                episode_end = self.iter_ct

            # when two consecutive steps are termination/truncates
            if episode_start+1 >= episode_end:
                continue

            # rewards appear on second step 
            rwds = self.r_batch[episode_start+1:episode_end]
            weight_factors = np.power(self.gamma, np.arange(len(rwds)))
            weighted_rwds = np.multiply(weight_factors, rwds)
            cumulative_weighted_rwds = np.cumsum(weighted_rwds[::-1])[::-1]
            # normalize discount because `cumulative_weighted_rwds` weighs k-th
            # state-action value by gamma^(k-1); we do not want this scaling
            np.divide(cumulative_weighted_rwds, weight_factors, out=cumulative_weighted_rwds)
            
            visited_this_episode = np.zeros(total_num_first_visits.shape, dtype=int)
            for dt in range(episode_end-episode_start-1):
                # TODO Provide mapping from raw state-action to state-action indices
                s = self.s_batch[dt+episode_start][0]
                a = self.a_batch[dt+episode_start][0]

                if visited_this_episode[s,a] == 0:
                    visited_this_episode[s,a] = 1
                    num_visits = total_num_first_visits[s,a]
                    theta = 1./(num_visits+1)
                    # average 
                    Q[s,a] = (1.-theta)*Q[s,a] + theta*cumulative_weighted_rwds[dt]

            total_num_first_visits += visited_this_episode

        Ind = total_num_first_visits > 0
        return (Q, Ind)
