""" 
Rollout object 
- Stores most recent rollout of (s, a, r, s', term, trnc)
- Computes basic statistics, simple evaluations, operators, etc.
"""
import warnings 

import numpy as np

import gymnasium as gym

from rl.utils import vec_to_int
from rl.utils import get_space_property
from rl.utils import get_space_cardinality

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
    if isinstance(x, np.ndarray):
        x = x.flat[0]
    # trick: https://stackoverflow.com/questions/12569452/how-to-identify-numpy-types-in-python
    if type(x).__module__ == np.__name__:
        return type(x.item())
    return type(x)

class Rollout:
    def __init__(self, env, **kwargs):
        if "gamma" in kwargs:
            self.gamma = kwargs["gamma"]
            assert 0 <= self.gamma <= 1, f"gamma={gamma} not in [0,1]"
        else:
            self.gamma = 1.

        # cut-off for Monte-Carlo estimation
        if "cutoff" not in kwargs:
            if self.gamma == 1:
                self.cutoff = 100
            else:
                self.cutoff = int(1+1./(1-self.gamma))
            # warnings.warn(f"No cutoff specified, setting to {self.cutoff}")
        else:
            self.cutoff = max(0, kwargs["cutoff"])
        assert self.cutoff > 0, "Monte-Carlo cutoff {self.cutoff} not positive"

        self.use_gae = kwargs.get("use_gae", False)
        self.gae_lambda = kwargs.get("gae_lambda", 1.)
        if self.use_gae:
            print(f"Running GAE with gae_lambda={self.gae_lambda}")

        # Process the environment's state and action space
        self.obs_space = env.observation_space
        self.action_space = env.action_space
        (_, obs_dim, obs_type) = get_space_property(self.obs_space)
        (_, action_dim, action_type) = get_space_property(self.action_space)

        assert len(obs_dim) <= 1, "Only support 1D observation spaces, given {len(obs_dim)}D"
        assert len(action_dim) <= 1, "Only support 1D action spaces, given {len(action_dim)}D"

        # Create arrays to store the batch information (row by row)
        capacity = 32
        self.time_ct = 0
        self.reset_time_ct = 1 # first step is a reset step
        self.s_batch = np.zeros((capacity,) + obs_dim, dtype=obs_type)  
        self.a_batch = np.zeros((capacity,) + action_dim, dtype=action_type)
        self.r_batch = np.zeros(capacity, dtype=float)
        self.v_batch = np.copy(self.r_batch)
        self.done_batch = np.zeros(capacity, dtype=bool)
        self.reset_steps = np.zeros(capacity, dtype=int)

        self.s_raw_batch = np.copy(self.s_batch)
        self.a_raw_batch = np.copy(self.a_batch)
        self.r_raw_batch = np.copy(self.r_batch)

        self.all_ep_cum_rwd = []
        self.all_ep_len = []
        self.curr_ep_len = 0
        self.curr_ep_cum_rwd = 0

        self.triggered_restart_warning = False

        # for validation
        self.curr_estQ = np.zeros(env.action_space.n, dtype=float)
        self.curr_avgA = np.zeros(env.action_space.n, dtype=float)
        self.curr_avgV = 0.
        self.curr_action = 0 # which action is currently being estimated
        self.num_V_mc_est = 0 # how many low-bias Monte-Carlo estimates we have had
        self.num_Q_mc_est = 0 # how many low-bias Monte-Carlo estimates we have had
        self.actions_est_arr = np.zeros(env.action_space.n, dtype=int) # num of estimators for each action
        self.all_ep_avg_V = []
        self.all_ep_avg_agap = []

    def set_gamma(self, gamma):
        self.gamma = gamma

    def add_step_data(
            self, 
            state, 
            action, 
            reward, 
            done,
            value=0,
            s_raw=None,
            a_raw=None,     
            r_raw=None,
            # pi_at_s=None,
            skip_validation=False,
        ):
        """ Adds data returned from step 
        :param s: current state after step
        :param a: action
        :param r: reward
        :param terminate: environment terminated
        :param truncate: environment truncated
        """
        assert native_type(state) == native_type(self.s_batch), \
            f"type(state)={native_type(state)} does not match {native_type(self.s_batch[0])}"
        assert native_type(action) == native_type(self.a_batch), \
            f"type(action)={native_type(action)} does not match {native_type(self.a_batch[0])}"
        assert native_type(reward) in [int, float], \
            f"type(reward)={native_type(reward)} not numerical"
        assert native_type(done) == bool, \
            f"type(done)={native_type(done)} not bool"

        s_raw = state if s_raw is None else s_raw
        a_raw = action if a_raw is None else a_raw
        r_raw = reward if r_raw is None else r_raw
        if self.time_ct > 0:
            if self.done_batch[self.time_ct-1] and done and not self.triggered_restart_warning:
                warnings.warn("Recieved two consecutive done flags, which can distort value function. Make sure to restart environment")
                self.triggered_restart_warning = True

        self.s_batch[self.time_ct] = state
        self.a_batch[self.time_ct] = action
        self.r_batch[self.time_ct] = reward
        self.done_batch[self.time_ct] = done
        self.v_batch[self.time_ct]= value
        self.s_raw_batch[self.time_ct] = s_raw
        self.a_raw_batch[self.time_ct] = a_raw
        self.r_raw_batch[self.time_ct] = r_raw
        self.time_ct += 1

        # for validation
        if self.curr_ep_len == 0:
            self.curr_action = a_raw

        # NEW: For discounted reward
        self.curr_ep_cum_rwd += (self.gamma**self.curr_ep_len) * r_raw
        ep_rew = self.curr_ep_cum_rwd
        # self.curr_ep_cum_rwd += r_raw
        self.curr_ep_len += 1

        if done:
            # process for validation
            if not skip_validation:
                self.actions_est_arr[self.curr_action] += 1
                _n = float(self.actions_est_arr[self.curr_action])
                alpha = (_n-1.)/_n
                Q_est_a = self.curr_estQ[self.curr_action]
                self.curr_estQ[self.curr_action] = alpha*Q_est_a + (1.-alpha)*(-ep_rew)
                alpha_V = float(self.num_V_mc_est)/(1.+self.num_V_mc_est)
                alpha_Q = float(self.num_Q_mc_est)/(1.+self.num_Q_mc_est)
                self.curr_avgV = alpha_V*self.curr_avgV + (1.-alpha_V)*(-ep_rew)
                self.num_V_mc_est += 1
                temp_curr_avgV = np.dot(self.actions_est_arr, self.curr_estQ)/np.sum(self.actions_est_arr)
                temp_curr_A = self.curr_estQ - temp_curr_avgV
                temp_curr_avgA = alpha_Q*self.curr_avgA + (1.-alpha_Q)*temp_curr_A

                self.all_ep_avg_V.append(self.curr_avgV)
                self.all_ep_avg_agap.append(np.max(-temp_curr_avgA))
                if np.min(self.actions_est_arr) > 0:
                    # reset and update
                    self.curr_avgA[:] = temp_curr_avgA
                    self.num_Q_mc_est += 1
                    self.actions_est_arr[:] = 0
                    self.curr_estQ[:] = 0.

            moving_avg = 0
            self.all_ep_cum_rwd.append(self.curr_ep_cum_rwd)

            self.all_ep_len.append(self.curr_ep_len)
            self.curr_ep_cum_rwd = 0
            self.curr_ep_len = 0

        if self.time_ct == len(self.s_batch):
            self.s_batch = add_rows_array(self.s_batch)
            self.a_batch = add_rows_array(self.a_batch)
            self.r_batch = add_rows_array(self.r_batch)
            self.v_batch = add_rows_array(self.v_batch)
            self.done_batch = add_rows_array(self.done_batch)
            self.s_raw_batch = add_rows_array(self.s_raw_batch)
            self.a_raw_batch = add_rows_array(self.a_raw_batch)
            self.r_raw_batch = add_rows_array(self.r_raw_batch)

    def clear_batch(self):
        self.time_ct = 0
        self.s_batch[:] = 0
        self.a_batch[:] = 0
        self.r_batch[:] = 0
        self.v_batch[:] = 0
        self.done_batch[:] = 0
        self.s_raw_batch[:] = 0
        self.a_raw_batch[:] = 0
        self.r_raw_batch[:] = 0

    def get_state(self, t):
        assert 0 <= abs(t) <= len(self.s_batch)
        return self.s_batch[t]

    def get_ep_rwds(self):
        return self.all_ep_cum_rwd

    def get_ep_lens(self):
        return self.all_ep_len

    def get_ep_avg_V(self):
        return self.all_ep_avg_V

    def get_ep_avg_agap(self):
        return self.all_ep_avg_agap

    def reset_validation(self):
        self.curr_estQ[:] = 0
        self.curr_avgA[:] = 0
        self.curr_avgV = 0.
        self.curr_action = 0
        self.num_V_mc_est = 0
        self.num_Q_mc_est = 0
        self.actions_est_arr[:]
        self.all_ep_avg_V = []
        self.all_ep_avg_agap = []

    def compute_all_stateaction_value(self):
        """ Compute advantage function 
        # TODO: Compute GAE (https://arxiv.org/pdf/1506.02438.pdf)

        :return Q: 2d array of estimated Q function (via Monte Carlo)
        :return Ind: 2d array of booleans whether we visited (s,a0
        """
        n_states = get_space_cardinality(self.obs_space)
        n_actions = get_space_cardinality(self.action_space)
        q_est_enum = np.zeros((n_states, n_actions), dtype=float)
        adv_est_enum = np.copy(q_est_enum)
        num_visit_enum = np.zeros((n_states, n_actions), dtype=int)
        already_visited = np.copy(num_visit_enum).astype("bool")

        output = self.get_montecarlo_stateaction_value(0, False)
        (q_est, adv_est, s_visited, a_visited) = output
        s_visited = s_visited.astype("int")
        a_visited = a_visited.astype("int")
        assert 0 <= np.min(s_visited)
        assert np.max(s_visited) < n_states
        assert 0 <= np.min(a_visited)
        assert np.max(a_visited) < n_actions

        for i,(s,a) in enumerate(zip(s_visited, a_visited)):
            if not already_visited[s,a]:
                num_visit_enum[s,a] += 1
                already_visited[s,a] = True
                alpha = 1./num_visit_enum[s,a]

                q_est_enum[s,a] = (1-alpha)*q_est_enum[s,a] + alpha*q_est[i]
                adv_est_enum[s,a] = (1-alpha)*adv_est_enum[s,a] + alpha*adv_est[i]

            # if episode is done, allow averaging
            if self.done_batch[i]:
                already_visited.fill(False)

        return q_est_enum, adv_est_enum, num_visit_enum

    def get_est_stateaction_value(self, last_pred_value=0, last_state_done=False):
        if self.use_gae:
            return self.get_gae_stateaction_value(last_pred_value, last_state_done)
        else:
            return self.get_montecarlo_stateaction_value(last_pred_value, last_state_done)

    def get_montecarlo_stateaction_value(self, last_pred_value, last_state_done):
        """
        Returns Q-function, advantage function with the corresponding
        state-action pairs

        :param last_pred_value: estimated cost-to-go for last state in rollout 
        :param last_state_terminated: last state terminated (not truncated)
        """
        q_est = np.zeros(self.time_ct, dtype=float)
        adv_est = np.copy(q_est)
        # if truncated, we already appended cost-to-go
        # if termianted, not cost-to-go
        cum_rwd = last_pred_value * (1-int(last_state_done))
        for t in reversed(range(self.time_ct)):
            # if t == self.time_ct-1:
            #     q_est[t] = self.r_batch[t] + self.gamma * cum_rwd
            # else:
            #     q_est[t] = self.r_batch[t] + self.gamma * self.v_batch[t+1]
            # adv_est[t] = q_est[t] - self.v_batch[t]
                
            # if we recieved done, reset cum_rwd for new episode
            if t < self.time_ct-1 and self.done_batch[t]:
                cum_rwd = 0
            cum_rwd = self.r_batch[t] + self.gamma * cum_rwd
            q_est[t] = cum_rwd
            adv_est[t] = cum_rwd - self.v_batch[t]

        s_visited = np.copy(self.s_batch[:self.time_ct])
        a_visited = np.copy(self.a_batch[:self.time_ct])

        return (q_est, adv_est, s_visited, a_visited)

    def get_gae_stateaction_value(self, last_pred_value, last_state_done):
        """
        Returns Q-function, advantage function with the corresponding
        state-action pairs

        :param last_pred_value: estimated cost-to-go for last state in rollout 
        :param last_state_terminated: last state terminated (not truncated)
        """
        q_est = np.zeros(self.time_ct, dtype=float)
        adv_est = np.copy(q_est)
        # if truncated, we already appended cost-to-go
        # if termianted, not cost-to-go
        last_gae_lam = 0
        next_v = last_pred_value * (1-int(last_state_done))
        for t in reversed(range(self.time_ct)):
            # if we recieved done, reset last_gae_lam for new episode
            if t < self.time_ct-1 and self.done_batch[t]:
                last_gae_lam = 0
            # also the advantage function
            delta = self.r_batch[t] + self.gamma * next_v - self.v_batch[t]
            last_gae_lam = delta + self.gamma * self.gae_lambda * last_gae_lam
            adv_est[t] = last_gae_lam
            q_est[t] = last_gae_lam + self.v_batch[t]

        s_visited = np.copy(self.s_batch[:self.time_ct])
        a_visited = np.copy(self.a_batch[:self.time_ct])

        return (q_est, adv_est, s_visited, a_visited)
