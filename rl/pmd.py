""" Policy mirror descent """
from abc import ABC
from abc import abstractmethod

import warnings 

import numpy as np
import numpy.linalg as la

import gymnasium as gym

import sklearn.linear_model as sklm

from rl import Rollout
from rl import FunctionApproximator
from rl.utils import vec_to_int
from rl.utils import safe_normalize_row
from rl.utils import rbf_kernel
from rl.utils import get_rbf_features
from rl.utils import get_space_property
from rl.utils import get_space_cardinality
from rl.utils import pretty_print_gridworld
from rl.utils import RunningStat

class PMD(ABC):
    def __init__(self, env, params):
        self.params = params

        self.env = env
        self.params = params
        self.check_params()
        self.rng = np.random.default_rng(params.get("seed", None))

        self.rollout = Rollout(env, params["gamma"])
        self.rollout_len = self.params["rollout_len"]

        self.initialized_env = False

    def learn(self, n_iter: int=100):
        self.t = 0
        try:
            self._learn(n_iter)
        except KeyboardInterrupt:
            print(f"Terminated early at iteration {self.t}")

    def _learn(self, n_iter):
        """ Runs PMD algorithm for `n_iter`s """
        for t in range(n_iter):
            print(f"=== Start iteration {t} ===")
            self.t = t

            self.params["eps_explore"] = 0.05 + 0.95 * 0.99**t

            self.collect_rollouts()

            self.policy_evaluate()

            self.policy_update()

            mean_perf = self.policy_performance()
            # print(f"[{self.t}] mean episode len={np.mean(self.rollout.get_episode_lens()):.1f}")
            # print(f"[{self.t}] mean episode rwd={np.mean(self.rollout.get_episode_rewards()):.1f}")
            # print(f"[{self.t}] mean(V)={mean_perf}")

            if self.params.get("verbose", 0) >= 2:
                print("policy:")
                pretty_print_gridworld(self.policy, self.env.observation_space)

                print("Q:")
                pretty_print_gridworld(self.Q_est, self.env.observation_space)
                print("")

        # print(f"Final policy:\n{self.policy}")

    def check_params(self):
        if "single_trajectory" not in self.params:
            warnings.warn("Did not pass in 'single_trajectory' into params, defaulting to False")
            self.params["single_trajectory"] = False
        if "rollout_len" not in self.params:
            warnings.warn("Did not pass in 'rollout_len' into params, defaulting to 1000")
            self.params["rollout_len"] = 1000

    def collect_rollouts(self):
        """ Collect samples for policy evaluation """
        self.rollout.clear_batch(keep_last_obs=self.params["single_trajectory"])

        if not self.initialized_env or not self.params["single_trajectory"]:
            (s,_) = self.env.reset()
            self.initialized_env = True
        else:
            s = self.rollout.get_state(0)

        s_ = self.normalize_obs(s)
        a = self.policy_sample(s_)
        a_ = self.normalize_action(a)
        # TODO: Should we use normalized action
        self.rollout.add_step_data(s_, a, s_raw=s, a_raw=a)
        # self.rollout.add_step_data(s_, a_, s_raw=s, a_raw=a)

        for t in range(self.rollout_len): 
            (s, r, term, trunc, _)  = self.env.step(a)

            s_ = self.normalize_obs(s)
            r_ = self.normalize_rwd(r)
            a = self.policy_sample(s_)
            a_ = self.normalize_action(a)
            # TODO: Ditto for action
            self.rollout.add_step_data(s_, a, r_, term, trunc, 
                                       s_raw=s, a_raw=a, r_raw=r)

            if term or trunc:
                (s, _) = self.env.reset()
                s_ = self.normalize_obs(s)
                a = self.policy_sample(s_)
                a_ = self.normalize_action(a)
                # TODO: Ditto for action
                self.rollout.add_step_data(s_, a, s_raw=s, a_raw=a)

    @abstractmethod
    def policy_sample(self, s):
        """ Returns an action uses current policy 

        :param s: state/observation we want an action for
        :return a: 
        """
        return self.env.action_space.sample()

    @abstractmethod
    def policy_evaluate(self):
        """ Uses samples to estimate policy value """
        raise NotImplemented

    @abstractmethod
    def policy_update(self): 
        """ Uses policy estimation from `policy_evaluate()` and updates new policy (can
            differ depending on policy approximation).
        """
        raise NotImplemented

    @abstractmethod
    def policy_performance(self) -> float: 
        """ Uses policy estimation from `policy_evaluate()` and updates new policy (can
            differ depending on policy approximation).
        """
        raise NotImplemented

    def get_stepsize_schedule(self):
        # return np.sqrt(1-self.params["gamma"])
        return (1./(self.t+1))**0.5 
        # return np.sqrt((1-self.params["gamma"])/(self.t))

    def remap_obs(self, obs): 
        """ 
        Possibly remaps state (e.g., for Discrete environments, we want states
        to start from index 0 -- while the environment can start from a
        different index.
        """
        return obs

    def remap_action(self, a): 
        """ 
        Same as `remap_obs`
        """
        return a

    def env_warmup(self):
        return

    def normalize_obs(self, s):
        return s

    def normalize_action(self, a):
        return a

    def normalize_rwd(self, r):
        return r

class PMDFiniteStateAction(PMD):
    def __init__(self, env, params):
        super().__init__(env, params)

        self.action_space = env.action_space
        self.obs_space = env.observation_space
        (action_is_finite, _, _) = get_space_property(self.action_space)
        (obs_is_finite, _, _)  = get_space_property(self.obs_space)

        assert action_is_finite, "Action space not finite"
        assert obs_is_finite, "Observation space not finite"

        # uniform policy
        self.n_states = get_space_cardinality(self.obs_space)
        self.n_actions = get_space_cardinality(self.action_space)
        self.policy = np.ones((self.n_states, self.n_actions), dtype=float)
        self.policy /= self.n_actions
        dim = self.n_states*self.n_actions

        (self.W, self.b) = rbf_kernel(dim, dim, self.rng)

    def policy_sample(self, s):
        """ Samples random action from current policy """
        s_ = vec_to_int(s, self.obs_space)
        assert 0 <= s_ < self.n_states, f"invalid s={s} ({self.n_states} states)"
        eps = self.params.get("eps_explore", 0)
        assert 0 <= eps <= 1

        pi = (1.-eps)*self.policy[s_] + eps/self.n_actions
        return self.rng.choice(self.n_actions, p=pi)

    def policy_evaluate(self):
        """ Estimates Q function and stores in self.Q_est """
        if self.params.get("train_method", "mc") == "mc":
            self.monte_carlo_Q()
        elif self.params["train_method"] == "ctd":
            self.ctd_Q()
        elif self.params["train_method"] == "vrftd":
            self.vrftd_Q()

    def monte_carlo_Q(self):
        """ 
        Constructs Q/advantage function by Monte-Carlo. If missing any data
        points, we use RKHS to fill missing points.
        """
        # check if we are missing any points 
        Q_est, Ind_est = self.rollout.compute_all_stateaction_value()

        assert Q_est.shape == Ind_est.shape
        assert len(Q_est.shape) == 2

        if self.params.get("verbose", 0) >= 1:
            visited = np.sum(Ind_est)
            total = np.prod(np.shape(Ind_est))
            print(f"Visited {visited}/{total} sa pairs")

        self.Q_est = Q_est
        self.Ind_est = Ind_est

    def ctd_Q(self):
        raise NotImplemented

    def vrftd_Q(self):
        raise NotImplemented

    def policy_update(self): 
        if self.params.get("entropy", "kl").lower() == "kl":
            self.kl_policy_update()
        elif self.params["entropy"].lower() == "tsallis":
            self.tsallis_policy_update()
        else:
            self.kl_policy_update()

    def kl_policy_update(self):
        """ Policy update with PMD and KL divergence """
        eta = self.get_stepsize_schedule()
        G = np.copy(self.Q_est)
        policy_threshold = self.n_actions**(-2/(1-self.params["gamma"]))
        self.policy_update_gradient_processing(G, policy_threshold)
        G -= np.atleast_2d(np.max(G, axis=1)).T
        self.policy = np.multiply(self.policy, np.exp(-eta*G))

        safe_normalize_row(self.policy)

    def policy_update_gradient_processing(self, G, policy_threshold):
        """ 
        Sets Q-function large (in-place) for rarefly visited state-action pairs
        """
        below_threshold_idxs = np.where(self.policy <= policy_threshold)
        if np.sum(below_threshold_idxs) == 0:
            return 
        upper_vals = np.maximum(np.max(G, axis=1), 1./(1-self.params["gamma"]))
        upper_vals = np.repeat(np.expand_dims(upper_vals, axis=1), self.n_actions, axis=1)
        np.put(G, below_threshold_idxs, uppder_vals)

    def tsallis_policy_update(self):
        """ Policy update with PMD and Tsallis divergence (with p=1/2) """
        policy_threshold = (1-self.params["gamma"])**2 * self.n_actions**(-1)
        warnings.warn("Tsallis entropy not yet implemented, not policy update")
        pass

    def policy_performance(self) -> float: 
        """ Estimate value function """
        if not hasattr(self, "Q_est"):
            return np.inf
        return np.mean(self.get_value_function())

    def get_value_function(self):
        return np.sum(np.multiply(self.Q_est, self.policy), axis=1)

    def remap_obs(self, obs):
        if isinstance(obs, np.ndarray):
            return obs.flat[0]
        return obs

    def remap_action(self, action):
        if isinstance(action, np.ndarray):
            return action.flat[0]
        return action

class PMDGeneralStateFiniteAction(PMD):
    def __init__(self, env, params):
        super().__init__(env, params)
        self.check_PMDGeneralStateFiniteAction_params()
        self.action_space = env.action_space
        self.obs_space = env.observation_space
        (action_is_finite, action_dim, _) = get_space_property(self.action_space)
        (obs_is_finite, obs_dim, _) = get_space_property(self.obs_space)

        assert action_is_finite, "Action space not finite"

        self.n_actions = get_space_cardinality(self.action_space)
        if "dim" not in params:
            warnings.warn("Did not pass 'dim' into params, setting dim=100")
        dim = params.get("dim", 100)
        if isinstance(obs_dim, tuple):
            assert len(obs_dim) == 1, "Can only handle 1D observation space"
            self.obs_dim = obs_dim[0]
        if isinstance(action_dim, tuple):
            assert len(action_dim) == 1, "Can only handle 1D action space"
            action_dim = action_dim[0]

        assert action_dim == 1, f"action dim={action_dim}"

        # Env normalization
        self.obs_runstat = RunningStat(obs_dim)
        self.action_runstat = RunningStat(self.n_actions)
        self.rwd_runstat = RunningStat(1)
        self.updated_at_least_once = False

        # uniform policy and function approximation
        self.env_warmup()
        (q_est, sa_visited_tuple) = self.rollout.visited_stateaction_value()
        (X, _) = sa_visited_tuple
        self.fa = FunctionApproximator(self.n_actions, X, params)
        self.last_thetas = np.zeros((self.n_actions, self.fa.dim), dtype=float)
        self.theta_accum = np.copy(self.last_thetas)

    def check_PMDGeneralStateFiniteAction_params(self):
        # TODO: Remove this is doing NN or SGD
        return

    def policy_sample(self, s):
        """ Samples random action from current policy """
        if not self.updated_at_least_once:
            return self.rng.integers(self.n_actions)

        eps = self.params.get("eps_explore", 0)
        assert 0 <= eps <= 1

        log_policy_at_s = np.zeros(self.n_actions, dtype=float)
        for i in range(self.n_actions):
            self.fa.set_coef(self.theta_accum[i], i)
            log_policy_at_s[i] = self.fa.predict(np.atleast_2d(s), i)
        policy_at_s = np.exp(-(log_policy_at_s - np.min(log_policy_at_s)))
        policy_at_s = np.atleast_2d(policy_at_s)

        safe_normalize_row(policy_at_s)

        pi = (1.-eps)*np.squeeze(policy_at_s) + eps/self.n_actions
        return self.rng.choice(self.n_actions, p=pi)

    def policy_evaluate(self):
        """ Estimates Q function and stores in self.Q_est """
        if self.params.get("train_method", "mc") == "mc":
            self.monte_carlo_Q()
        elif self.params["train_method"] == "ctd":
            self.ctd_Q()
        elif self.params["train_method"] == "vrftd":
            self.vrftd_Q()

    def monte_carlo_Q(self):
        """ 
        Constructs Q/advantage function by Monte-Carlo. If missing any data
        points, we use RKHS to fill missing points.

        We update our running stats (e.g., mean and variance) for next
        iteration
        """
        self.obs_runstat.update()
        self.action_runstat.update()
        self.rwd_runstat.update()

        # check if we are missing any points 
        (q_est, sa_visited_tuple) = self.rollout.visited_stateaction_value()
        (s_visited, a_visited) = sa_visited_tuple
        X = s_visited
        y = q_est

        # extract learned features
        for i in range(self.n_actions):
            action_i_idx = np.where(a_visited==i)[0]
            if len(action_i_idx) == 0:
                print(f"Did not update action {i}")
                continue
            self.fa.update(X[action_i_idx], y[action_i_idx], i)
            self.last_thetas[i] = np.copy(self.fa.get_coef(i))

    def ctd_Q(self):
        raise NotImplemented

    def vrftd_Q(self):
        raise NotImplemented

    def policy_update(self): 
        if self.params.get("entropy", "kl").lower() == "kl":
            self.kl_policy_update()
        elif self.params["entropy"].lower() == "tsallis":
            self.tsallis_policy_update()
        else:
            self.kl_policy_update()

    def kl_policy_update(self):
        """ Policy update with PMD and KL divergence """
        self.updated_at_least_once = True
        self.theta_accum += self.get_stepsize_schedule()*self.last_thetas

    def tsallis_policy_update(self):
        """ Policy update with PMD and Tsallis divergence (with p=1/2) """
        raise NotImplemented

    def policy_performance(self) -> float: 
        """ Estimate value function """
        if not hasattr(self, "Q_est"):
            return 0
        return np.mean(self.get_value_function())

    def get_value_function(self):
        return np.sum(np.multiply(self.Q_est, self.policy), axis=1)

    def remap_obs(self, obs):
        if isinstance(obs, np.ndarray):
            return obs.flat[0]
        return obs

    def remap_action(self, action):
        if isinstance(action, np.ndarray):
            return action.flat[0]
        return action

    def env_warmup(self):
        """ 
        Runs the environment for a fixed number of iterations estimate
        empirical mean and variance of observations, actions, and rewards
        """
        old_rollout_len = self.rollout_len
        self.rollout_len = max(1000, self.n_actions*100)
        self.collect_rollouts()
        self.rollout_len = old_rollout_len

        self.obs_runstat.update()
        self.action_runstat.update()
        self.rwd_runstat.update()

        print("Finished normalization warmup")

    def normalize_obs(self, s):
        self.obs_runstat.push(s)
        if self.params.get("normalize_obs", False):
            return np.divide(s-self.obs_runstat.mean, self.obs_runstat.var)
        return s

    def normalize_action(self, a):
        self.obs_runstat.push(a)
        if self.params.get("normalize_action", False):
            return np.divide(a-self.action_runstat.mean, self.action_runstat.var)
        return a

    def normalize_rwd(self, r):
        if self.params.get("normalize_rwd", False):
            return np.divide(r-self.rwd_runstat.mean, self.rwd_runstat.var)
        return r

# class PMDGeneralStateAction(PMD):
