""" Policy mirror descent """
from abc import ABC
from abc import abstractmethod

import warnings 

import numpy as np
import numpy.linalg as la

import gymnasium as gym

import sklearn.linear_model as sklm

from rl import Rollout
from rl.utils import vec_to_int
from rl.utils import safe_normalize_row
from rl.utils import get_space_property
from rl.utils import get_space_cardinality
from rl.utils import pretty_print_gridworld

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
        """ Runs PMD algorithm for `n_iter`s """
        self.t = 0
        while self.t < n_iter:
            self.t += 1

            self.collect_rollouts()

            self.policy_evaluate()

            self.policy_update()

            mean_perf = self.policy_performance()
            print(f"[{self.t}] episode mean len={np.mean(self.rollout.get_episode_lens()):.1f}")
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

        a = self.policy_sample(s)
        self.rollout.add_step_data(s, a)

        for t in range(self.rollout_len): 
            (s, r, term, trunc, _)  = self.env.step(a)
            a = self.policy_sample(s)
            self.rollout.add_step_data(s, a, r, term, trunc)

            if term or trunc:
                (s, _) = self.env.reset()
                a = self.policy_sample(s)
                self.rollout.add_step_data(s, a)

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
        return (1./self.t)**0.5 
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
        self.params["dim"] = self.n_states*self.n_actions

        self._setup_random_fourier_features(self.n_states*self.n_actions)

    def policy_sample(self, s):
        """ Samples random action from current policy 

        TODO: Find mapping from integers to original action (see #1)
        """
        s_ = vec_to_int(s, self.obs_space)
        assert 0 <= s_ < self.n_states, f"invalid s={s} ({self.n_states} states)"
        eps = self.params.get("eps_explore", 0)
        assert 0 <= eps <= 1

        pi = (1.-eps)*self.policy[s_] + eps/self.n_actions
        return self.rng.choice(self.n_actions, p=pi)

    def policy_evaluate(self):
        """ Estimates Q function and stores in self.Q_est """
        if self.params.get("train_method", "mc") == "mc":
            self.Q_est = self.monte_carlo_Q()
        elif self.params["train_method"] == "ctd":
            self.Q_est = self.ctd_Q()
        elif self.params["train_method"] == "vrftd":
            self.Q_est = self.vrftd_Q()

    def _setup_random_fourier_features(self, n):
        """ 
        Creates random affine transformation via random Fourier features. 
        If no feature dimsion is passed in, use 100.
        Link: https://people.eecs.berkeley.edu/~brecht/papers/07.rah.rec.nips.pdf
        """
        assert "dim" in self.params, "missing key 'dim' in self.params"
        if hasattr(self, "W") or hasattr(self, "b"):
            warnings.warn("Already set 'W' and 'b', now replacing 'W' and 'b'")

        dim = self.params["dim"]
        self.W = 2**0.5 * self.rng.normal(size=(dim, n))
        self.b = 2*np.pi*self.rng.random(size=dim)

    def monte_carlo_Q(self):
        """ 
        Constructs Q/advantage function by Monte-Carlo. If missing any data
        points, we use RKHS to fill missing points.
        """
        # check if we are missing any points 
        Q_est, Ind_est = self.rollout.compute_enumerated_stateaction_value()
        self.Ind_est = Ind_est

        assert Q_est.shape == Ind_est.shape
        assert len(Q_est.shape) == 2

        if self.params.get("verbose", 0) >= 1:
            print(f"Visited {np.sum(Ind_est)}/{np.prod(np.shape(Ind_est))} sa pairs")

        # if not np.all(Ind_est):
        # TODO: Remove since this adds a lot of error
        if False:
            ind_est = np.reshape(Ind_est, newshape=(-1,))
            # use visited state-action pairs to fit model
            # TODO: Why does `take` not work?
            # X = np.take(np.eye(len(ind_est)), ind_est, axis=0)
            X = np.eye(len(ind_est))[ind_est]
            partial_q_est = np.reshape(Q_est, newshape=(-1,))[ind_est]
            fitted_Q_est = self.ridge_regression(X, partial_q_est)

            print(f"Q error with RKHS: {la.norm(np.multiply(Ind_est, fitted_Q_est-Q_est))}")

            Q_est = fitted_Q_est

        return Q_est

    def ridge_regression(self, X, y):
        """ Fit and predicts with ridge regression """
        N = self.n_actions*self.n_states
        alpha = self.params.get("alpha", 0.1)
        sigma = self.params.get("sigma", 1)

        model = sklm.Ridge(alpha=alpha)
        Z = self.get_rbf_features(X, sigma)
        model.fit(Z.T, y)
        Z_all = self.get_rbf_features(np.eye(N), sigma)

        return np.reshape(
            model.predict(Z_all), 
            newshape=(self.n_states, self.n_actions)
        )

    def get_rbf_features(self, X, sigma):
        """ Construct feature map from X (i-th rows is x_i) 
        - W is dxn matrix (d is feature dimension, n is input dim)
        - X is mxn matrix (m is number of input points)
        """
        B = np.repeat(self.b[:, np.newaxis], X.shape[0], axis=1)
        return np.sqrt(2./self.W.shape[0]) * np.cos(sigma * self.W @ X.T + B)

    def ctd_Q(self):
        raise NotImplemented

    def vrftd_Q(self):
        raise NotImplemented

    def policy_update(self): 
        """ Policy update with PMD and KL divergence """
        eta = self.get_stepsize_schedule()
        G = np.copy(self.Q_est)
        G -= np.atleast_2d(np.min(G, axis=1)).T
        self.policy = np.multiply(self.policy, np.exp(-eta*G))

        safe_normalize_row(self.policy)

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

# class PMDGeneralStateFiniteAction(PMD):

# class PMDGeneralStateAction(PMD):
