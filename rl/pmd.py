""" Policy mirror descent """
from abc import ABC
from abc import abstractmethod

import warnings 

import numpy as np

import gymnasium as gym

import sklearn.linear_model as sklm

import rl.rollout as rollout

class PMD(ABC):
    def __init__(self, env, params):
        self.params = params

        self.env = env
        self.params = params
        self.check_params()
        self.rng = np.random.default_rng(params.get("seed", None))

        self.rollout = rollout.Rollout(env, params["gamma"])
        self.rollout_len = self.params["rollout_len"]

        self.initialized_env = False

    def learn(self, n_iter: int=100):
        """ Runs PMD algorithm for `n_iter`s """
        self.t = 0
        while self.t < n_iter:
            self.t += 1

            self.collect_rollouts()

            self.train()

            self.policy_update()

            mean_perf = self.policy_performance()
            if mean_perf == 0:
                import ipdb; ipdb.set_trace()
            print(f"[{self.t}] mean(V)={mean_perf}")

    def check_params(self):
        if "single_trajectory" not in self.params:
            warnings.warn("Did not pass in 'single_trajectory' into params, defaulting to False")
            self.params["single_trajectory"] = False
        if "rollout_len" not in self.params:
            warnings.warn("Did not pass in 'rollout_len' into params, defaulting to 1000")
            self.params["rollout_len"] = 1000

    def collect_rollouts(self):
        """ Collect samples for policy evaluation """
        self.rollout.clear_batch()
        if self.initialized_env or not self.params["single_trajectory"]:
            (s,_) = self.env.reset()
            self.initialized_env = True

        s = self.remap_obs(self.rollout.get_state(0))
        a = self.remap_action(self.policy_evaluate(s, eps=0.05))
        self.rollout.add_step_data(s, a)

        for t in range(self.rollout_len): 
            (s, r, term, trunc, _)  = self.env.step(a)
            s = self.remap_obs(s)
            a = self.remap_action(self.policy_evaluate(s, eps=0.05))
            self.rollout.add_step_data(s, a, r, term, trunc)

            if term or trunc:
                (s, _) = self.env.reset()
                s = self.remap_obs(s)
                a = self.remap_action(self.policy_evaluate(s))
                self.rollout.add_step_data(s, a)

    @abstractmethod
    def policy_evaluate(self, s):
        """ Returns an action uses current policy 

        :param s: state/observation we want an action for
        :return a: 
        """
        return self.env.action_space.sample()

    @abstractmethod
    def train(self):
        """ Uses samples to estimate policy value """
        raise NotImplemented

    @abstractmethod
    def policy_update(self): 
        """ Uses policy estimation from `train()` and updates new policy (can
            differ depending on policy approximation).
        """
        raise NotImplemented

    @abstractmethod
    def policy_performance(self) -> float: 
        """ Uses policy estimation from `train()` and updates new policy (can
            differ depending on policy approximation).
        """
        raise NotImplemented

    def get_stepsize_schedule(self):
        # TODO: Make this more robust
        # return self.params["gamma"]**(-self.t)
        return (self.t+1)**0.5

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
        assert isinstance(env.observation_space, gym.spaces.discrete.Discrete)
        assert isinstance(env.action_space, gym.spaces.discrete.Discrete)
        assert hasattr(env.observation_space, "start")
        assert hasattr(env.observation_space, "n")
        assert hasattr(env.action_space, "start")
        assert hasattr(env.action_space, "n")

        self.a_0 = env.action_space.start
        self.n_actions = env.action_space.n

        self.s_0 = env.observation_space.start
        self.n_states  = env.observation_space.n

        # uniform policy
        self.policy = np.ones((self.n_states, self.n_actions), dtype=float)
        self.policy /= self.n_actions

        self._setup_random_fourier_features(self.n_states*self.n_actions)

    def policy_evaluate(self, s, eps=0.):
        """ Samples random action from current policy 

        TODO: Find mapping from integers to original action (see #1)
        """
        assert 0 <= s < self.n_states, f"invalid state s={s} (only {self.n_states} states)"
        assert 0 <= eps <= 1

        pi = (1.-eps)*self.policy[s] + eps/self.n_actions
        return self.rng.choice(self.n_actions, p=pi)

    def train(self):
        """ Estimates Q function and stores in self.Q_est """
        self.Q_est = self.Q_via_monte_carlo()

    def _setup_random_fourier_features(self, n):
        """ 
        Creates random affine transformation via random Fourier features. 
        If no feature dimsion is passed in, use 100.
        Link: https://people.eecs.berkeley.edu/~brecht/papers/07.rah.rec.nips.pdf
        """
        if hasattr(self, "W") or hasattr(self, "b"):
            warnings.warn("Already set 'W' and 'b', now replacing 'W' and 'b'")

        # no approximation, have one feature go to one state-action pair
        dim = n
        self.W = 2**0.5 * self.rng.normal(size=(dim, n))
        self.b = 2*np.pi*self.rng.random(size=dim)

    def Q_via_monte_carlo(self):
        """ 
        Constructs Q/advantage function by Monte-Carlo. If missing any data
        points, we use RKHS to fill missing points.
        """
        # check if we are missing any points 
        Q_est, Ind_est = self.rollout.compute_enumerated_stateaction_value()

        assert Q_est.shape == Ind_est.shape
        assert len(Q_est.shape) == 2

        print(f"Visited {np.sum(Ind_est)}/{np.prod(np.shape(Ind_est))} sa pairs")
        if not np.all(Ind_est):
            ind_est = np.reshape(Ind_est, newshape=(-1,))
            # use visited state-action pairs to fit model
            # TODO: Why does `take` not work?
            # X = np.take(np.eye(len(ind_est)), ind_est, axis=0)
            X = np.eye(len(ind_est))[ind_est]
            partial_q_est = np.reshape(Q_est, newshape=(-1,))[ind_est]
            Q_est = self.ridge_regression(X, partial_q_est)

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
        return np.reshape(model.predict(Z_all), newshape=(self.n_states, self.n_actions))

    def get_rbf_features(self, X, sigma):
        """ Construct feature map from X (i-th rows is x_i) 
        - W is dxn matrix (d is feature dimension, n is input dim)
        - X is mxn matrix (m is number of input points)
        """
        B = np.repeat(self.b[:, np.newaxis], X.shape[0], axis=1)
        return np.sqrt(2./self.W.shape[0]) * np.cos(sigma * self.W @ X.T + B)

    def ctd(self):
        raise NotImplemented

    def ftd(self):
        raise NotImplemented

    def vrftd(self):
        raise NotImplemented

    def policy_update(self): 
        """ Policy update with PMD and KL divergence """
        # TODO: Check for NaNs

        eta = self.get_stepsize_schedule()
        G = eta * self.Q_est 

        # subtract so each row's largest element is 0
        G -= np.atleast_2d(np.min(G, axis=1)).T
        self.policy = np.multiply(self.policy, np.exp(-G))

        # normalize so each row is a simplex
        self.policy /= np.atleast_2d(np.sum(self.policy, axis=1)).T

        assert np.min(self.policy) >= 0, f"negative value in policy ({np.min(self.policy)})"
        assert np.allclose(np.sum(self.policy, axis=1), 1.), "not all rows sum to 1"

    def policy_performance(self) -> float: 
        """ Estimate value function """
        if not hasattr(self, "Q_est"):
            return 0
        return np.mean(self.get_value_function())

    def get_value_function(self):
        return np.sum(np.multiply(self.Q_est, self.policy), axis=1)

    def remap_obs(self, obs): 
        if isinstance(obs, np.ndarray):
            obs = obs[0]
        return obs - self.s_0

    def remap_action(self, a): 
        if isinstance(a, np.ndarray):
            a = a[0]
        return a - self.a_0
