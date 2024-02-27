""" Policy mirror descent """
from abc import ABC
from abc import abstractmethod

import warnings 

import rl.rollout as rollout

class PMD(ABC):
    def __init__(self, env, params):
        self.params = params

        self.env = env
        self.params = params
        self.check_params()

        self.rollout = rollout.Rollout(env.observation_space, env.action_space)
        self.rollout_len = self.params["rollout_len"]
        self._initialize_env()

    def learn(self, n_iter: int=20):
        """ Runs PMD algorithm for multiple iterations

        :param n_iter: number of iterations to learn
        """
        for k in range(n_iter):
            self.collect_rollouts()

            self.train()

            self.policy_update()

            # TODO: Logging

    def check_params(self):
        if "single_trajectory" not in self.params:
            warnings.warn("Did not pass in 'single_trajectory' into params, defaulting to False")
            self.params["single_trajectory"] = False
        if "rollout_len" not in self.params:
            warnings.warn("Did not pass in 'rollout_len' into params, defaulting to 1000")
            self.params["rollout_len"] = 1000

    def _initialize_env(self):
        (s, _) = self.env.reset()
        self.rollout.add_reset_data(s)

    def collect_rollouts(self):
        """ Collect samples for policy evaluation """
        self.rollout.clear_batch()
        if not self.params["single_trajectory"]:
            (s,_) = self.env.reset()
            self.rollout.add_reset_data(s)
        s = self.rollout.get_state(0)

        for t in range(self.rollout_len):
            a = self.policy_evaluate(s)

            (s, r, term, trunc, _)  = self.env.step(a)
            if term or trunc:
                (s, _) = self.env.reset()

            self.rollout.add_step_data(s, a, r, term, trunc)

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
