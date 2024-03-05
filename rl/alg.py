from abc import ABC
from abc import abstractmethod

import warnings

import numpy as np

class RLAlg(ABC):
    def __init__(self, env, params):
        self.params = params
        self.env = env
        self.params = params
        self.rng = np.random.default_rng(params.get("seed", None))
        self.warned_about_sample = False

    def learn(self, n_iter: int=100):
        self.t = 0
        try:
            self._learn(n_iter)
        except KeyboardInterrupt:
            print(f"Terminated early at iteration {self.t}")

    @abstractmethod
    def _learn(self, n_iter):
        raise NotImplemented

    def policy_sample(self, s):
        """ Returns an action uses current policy 

        :param s: state/observation we want an action for
        :return a: 
        """
        if not self.warned_about_sample:
            warnings.warn("Sampling randomly since `policy_sample` not yet implemented")
            self.warned_about_sample = True
        return self.env.action_space.sample()

