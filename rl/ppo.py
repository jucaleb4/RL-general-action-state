import os
import warnings

import numpy as np

import stable_baselines3 as sb3
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import StopTrainingOnMaxEpisodes

import gymnasium as gym
import gym_examples

from rl import RLAlg

class PPO(RLAlg):
    def __init__(self, env, params):
        super().__init__(env, params)

    def _learn(self, n_iter):
        self.env = Monitor(env=self.env)
    
        max_ep = self.params["max_ep"] if self.params["max_ep"] > 0 else np.inf
        callback_max_episodes = StopTrainingOnMaxEpisodes(
            max_episodes=max_ep, 
            verbose=1
        )
        model = sb3.PPO(
            "MlpPolicy", 
            self.env, 
            verbose=1, 
        )
        model.learn(int(1e10), callback=callback_max_episodes)

        rwd_arr = self.env.get_episode_rewards()
        len_arr = self.env.get_episode_lengths()
        time_arr = self.env.get_episode_times() 
        print(f"Runtime: {np.sum(time_arr):.2f}s")

        if len(self.params["fname"]) > 0:
            self.save_episode_rewards(rwd_arr, len_arr)

    def save_episode_rewards(self, rwd_arr, len_arr):
        if "fname" not in self.params:
            warnings.warn("No filename given, not saving")
            return
        fmt="%1.2f,%i"
        arr = np.vstack((np.atleast_2d(rwd_arr), np.atleast_2d(len_arr))).T
        with open(self.params["fname"], "wb") as fp:
            fp.write(b"episode rewards,episode len\n")
            np.savetxt(fp, arr, fmt=fmt)
        print(f"Saved episode data to {self.params['fname']}")
