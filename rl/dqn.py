import os
import warnings

import numpy as np

import stable_baselines3 as sb3
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import StopTrainingOnMaxEpisodes

import gymnasium as gym
import gym_examples

from rl import RLAlg

import numpy as np

class DQN(RLAlg):
    def __init__(self, env, params):
        super().__init__(env, params)

    def _learn(self, max_iters):
        # Above is for a modified sb3
        self.env = Monitor(env=self.env, gamma=self.params["gamma"])
        # self.env = Monitor(env=self.env)
    
        max_episodes = self.params["max_episodes"] if self.params["max_episodes"] > 0 else np.inf
        callback_max_episodes = StopTrainingOnMaxEpisodes(
            max_episodes=max_episodes, 
            verbose=1
        )
        model = sb3.DQN("MlpPolicy", self.env, verbose=1, seed=self.params['seed'])
        model.learn(max_iters, callback=callback_max_episodes)

        rwd_arr = self.env.get_episode_rewards()
        len_arr = self.env.get_episode_lengths()
        # time_arr = self.env.get_episode_times() 
        # print(f"Runtime: {np.sum(time_arr):.2f}s")

        log_file = os.path.join(self.params['log_folder'], "seed=%i.csv" % self.params['seed'])
        self.save_episode_rewards(log_file, rwd_arr, len_arr)

    def save_episode_rewards(self, log_file, rwd_arr, len_arr):
        fmt="%1.2f,%i"
        arr = np.vstack((np.atleast_2d(rwd_arr), np.atleast_2d(len_arr))).T
        with open(log_file, "wb") as fp:
            fp.write(b"episode rewards,episode len\n")
            np.savetxt(fp, arr, fmt=fmt)
        print(f"Saved episode data to {log_file}")
