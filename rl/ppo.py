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

class PPO(RLAlg):
    def __init__(self, env, params):
        super().__init__(env, params)
        self.model = None

    def predict(self, obs, deterministic=False):
        return self.model.predict(obs, deterministic=deterministic)[0].flat[0]

    def _learn(self, max_iters):
        # Above is for a modified sb3
        self.env = Monitor(env=self.env, gamma=self.params["gamma"])
        # self.env = Monitor(env=self.env)
    
        max_episodes = self.params["max_episodes"] if self.params["max_episodes"] > 0 else np.inf
        callback_max_episodes = StopTrainingOnMaxEpisodes(
            max_episodes=max_episodes, 
            verbose=1
        )
        clip_range = self.params["ppo_clip_range"] if self.params["ppo_clip_range"] >= 0 else np.inf
        max_grad_norm = self.params["ppo_max_grad_norm"] if self.params["ppo_max_grad_norm"] >= 0 else np.inf
        self.model = sb3.PPO(
            policy=self.params['ppo_policy'],
            env=self.env, 
            verbose=1, 
            n_steps=self.params['ppo_rollout_len'],
            learning_rate=self.params['ppo_lr'],
            n_epochs=self.params['ppo_n_epochs'],
            batch_size=self.params['ppo_batch_size'],
            gamma=self.params['gamma'],
            gae_lambda=self.params['ppo_gae_lambda'],
            clip_range=clip_range,
            max_grad_norm=max_grad_norm,
            normalize_advantage=self.params["ppo_normalize_adv"],
            seed=self.params["seed"]
        )
        self.model.learn(max_iters, callback=callback_max_episodes)

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
