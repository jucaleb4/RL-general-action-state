import os

import gymnasium as gym
import gym_examples

from rl import PMDFiniteStateAction
from rl import PMDGeneralStateFiniteAction
from rl import QLearn

def main():
    # env = gym.make(
    #     "gym_examples/GridWorld-v0", 
        # "gym_examples/SimpleWorld-v0", 
    #     size=4,
    #     action_eps=0.00,
    #     max_episode_steps=1000, # can change length here!
    # )
    # env = gym.wrappers.FlattenObservation(env)
    # env = gym.wrappers.TransformReward(env, lambda r : 1-r)

    env = gym.make(
        "LunarLander-v2", 
        # "MountainCar-v0", 
        # render_mode="human",
        max_episode_steps=1000, # can change length here!
    )

    # env = gym.wrappers.TransformReward(env, lambda r : -r)
    output = env.reset()

    # import ipdb; ipdb.set_trace()

    for i in range(1,10):
        fname = os.path.join("logs", f"qlearn_ll_seed={i}.csv")

        params = dict({
            "gamma": 1.0,
            "verbose": False,
            "rollout_len": 1000,
            "single_trajectory": True,
            "alpha": 0.1,
            "dim": 100,
            "normalize": True,
            "fit_mode": 1,
            "fname": fname,
        })
        # alg = PMDFiniteStateAction(env, params)
        # alg = PMDGeneralStateFiniteAction(env, params)
        alg = QLearn(env, params)

        alg.learn(n_iter=1000)

if __name__ == "__main__":
    main()
