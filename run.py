import os

import multiprocessing as mp

import gymnasium as gym
import gym_examples

from rl import PMDFiniteStateAction
from rl import PMDGeneralStateFiniteAction
from rl import QLearn

def main(seed):
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

    fname = os.path.join("logs", f"qlearn_ll_seed={seed}.csv")

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

def run_main_multiprocessing(num_start, num_end):
    num_exp = num_end - num_start
    assert num_exp >= 1
    num_proc = mp.cpu_count()
    num_threads = min(num_proc, num_exp)
    procs = []

    for i in range(num_start, num_end):
        if len(procs) == num_threads:
            for p in procs:
                p.join()
            procs = []

        p = mp.Process(target=main, args=(i,))
        p.start()
        procs.append(p)

if __name__ == "__main__":
    run_main_multiprocessing(1,10)
