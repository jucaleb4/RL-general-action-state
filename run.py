import os

# just for perlmutter
if False:
    import sys
    sys.path.append("/global/homes/c/cju33/.conda/envs/venv/lib/python3.12/site-packages")
    sys.path.append("/global/homes/c/cju33/gym-examples")
import multiprocessing as mp

import argparse

import gymnasium as gym
import gym_examples

from rl import PMDFiniteStateAction
from rl import PMDGeneralStateFiniteAction
from rl import QLearn
from rl import PPO

def main(alg, env_name, seed, settings):
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
        env_name,
        # render_mode="human",
        max_episode_steps=1000, # can change length here!
    )

    # env = gym.wrappers.TransformReward(env, lambda r : -r)
    output = env.reset()

    # import ipdb; ipdb.set_trace()

    fname = os.path.join("logs", f"{alg}_{env_name}_seed={seed}.csv")

    params = dict({
        "gamma": settings["gamma"],
        "verbose": False,
        "rollout_len": settings["rollout_len"],
        "single_trajectory": True,
        "dim": 100,
        "normalize": True,
        "fit_mode": 1,
        "cutoff": 1,
        "fname": fname,
        "stepsize": settings["stepsize"],
    })
    # alg = PMDFiniteStateAction(env, params)
    if alg == "pmd":
        alg = PMDGeneralStateFiniteAction(env, params)
    elif alg == "qlearn":
        alg = QLearn(env, params)
    elif alg == "ppo":
        alg = PPO(env, params)
    else:
        return 

    alg.learn(n_iter=settings["n_iter"])

def run_main_multiprocessing(alg, env_name, num_start, num_end):
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

        p = mp.Process(target=main, args=(alg, env_name, i,))
        p.start()
        procs.append(p)

if __name__ == "__main__":
    # TODO: Print settings of the problem
    # TODO: Reformat code so we don't pass params in the final function invokation
    parser = argparse.ArgumentParser(prog='RL algs', description='RL algorithms')
    parser.add_argument('--alg', default="pmd", choices=["qlearn", "pmd", "ppo"], help="Algorithm")
    parser.add_argument('--env_name', default="LunarLander-v2", choices=["LunarLander-v2", "MountainCar-v0"], help="Environment")
    parser.add_argument('--seed', type=int, default=0, help="Seed (or starting seed if parallel runs)")

    parser.add_argument('--n_iter', type=int, default=100, help="Number of training iterations/episodes")
    parser.add_argument('--gamma', default=0.99, type=float, help="Discount factor")
    parser.add_argument('--rollout_len', default=1000, type=int, help="Trajectory length for one iteration/episode")
    parser.add_argument('--stepsize', default="decreasing", choices=["decreasing", "constant"], help="Policy optimization stepsize")

    parser.add_argument('--parallel', action="store_true", help="Use multiprocessing")
    parser.add_argument('--parallel_runs', type=int, default=10, help="Number of parallel runs")

    args = parser.parse_args()

    if args.parallel:
        run_main_multiprocessing(args.alg, args.env_name, args.seed, args.seed+args.parallel_runs)
    else:
        # no seeding yet
        main(args.alg, args.env_name, args.seed, vars(args))
