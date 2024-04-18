import os
import sys
import ast

# for perlmutter
if os.path.exists("/global/homes/c/cju33/.conda/envs/venv/lib/python3.12/site-packages"):
    sys.path.append("/global/homes/c/cju33/.conda/envs/venv/lib/python3.12/site-packages")
if os.path.exists("/global/homes/c/cju33/gym-examples"):
    sys.path.append("/global/homes/c/cju33/gym-examples")
import multiprocessing as mp

import argparse

import json

import gymnasium as gym
import gym_examples
# import or_gym

from rl import PMDFiniteStateAction
from rl import PMDGeneralStateFiniteAction
from rl import PDAGeneralStateAction

from rl import utils

def dictionary_clear_nones(dt):
    """ Returns a copied dictionary and removes keys whose value is None """
    return dict({k: v for k, v in dt.items() if v is not None})

def main(params, output={}):
    env_name = params['env_name']
    if 'VMPacking' in env_name:
        env = or_gym.make(env_name)
    elif 'LunarLander' in env_name:
        env = gym.make(
            env_name,
            # render_mode="human",
            max_episode_steps=1000, # can change length here!
            # gravity = -4.0 if params.get("lunar_perturbed", False) else -10.,
            # enable_wind= params.get("lunar_perturbed", False),
        )
    else:
        if "GridWorld" in env_name:
            full_env_name = os.path.join("gym_examples", env_name)
            env = gym.make(
                full_env_name,
                # render_mode="human",
                max_episode_steps=1000, # can change length here!
                size=2,
            )
        else:
            full_env_name = env_name
            env = gym.make(
                full_env_name,
                # render_mode="human",
                max_episode_steps=1000, # can change length here!
                # size=10,
            )

    # add penalty of 1
    # if params.get("lunar_perturbed", False):
    #     env = gym.wrappers.TransformReward(env, lambda r : r-1)

    # env = gym.wrappers.TimeAwareObservation(env)
    # env = gym.wrappers.NormalizeObservation(env)

    if isinstance(env.observation_space, gym.spaces.Dict):
        env = gym.wrappers.FlattenObservation(env)

    (obs_is_finite, obs_dim, _) = utils.get_space_property(env.observation_space)
    (act_is_finite, act_dim, _) = utils.get_space_property(env.action_space)
    is_enumerable = obs_is_finite and act_is_finite

    assert params['alg'] not in ["pmd", "pda"] or params["pmd_fa_type"] != "none" or is_enumerable, \
           "Must use function approximation if not enumerable"

    params['log_file'] = os.path.join(params['log_folder'], "seed=%s.csv" % params['seed'])

    alg = params['alg']
    if alg == "pmd" and params["pmd_fa_type"] == "none":
        alg = PMDFiniteStateAction(env, params)
    elif alg == "pmd":
        assert params["pmd_fa_type"] != "none" and act_is_finite, \
        "PMD cannot use neural network with general actions; run PDA instead"
        alg = PMDGeneralStateFiniteAction(env, params)
    elif alg == "pda":
        alg = PDAGeneralStateAction(env, params)
    elif alg == "qlearn":
        from rl import QLearn
        alg = QLearn(env, params)
    elif alg == "ppo":
        from rl.ppo import PPO
        alg = PPO(env, params)
    else:
        return 

    if alg in ["pmd", "pda"]:
        output[params['seed']] = alg.learn(params["max_iters"])
    else:
        output[params['seed']] = alg.learn(params["max_steps"])

def run_main_multiprocessing(alg, env_name, num_start, num_end, params):
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

        p = mp.Process(target=main, args=(alg, env_name, i, params))
        p.start()
        procs.append(p)

if __name__ == "__main__":
    # TODO: Print settings of the problem
    # TODO: Reformat code so we don't pass params in the final function invokation
    
    parser = argparse.ArgumentParser(
        prog='RL algs', 
        description='RL algorithms',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    """
    parser.add_argument('--alg', default="pmd", choices=["pmd", "pda", "qlearn", "ppo"], help="Algorithm")
    parser.add_argument('--env_name', choices=[
        "gym_examples/GridWorld-v0", 
        "LunarLander-v2", 
        "MountainCar-v0", 
        "Pendulum-v1",
        "Humanoid-v4",
        "HalfCheetah-v4",
        "VMPacking-v0",
        ],
        help="Environment"
    )
    parser.add_argument('--lunar_perturbed', type=ast.literal_eval, help="Perturb the lunar problem")
    parser.add_argument('--seed', type=int, help="Seed (or starting seed if parallel runs)")
    """
    parser.add_argument('--settings', type=str, required=True, help="Load settings")

    """
    parser.add_argument('--max_iter', type=int, default=100, help="Max number of training iterations")
    parser.add_argument('--max_ep', type=int, default=-1, help="Max number of training episodes")
    parser.add_argument('--fa_type', default="none", choices=["none", "linear", "nn"], help="Type of function approximation")

    parser.add_argument('--gamma', default=0.99, type=float, help="Discount factor")
    parser.add_argument('--gae_lambda', default=1., type=float, help="Additional discount factor")
    parser.add_argument('--use_gae', action="store_true", help="Use generalized advantage function")
    parser.add_argument('--rollout_len', default=1000, type=int, help="Trajectory length for one iteration/episode. Rollout_len overwritten by max_ep_per_iter")
    parser.add_argument('--max_ep_per_iter', default=-1, type=int, help="Max episodes per training epoch. If negative, then no max episode (uses rollout_len instead)")

    parser.add_argument('--stepsize', default="constant", choices=["decreasing", "constant"], help="Policy optimization stepsize")
    parser.add_argument('--base_stepsize', default=-1, type=float, help="base stepsize")
    parser.add_argument('--mu_h', default=0, type=float, help="entropy regularizations trength")
    parser.add_argument('--normalize_obs', action="store_true", help="Normalize observations via warm-start")
    parser.add_argument('--normalize_rwd', action="store_true", help="Dynamically scale rewards")
    parser.add_argument('--dynamic_stepsize', action="store_true", help="Dynamic scale stepsize by Q-estimate")
    parser.add_argument('--use_advantage', action="store_true", help="Use advantage function for policy update")
    parser.add_argument('--normalize_sa_val', action="store_true", help="Normalize Q or advantage function")
    parser.add_argument('--max_grad_norm', type=float, default=-1, help="Max l_inf norm of the gradient")

    parser.add_argument("--sgd_stepsize", default="constant", choices=["constant", "optimal"])
    parser.add_argument("--sgd_n_iter", type=int, default=1_000, help="number of SGD iterations (e.g. 10-50x the rollout_len)")
    parser.add_argument("--sgd_alpha", type=float, default=0.0001, help="Regularization strength")

    parser.add_argument("--ppo_clip_range", type=float, default=0.2, help="PPO clip range (negative value goes to inf)")

    parser.add_argument('--parallel', action="store_true", help="Use multiprocessing")
    parser.add_argument('--parallel_runs', type=int, default=10, help="Number of parallel runs")

    """
    args = parser.parse_args()

    if len(args.settings) > 5 and args.settings[-len('.json'):] == '.json':
        with open(args.settings, "r") as fp:
            params = json.load(fp)

        # settings, valid_hyperparams  = create_and_validate_settings(settings)
        # if not valid_hyperparams:
        #     exit(0)
    else:
        raise Exception("No valid json file %s passed in" % args.settings)

    main(params)

    # if settings.parallel:
    #     run_main_multiprocessing(args.alg, args.env_name, args.seed, args.seed+args.parallel_runs, settings)
    # else:
        # no seeding yet
    #     main(args.alg, args.env_name, args.seed, settings)
