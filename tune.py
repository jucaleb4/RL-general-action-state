import multiprocessing as mp

import numpy as np
import wandb

from run import main as run_pmd_exp

def get_wandb_tuning_sweep_id():
    sweep_config = {
        "method": "random",
    }

    metric = {
        'name': 'median_reward',
        'goal': 'maximize'
    }
    sweep_config['metric'] = metric

    parameters_dict = {
        'c_h': { 
            'distribution': 'q_log_uniform_values',
            'q': 10,
            'min': 1e-10,
            'max': 10,
        },
        'use_advantage': {
            'values': [True, False],
        },
        'gamma': {  # a flat distribution between 0 and 0.1
            'distribution': 'uniform',
            'min': 0.9,
            'max': 0.995,
        },
        'stepsize': {
            'values': ['constant', 'decreasing'],
        },
        'base_stepsize': {
            'distribution': 'q_log_uniform_values',
            'q': 10,
            'min': 1e-4,
            'max': 1,
        },
        'rollout_len': { 
            'distribution': 'q_log_uniform_values',
            'q': 2,
            'min': 100,
            'max': 4096,
        },
        'normalize_obs': {
            'values': [True, False],
        },
        'normalize_rwd': {
            'values': [True, False],
        },
        'sgd_alpha': {
            'distribution': 'q_log_uniform_values',
            'q': 10,
            'min': 1e-6,
            'max': 1,
        },
        'sgd_stepsize': {
            'values': ['constant', 'optimal'],
        },
        'sgd_n_iter': {
            'distribution': 'q_log_uniform_values',
            'q': 10,
            'min': 5000,
            'max': 50000,
        },
    }
    sweep_config['parameters'] = parameters_dict

    sweep_id = wandb.sweep(sweep_config, project=f"rl-general-pmd-linearfunction")

    return sweep_id

def hi():
    params = dict({
        "verbose": False,
        "c_h": 0,
        "use_advantage": True,
        "gamma": 1.,
        "stepsize": "constant",
        "base_stepsize": 1,
        "rollout_len": 1024, 
        "normalize_obs": True,
        "normalize_rwd": True,
        "sgd_alpha": 0.0001,
        "sgd_stepsize": "constant", 
        "sgd_n_iter": 10000,
    })
    for _ in range(1):
        params["verbose"] = False
        params["n_iter"] = 2 # 200

        n_trials = 1 # 10
        n_proc = mp.cpu_count()
        n_threads = min(n_proc, n_trials)
        final_rewards_arr = np.zeros(n_trials, dtype=float)

        manager = mp.Manager()
        returned_dict = manager.dict()
        procs = []
        for i in range(n_trials):
            if len(procs) == n_threads:
                for p in procs:
                    p.join()
                procs = []

            p = mp.Process(target=run_pmd_exp, args=(
                "pmd", 
                "LunarLander-v2", 
                i, 
                params, 
                returned_dict)
            )
            p.start()
            procs.append(p)

        if len(procs) > 0:
            for p in procs:
                p.join()
            procs = []

        final_rwds = []
        for i in range(n_trials):
            final_rwds.append(returned_dict[i])
            
        wandb.log({
            "median_reward": np.median(final_rwds), 
            "all_rewards": final_rwds,
        })    

def wandb_tune_pmd_linear(config=None):
    # Initialize a new wandb run
    with wandb.init(config=config):
        config = wandb.config
        params = dict(config)
        params["verbose"] = False
        params["n_iter"] = 20

        n_trials = 10
        n_proc = mp.cpu_count()
        n_threads = min(n_proc, n_trials)
        final_rewards_arr = np.zeros(n_trials, dtype=float)

        manager = mp.Manager()
        returned_dict = manager.dict()
        procs = []
        for i in range(n_trials):
            if len(procs) == n_threads:
                for p in procs:
                    p.join()
                procs = []

            p = mp.Process(target=run_pmd_exp, args=(
                "pmd", 
                "LunarLander-v2", 
                i, 
                params, 
                returned_dict)
            )
            p.start()
            procs.append(p)

        if len(procs) > 0:
            for p in procs:
                p.join()

        final_rwds = []
        for i in range(n_trials):
            final_rwds.append(returned_dict[i])
            
        wandb.log({
            "median_reward": np.median(final_rwds), 
            "all_rewards": final_rwds,
        })    

if __name__ == "__main__":
    n_runs = 64
    sweep_id = get_wandb_tuning_sweep_id()
    wandb.agent(sweep_id, wandb_tune_pmd_linear, count=n_runs)
