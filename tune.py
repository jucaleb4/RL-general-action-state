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
        'mu_h': { 
            'distribution': 'uniform',
            'min': -4,
            'max': 0,
        },
        "use_reg": {
            'values': [True, False],
        },
        'gamma': {  # a flat distribution between 0 and 0.1
            'distribution': 'uniform',
            'min': 0.9,
            'max': 0.99,
        },
        'base_stepsize': {
            'distribution': 'uniform',
            'min': -2,
            'max': 1,
        },
        'rollout_len': { 
            'distribution': 'q_log_uniform_values',
            'q': 2,
            'min': 1024,
            'max': 4096,
        },
        'max_ep_per_iter': { 
            'values': [2,4,8],
        },
        'sgd_alpha': {
            'distribution': 'uniform',
            'min': -4,
            'max': 0,
        },
        'sgd_n_iter': {
            'distribution': 'q_log_uniform_values',
            'q': 10,
            'min': 5000,
            'max': 50000,
        },
    }
    sweep_config['parameters'] = parameters_dict

    sweep_id = wandb.sweep(sweep_config, project=f"rl-general-pmd-linearfunction-v2")

    return sweep_id

def wandb_tune_pmd_linear(config=None):
    # Initialize a new wandb run
    with wandb.init(config=config):
        config = wandb.config
        params = dict(config).copy()
        params["verbose"] = False
        params["stepsize"] = "decreasing"
        params["sgd_stepsize"] = "constant"
        params["use_advantage"] = True
        params["normalize_obs"] = False
        params["normalize_rwd"] = False
        params["dynamic_stepsize"] = False
        params["n_iter"] = 100
        params["n_ep"] = 100
        params["mu_h"] = 10**params["mu_h"] if params["use_reg"] else 0.
        params["base_stepsize"] = 10**params["base_stepsize"]
        params["sgd_alpha"] = 10**params["sgd_alpha"]
        params["normalize_sa_val"] = True
        params["max_grad_norm"] = -1

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
