import multiprocessing as mp

import numpy as np
import wandb

from run import main as run_pmd_exp

def get_wandb_tuning_sweep_id():
    sweep_config = {
        "method": "grid",
        # "method": "random",
    }

    metric = {
        'name': 'median_reward',
        'goal': 'maximize'
    }
    sweep_config['metric'] = metric

    parameters_dict = {
        'base_stepsize': {
            'values': [0.01, 0.1, 1, 10]
            # 'distribution': 'uniform',
            # 'min': -2,
            # 'max': 1,
        },
        'stepsize': {
            'values': ['adaptive', 'pre-norm', 'standard']
        },
        'sgd_alpha': {
            'values': [0.0001, 0.001, 0.01]   
        },
    }
    sweep_config['parameters'] = parameters_dict

    sweep_id = wandb.sweep(sweep_config, project=f"rl-general-pmd-linearfunction-v4")

    return sweep_id

def wandb_tune_pmd_linear(config=None):
    # Initialize a new wandb run
    with wandb.init(config=config):
        config = wandb.config
        params = dict(config).copy()
        params["verbose"] = False
        params["stepsize"] = "decreasing"
        params["sgd_stepsize"] = "constant"
        params["gamma"] = 0.995
        params["use_advantage"] = True
        params["use_advantage"] = params['stepsize'] == 'pre-norm'
        params["normalize_rwd"] = params['stepsize'] == 'pre-norm'
        params["dynamic_stepsize"] = False
        params["mu_h"] = 0
        params["base_stepsize"] = params["base_stepsize"] * (0.1 if params['stepsize'] == 'standard' else 1.)
        params["sgd_alpha"] = params["sgd_alpha"]
        params["normalize_sa_val"] = params['stepsize'] == 'adaptive'
        params["max_grad_norm"] = -1
        params["max_ep_per_iter"] = -1
        params["max_iter"] = 20
        params["fa_type"] = "linear"
        params["rollout_len"] = 4096
        params["sgd_n_iter"] = 2048

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
    wandb.agent(sweep_id, wandb_tune_pmd_linear)
    # wandb.agent(sweep_id, wandb_tune_pmd_linear, count=n_runs)
