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
        'base_stepsize': {
            'distribution': 'uniform',
            'min': -3,
            'max': 0,
        },
        'sgd_n_iter': {
            'values': [10,100],
        },
        'stepsize': {
            'values': ['decreasing', 'constant']
        },
        'sgd_base_stepsize': 
            'distribution': 'uniform'
            'min': -5,
            'max': -1,
        },
        'pe_update':
            'values': ['sgd', 'adam', 'sgd_mom'],
        },
        'max_grad_norm': 
            'values': [1,-1],
        },
        'sgd_alpha':
            'values': [0, 1e-3],
        },
        'network_type':
            'values': ['deep', 'shallow', 'small'],
        },
        'normalize_obs':
            'values': [True, False]
        }
        'gamma':{
            'distribution': 'uniform',
            'min': 0.9,
            'max': 1.0,
        },
    }
    sweep_config['parameters'] = parameters_dict

    sweep_id = wandb.sweep(sweep_config, project=f"rl-general-pmd-nn")

    return sweep_id

def wandb_tune_pmd_nn(config=None):
    # Initialize a new wandb run
    with wandb.init(config=config):
        config = wandb.config
        params = dict(config).copy()
        params["verbose"] = False
        params["use_advantage"] = True
        params["max_iter"] = 100
        params["max_step"] = 15000
        params["rollout_len"] = 1024
        params["base_stepsize"] = 10**params["base_stepsize"]
        params["sgd_base_stepsize"] = 10**params["sgd_base_stepsize"]
        params["sgd_warmstart"] = True
        params["fa_type"] == "nn"

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
    wandb.agent(sweep_id, wandb_tune_pmd_nn, count=n_runs)
