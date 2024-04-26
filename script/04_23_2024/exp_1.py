import os
import sys

import argparse
from collections import OrderedDict
import json

DATE = "04_23_2024"
EXP_ID = 1
MAX_RUNS = 6

def parse_sub_runs(sub_runs):
    start_run_id, end_run_id = 0, MAX_RUNS-1
    if (sub_runs is not None):
        try:
            start_run_id, end_run_id = sub_runs.split(",")
            start_run_id = int(start_run_id)
            end_run_id = int(end_run_id)
            assert 0 <= start_run_id <= end_run_id <= MAX_RUNS, "sub_runs id must be in [0,%s]" % MAX_RUNS
            
        except:
            raise Exception("Invalid sub_runs id. Must be two integers split between [0,%s] split by a single comma with no space" % MAX_RUNS)

    return start_run_id, end_run_id

def create_settings_and_logs_folders(od):
    """ 
    Creates folder name to store settings and logs (if they do not exist).
    Returns base folder_name to store settings.
    """
    folder_name = os.path.join("settings", DATE, "exp_%i" % EXP_ID)
    if not(os.path.exists(folder_name)):
        os.makedirs(folder_name)
    for i in range(0,MAX_RUNS):
        log_folder_base = os.path.join("logs", DATE, "exp_%i" % EXP_ID)
        od["log_folder"] = os.path.join(log_folder_base,  "run_%s" % i)
        if not(os.path.exists(log_folder_base)):
            os.makedirs(log_folder_base)

def setup_setting_files(seed, max_steps):
    od = OrderedDict([
        ('alg', 'pmd'),
        ('env_name', 'LunarLander-v2'),
        ('lunar_perturbed', False),
        ('seed', seed),
        ('parallel', False),
        ('max_trials', 1),
        ('max_iters', 100),
        ('max_episodes', 10_000),
        ('max_steps', max_steps),
        ('gamma', 0.99),
        ('pmd_rollout_len', 1024),
        ('pmd_fa_type', "nn"),
        ('pmd_stepsize_type', 'pmd'),
        ('pmd_stepsize_base', 1),
        ('pmd_use_adv', True),
        ('pmd_normalize_sa_val', False),
        ('pmd_normalize_obs', False),
        ('pmd_normalize_rwd', False),
        ('pmd_mu_h', 0),
        ('pmd_pe_stepsize_type', 'constant'),
        ('pmd_pe_stepsize_base', 1e-3),
        ('pmd_pe_alpha', 1e-4),
        ('pmd_pe_max_epochs', 100),
        ('pmd_batch_size', 64),
        ('pmd_nn_update', 'adam'),
        ('pmd_nn_type', 'default'),
        ('pmd_max_grad_norm', 1),
        ('pmd_policy_divergence', 'tsallis'),
        ('pmd_sb3_policy', False),
        ('ppo_policy', "MlpPolicy"),
        ('ppo_lr', 0.0003),
        ('ppo_rollout_len', 2048),
        ('ppo_batch_size', 64),
        ('ppo_n_epochs', 10),
        ('ppo_gae_lambda', 0.95),
        ('ppo_clip_range', 0.2),
        ('ppo_max_grad_norm', -1),
        ('ppo_normalize_adv', False),
    ])

    create_settings_and_logs_folders(od)
    log_folder_base = os.path.join("logs", DATE, "exp_%s" % EXP_ID)
    setting_folder_base = os.path.join("settings", DATE, "exp_%s" % EXP_ID)
    ct = 0

    # PDA Lunar_lander with nn 
    env_names = ['GridWorld-v0', 'LunarLander-v2']
    # algs = ['ppo', 'dqn', 'qlearn']
    algs = ['ppo', 'dqn'] # qlearn takes too long

    for env_name in env_names:
        od['env_name'] = env_name
        for alg in algs:
            od['alg'] = alg

            setting_fname = os.path.join(setting_folder_base,  "run_%s.json" % ct)
            od['log_folder'] = os.path.join(log_folder_base, "run_%s" % ct)
            if not(os.path.exists(od["log_folder"])):
                os.makedirs(od["log_folder"])
            with open(setting_fname, 'w', encoding='utf-8') as f:
                json.dump(od, f, ensure_ascii=False, indent=4)
            ct += 1

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--setup", action="store_true", help="Setup environments. Otherwise we run the experiments")
    parser.add_argument(
        "--mode", 
        type=str, 
        default="work", 
        choices=["validate", "work"],
        help="Set up number of trials and max_step for various testing reasons"
    )
    parser.add_argument(
        "--sub_runs", 
        type=str, 
        help="Which experiments to run. Must be given as two integers separate by a comma with no space"
    )
    args = parser.parse_args()
    seed_0 = 0

    if args.setup:
        # TODO: Do we need to change this?
        seed = 0
        max_steps = 200_000
        if args.mode == "work":
            max_steps = 10_000

        setup_setting_files(seed, max_steps)
    else:
        start_run_id, end_run_id = parse_sub_runs(args.sub_runs)
        folder_name = os.path.join("settings", DATE, 'exp_%i' % EXP_ID)

        for i in range(start_run_id, end_run_id+1):
            settings_file = os.path.join(folder_name, "run_%s.json" % i)
            os.system("python run.py --settings %s" % settings_file)
