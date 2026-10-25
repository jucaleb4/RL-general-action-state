import os
import sys

import itertools
import argparse
from collections import OrderedDict
import json

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "."+".", "."+"."))
sys.path.insert(0, parent_dir)
from rl import utils

DATE = "10_24_2026"
EXP_ID = 2
MAX_RUNS = 48

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

def setup_setting_files(seed, max_trials, max_steps):
    od = OrderedDict([
        ('alg', 'pda'),
        ('env_name', 'Ptychography-v0'),
        ('seed', seed),
        ('parallel', False),
        ('time_limit', 3_600),
        ('max_trials', max_trials),
        ('max_iters', max_steps),
        ('max_episodes', max_steps),
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
        # Computed Tomograph variables
        ("n_images", 200), # shapes: 64, skimage: 200
        ("n_angles", 24),
        ("image_size", 256), 
        ("action_size", 256), 
        ("scan_type", "ptycho"),
        ("image_type", utils.ImageType.MIXED.value),
        ("reward_type", utils.RewardType.FWD_RND_1.value),
        ("pnsr_reg", 0.0),
        ("angle_in_degrees", True),
        ("corruption_percentage", 0),
        # Pytchography variables
        ("probe_size", 32),
        ("corruption_photons", None),
        ("penalty", "amplitude"),
        ("solver", "L-BFGS-B"),
        ("solver_iters", 6),
        ("periodic", True),
        ("jitter", 1),
        ("allow_repeats", False),
        ("state_mode", "reim"),
        ("warm_start", True),
        ("blind", True),
    ])

    create_settings_and_logs_folders(od)
    log_folder_base = os.path.join("logs", DATE, "exp_%s" % EXP_ID)
    setting_folder_base = os.path.join("settings", DATE, "exp_%s" % EXP_ID)
    ct = 0

    # loop over problem instances
    blind_arr = [True, False]
    reward_type_arr = [
        utils.RewardType.FWD_RND_1,
        utils.RewardType.PNSR_INCREMENTAL,
    ]
    image_type_arr = [
        utils.ImageType.SKIMAGE,
    ]
    # penalty_arr = ["intensity", "amplitude", "poisson", "huber"]
    penalty_arr = ["amplitude", "poisson"]
    solver_iter_arr = [6,60]

    # PDA Lunar_lander with nn 
    fa_types = ['linear', 'nn', 'nn']
    pe_base_stepsizes = [0.01, 0.001, 0.001]
    alphas = [1e-4, 0, 0]
    policy_dvgs = ['kl', 'kl', 'tsallis']
    lr_arr = [1.0, 0.1, 0.1]

    exp_metadata = ["id", "env_name", "blind", "rwd_type", "img_type", "penalty", "solver_iter", "Env", "Alg", "fa_type", "Bregman"]
    row_format ="{:>5}|{:>20}|{:>5}|{:>20}|{:>15}|{:>10}|{:>10}" + "|{:>10}" * 2 + "|{:>10}"
    print("")
    print(row_format.format(*exp_metadata))
    print("-" * (5+20+60+10*3+len(exp_metadata)-1))

    od['pmd_stepsize_type'] = 'pda_1'
    od['alg'] = 'pmd'
    for blind, reward_type, image_type, penalty, solver_iter, (fa_type, policy_dvg, pe_base_stepsize, pe_alpha, lr) in itertools.product(
       blind_arr,  reward_type_arr, image_type_arr, penalty_arr, solver_iter_arr, zip(
            fa_types, 
            policy_dvgs, 
            pe_base_stepsizes, 
            alphas, 
            lr_arr,
    )):
        od["blind"] = blind
        od["reward_type"] = reward_type.value
        od["image_type"] = image_type.value
        od["penalty"] = penalty
        od["solver_iter"] = solver_iter

        od['pmd_fa_type'] = fa_type
        od['pmd_policy_divergence'] = policy_dvg
        od['pmd_pe_stepsize_base'] = pe_base_stepsize
        od['pmd_pe_alpha'] = pe_alpha
        od['pmd_stepsize_base'] = lr

        print(row_format.format(
                ct, od['env_name'], od['blind'], reward_type.name, 
                image_type.name, od['penalty'], od['solver_iter'], 
                od['alg'], od['pmd_fa_type'], od['pmd_policy_divergence'],
        ))

        setting_fname = os.path.join(setting_folder_base,  "run_%s.json" % ct)
        od['log_folder'] = os.path.join(log_folder_base, "run_%s" % ct)
        if not(os.path.exists(od["log_folder"])):
            os.makedirs(od["log_folder"])
        with open(setting_fname, 'w', encoding='utf-8') as f:
            json.dump(od, f, ensure_ascii=False, indent=4)
        ct += 1

    # exp_metadata = ["id", "Env", "Alg"]
    # row_format ="{:>5}|{:>25}|{:10}"
    # print("")
    # print(row_format.format(*exp_metadata))
    # print("-" * (5+25+10+len(exp_metadata)-1))
    # alg_arr = ['ppo', 'dqn']
    # lr_arr = [0.001, 0.0003]
    # # lr_arr = [3e-4, 1e-4]
    # for alg, lr in zip(alg_arr, lr_arr):
    #     od['alg'] = alg
    #     lr_key = "%s_lr" % alg
    #     od[lr_key] = lr
    # 
    #     print(row_format.format(ct, od['env_name'], od['alg']))
    # 
    #     setting_fname = os.path.join(setting_folder_base,  "run_%s.json" % ct)
    #     od['log_folder'] = os.path.join(log_folder_base, "run_%s" % ct)
    #     if not(os.path.exists(od["log_folder"])):
    #         os.makedirs(od["log_folder"])
    #     with open(setting_fname, 'w', encoding='utf-8') as f:
    #         json.dump(od, f, ensure_ascii=False, indent=4)
    #     ct += 1

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--setup", action="store_true", help="Setup environments. Otherwise we run the experiments")
    parser.add_argument(
        "--mode", 
        type=str, 
        default="validate", 
        choices=["validate", "work", "full"],
        help="Set up number of trials and max_step for various testing reasons"
    )
    parser.add_argument(
        "--sub_runs", 
        type=str, 
        help="Which experiments to run. Must be given as two integers separate by a comma with no space"
    )
    args = parser.parse_args()
    seed_0 = 0
    max_trials = 10

    if args.setup:
        # TODO: Do we need to change this?
        seed = 0
        max_steps = 100_000
        if args.mode == "validate":
            max_trials = 1
        if args.mode == "work":
            max_steps = 100_000
            max_trials = 1

        setup_setting_files(seed, max_trials, max_steps)
    else:
        start_run_id, end_run_id = parse_sub_runs(args.sub_runs)
        folder_name = os.path.join("settings", DATE, 'exp_%i' % EXP_ID)

        for i in range(start_run_id, end_run_id+1):
            settings_file = os.path.join(folder_name, "run_%s.json" % i)
            os.system("python run.py --settings %s" % settings_file)
