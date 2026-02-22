import math
from pg_termination import pmd

def parse_sub_runs(sub_runs, total_runs):
    start_run_id, end_run_id = 0, total_runs
    if (sub_runs is not None):
        try:
            start_run_id, end_run_id = sub_runs.split(",")
            start_run_id = int(start_run_id)
            end_run_id = int(end_run_id)
            assert 0 <= start_run_id <= end_run_id <= total_runs, "sub_runs id must be in [0,%s]" % (total_runs-1)
            
        except:
            raise Exception("Invalid sub_runs id. Must be two integers split between [0,%s] split by a single comma with no space" % (total_runs-1))

    return start_run_id, end_run_id

def get_parameter_settings(seed_0, max_trials, max_iters, max_eps, max_steps, print_info, about):
    od = dict([
        ('alg', 'pmd'),
        ('env_name', 'LunarLander-v2'),
        ('lunar_perturbed', False),
        ('seed', seed_0),
        ('parallel', False),
        ('max_trials', max_trials),
        ('max_iters', max_iters),
        ('max_episodes', max_eps),
        ('max_steps', max_steps),
        ('gamma', 0.99),
        ('pmd_rollout_len', 1024),
        ('pmd_fa_type', "nn"),
        ('pmd_stepsize_type', 'pmd'),
        ('pmd_stepsize_base', 1),
        ('pmd_use_adv', True),
        ('pmd_normalize_sa_val', True),
        ('pmd_normalize_obs', False),
        ('pmd_normalize_rwd', False),
        ('pmd_mu_h', 0),
        ('pmd_pe_stepsize_type', 'constant'),
        ('pmd_pe_stepsize_base', 1e-3),
        ('pmd_pe_alpha', 1e-4),
        ('pmd_pe_max_epochs', 10),
        ('pmd_batch_size', 64),
        ('pmd_nn_update', 'adam'),
        ('pmd_nn_type', 'default'),
        ('pmd_max_grad_norm', -1),
        ('pmd_policy_divergence', 'kl'),
        ('pmd_sb3_policy', False),
        ('ppo_policy', "MlpPolicy"),
        ('ppo_lr', 0.0003),
        ('ppo_rollout_len', 2048),
        ('ppo_batch_size', 64),
        ('ppo_n_epochs', 10),
        ('ppo_gae_lambda', 0.95),
        ('ppo_clip_range', 0.2),
        ('ppo_max_grad_norm', -1),
        ('ppo_normalize_adv', False)
    ])

    od_info = [
        ('alg', 'Algorithm (pmd, ppo, qlearn, dppg)'),
        ('env_name', 'Environment name'),
        ('lunar_perturbed', '? (T/F)'),
        ('seed', 'starting seed'),
        ('parallel', '? (T/F)'),
        ('max_trials', '?'),
        ('max_iters', '?'),
        ('max_episodes', '?'),
        ('max_steps', '?'),
        ('gamma', 'discount factor'),
        ('pmd_rollout_len', 1024),
        ('pmd_fa_type', "PMD's function approximation (nn, linear)"),
        ('pmd_stepsize_type', 'stepsize type (pmd, pda_1, pda_2)'),
        ('pmd_stepsize_base', 'stepsize scaling (tuning param)'),
        ('pmd_use_adv', 'PMD learns advantage function (2X)'),
        ('pmd_normalize_sa_val', 'PMD normalize action-value to be [0,1] (2X)'),
        ('pmd_normalize_obs', 'PMD normalize state to be [0,1] (2X)'),
        ('pmd_normalize_rwd', 'PMD normalize reward to be [0,1] (2X)'),
        ('pmd_mu_h', 'regularization strength'),
        ('pmd_pe_stepsize_type', 'SGDRegressor learning_rate for PMD PE (see sklearn website)'),
        ('pmd_pe_stepsize_base', 'PMD PE stepsize scaling (tuning param)'),
        ('pmd_pe_alpha', 'PMD PE regularization (tuning param)'),
        ('pmd_pe_max_epochs', '? (default: 10)'),
        ('pmd_batch_size', '? (default: 64)'),
        ('pmd_nn_update', 'PMD nn update alg (adam, ?)'),
        ('pmd_nn_type', '? (default, ?)'),
        ('pmd_max_grad_norm', 'NN max grad norm (-1 is inf)'),
        ('pmd_policy_divergence', 'PMD Bregman divergence (kl, ?)'),
        ('pmd_sb3_policy', '? (T/F)'),
        ('ppo_policy', 'PPO Policy (MlpPolicy, ?)'),
        ('ppo_lr', 'PPO step size (default: 0.0003)'),
        ('ppo_rollout_len', 'PPO rolloutlen (defualt: 2048)'),
        ('ppo_batch_size', 'PPO batch size (default: 64)'),
        ('ppo_n_epochs', 'PPO num epochs (default: 10)'),
        ('ppo_gae_lambda', 'PPO GAE additional discount (default: 0.95)'),
        ('ppo_clip_range', 'PPO clip value range (default: 0.2 [2X])'),
        ('ppo_max_grad_norm', 'NN max grad norm (-1 is inf)'),
        ('ppo_normalize_adv', 'Normalize advantage in PPO'),
    ])

    if print_info:
        print("About:\n\t%s" % about)
        exp_metadata = ["setting", "description"]
        row_format ="{:<20}|{:<60}"
        print("")
        print(row_format.format(*exp_metadata))
        print("-" * (80+len(exp_metadata)-1))
        for name, description in od_info:
            print(row_format.format(name, description))
        print("-" * (80+len(exp_metadata)-1))

    return od
