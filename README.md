# RL-general-action-state
Implementation of RL ([PMD and PDA](https://arxiv.org/abs/2211.16715)) over general action and state space.

## Requirements
TODO

## Creating scripts
For these beginning scripts, we recommend using the mode `validate` during a full run.
- `04_18_2024/exp_0`: Lots of algorithms Runs GridWorld (PDA+nn; for sanity check), LunarLander (PDE+lin, PDE+lin+"SB3 policy", PDE+nn, PDE+nn+"SB3 policy"), LunarLander perturbed (PDA+lin, PDA+nn), LunarLander (PPO, QLearn), LunarLander perturbed (PPO, QLearn)
- `04_23_2024/exp_0`: PDA for GW and LL with different settings [note: use validate mode]
- `04_23_2024/exp_1`: PPO and DQN for GW and LL [note: use validate mode]
- `04_23_2024/exp_2`: PDA+nn for InvertedPendulum [deprecated - TBD: does not handle noise in PDA]

Starting after, here we recommend using the mode `full` for a full test run.
- `04_26_2024/exp_0`: Tuning PDA for GridWorld-v0 and LunarLander, and also just running PPO, DQN
- `04_26_2024/exp_1`: Tuning PDA for InvertedPendulum, and also just running PPO, DDPG 
- `04_27_2024/exp_0`: Tuning PDA for LQR, and also just running PPO, DDPG 
- `04_28_2024/exp_0`: Run PDA, PPO, DDPG for InvertedPendulum
- `04_28_2024/exp_1`: Run PDA, PPO, DDPG for LQR
- `11_01_2024/exp_0`: Tuning PDA for GridWorld-v1 (new version), and also just running PPO, DQN
- `02_10_2026/exp_0`: Tuning DQN, DDPG, PPO on LunarLander, InvertedPendulum, and LQR
- `02_10_2026/exp_1`: Tuning DQN, PPO on GridWorld-v1 and LunarLander
- `02_12_2026/exp_0`: Tuning PDA on Humanoid-v5
- `02_12_2026/exp_1`: Run PDA on Humanoid-v5
- `02_12_2026/exp_2`: Tuning PPO, DDPG on Humanoid-v5
- `02_12_2026/exp_3`: Run PPO, DDPG on Humanoid-v5

## Running scripts
TODO

## Code structure
TODO

## Plotting
TODO

## Citation
```
@misc{ju2024policyoptimizationgeneralstate,
      title={Policy Optimization over General State and Action Spaces},
      author={Caleb Ju and Guanghui Lan},
      year={2024},
      eprint={2211.16715},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2211.16715},
}
```
