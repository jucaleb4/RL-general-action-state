# RL-general-action-state
Implementation of RL ([PMD and PDA](https://arxiv.org/abs/2211.16715)) over general action and state space.

## Requirements
TODO

## Creating scripts

- `04_18_2024/exp_0`: Lots of algorithms Runs GridWorld (PDA+nn; for sanity check), LunarLander (PDE+lin, PDE+lin+"SB3 policy", PDE+nn, PDE+nn+"SB3 policy"), LunarLander perturbed (PDA+lin, PDA+nn), LunarLander (PPO, QLearn), LunarLander perturbed (PPO, QLearn)
- `04_23_2024/exp_0`: PDA for GW and LL with different settings
- `04_23_2024/exp_1`: PPO and DQN for GW and LL 
- `04_23_2024/exp_2`: PDA+nn for InvertedPendulum 
- `04_26_2024/exp_0`: PDA+rkhs or nn for GW and LL with different settings
- `04_26_2024/exp_1`: PDA+rkhs or nn for InvertedPendulum with different settings
- `04_27_2024/exp_0`: PDA, PPO, DDPG for LQR
- `04_28_2024/exp_0`: PDA, PPO, DDPG for InvertedPendulum
- `04_28_2024/exp_1`: PDA, PPO, DDPG for LQR
- `11_01_2024/exp_0`: PDA with different esttings, PPO, and DQN for LunarLander

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
