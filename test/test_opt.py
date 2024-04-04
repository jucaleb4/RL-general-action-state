# https://stackoverflow.com/questions/16780014/import-file-from-parent-directory
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import time

import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt

import gymnasium as gym

from rl import PDAGeneralStateAction
from rl import utils

from rl.gopt import BlackBox
from rl.gopt import ACFastGradDescent

def test_nn_subproblem():
    # create enviroment
    env = gym.make("HalfCheetah-v4", max_episode_steps=1000)
    params = {}

    alg = PDAGeneralStateAction(env, params)
    max_subprobs_solved = 5
    subprob_grad_arr = []

    for t in range(max_subprobs_solved):
        alg.collect_rollouts()
        alg.policy_evaluate()
        alg.l2_policy_update()

        # below is just policy sample
        s_ = env.observation_space.sample()
        warm_start = True
        raised_mu_Q = False
        alg.mu_Q = 0.
        alg.mu_d = 0.
        warm_start = False

        # TODO: How to automate this?
        a_0 = np.copy(alg.pi_0)
        a_prev = np.copy(a_0)

        while 1:
            (_, lam_t) = alg.get_stepsize_schedule()

            scale = lam_t/(alg.beta_sum)

            # TODO: Add regularization
            f = lambda a : -alg.fa_Q_accum.predict(np.atleast_2d(np.append(s_, a)))  \
                        + scale*0.5*la.norm(a-a_0)**2
            df = lambda a : -alg.fa_Q_accum.grad(np.atleast_2d(np.append(s_, a)))[len(s_):]  \
                        + scale*(a-a_0)
            oracle = BlackBox(f, df)

            # stop_nonconvex = abs(self.mu_d) <= ub_est_Q_grad_norm
            stop_nonconvex = True
            opt = ACFastGradDescent(
                oracle, 
                np.copy(a_0), 
                alpha=0.0, 
                tol=1e-4/alg.beta_sum, 
                stop_nonconvex=stop_nonconvex,
            )

            max_iter = 1_000 if warm_start else 50_000

            a_prev, _, grad_hist = opt.solve(n_iter=max_iter)

            if len(grad_hist) == max_iter:
                warnings.warn("Max iterations reached without converging. Consider increasing iterations")

            if stop_nonconvex and opt.detected_nonconvex:
                old_mu_d = alg.mu_d
                alg.mu_Q = 2*alg.mu_Q if alg.mu_Q > 0 else 1
                raised_mu_Q = True
                alg.mu_d = alg.params["mu_h"] - alg.mu_Q
            else:
                print(f"ended with a={a_prev}")
                subprob_grad_arr.append(grad_hist)
                if raised_mu_Q:
                    pass
                break

    # plot
    plt.style.use('ggplot')
    fig, ax= plt.subplots()
    fig.set_size_inches(5,5)
    for i in range(max_subprobs_solved):
        grad_hist = subprob_grad_arr[i]
        ax.plot(np.arange(len(grad_hist)), grad_hist, label=f"Iter k={i}")
    title= f"Gradient norm (after {max_subprobs_solved} iters and mu_Q={alg.mu_Q:.2e})"
    ax.set(yscale="log", title=title, ylabel=r"$\|\nabla f(x)\|_2$", xlabel="iter")
    ax.legend()
    plt.tight_layout()
    plt.show()

test_nn_subproblem()
