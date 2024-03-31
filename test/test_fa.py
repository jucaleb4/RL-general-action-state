# https://stackoverflow.com/questions/16780014/import-file-from-parent-directory
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import time

import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt

import gymnasium as gym

from rl import PMDGeneralStateFiniteAction
from rl import utils

def test_learning_value_linear_function():
    # create enviroment
    env = gym.make("LunarLander-v2", max_episode_steps=1000)
    n_actions = utils.get_space_cardinality(env.action_space)
    sgd_n_iter = 1000
    params = {"sgd_n_iter": sgd_n_iter}

    alg = PMDGeneralStateFiniteAction(env, params)

    alg.collect_rollouts()

    (q_est, adv_est, s_visited, a_visited) = alg.rollout.get_est_stateaction_value()
    X = s_visited
    y = q_est

    custom_time = 0
    sklearn_time = 0
    train_losses = np.zeros((n_actions, sgd_n_iter), dtype=float)
    val_losses = np.zeros((n_actions, sgd_n_iter), dtype=float)
    num_iters = np.zeros(n_actions, dtype=int)
    sklearn_iters = np.zeros(n_actions, dtype=int)
    sklearn_losses = np.zeros((n_actions, 2), dtype=float)
    for i in range(n_actions):
        action_i_idx = np.where(a_visited==i)[0]
        if len(action_i_idx) == 0:
            continue

        X_i = X[action_i_idx]
        y_i = y[action_i_idx]
        sklearn_losses[i,0] = la.norm(alg.fa_Q.predict(X_i, i)-y_i)**2/len(y_i)

        s_time = time.time()
        train_loss, val_loss = alg.fa_Q.update(X_i, y_i, i, use_custom_sgd=True)
        custom_time += time.time() - s_time
        train_losses[i,:len(train_loss)] = train_loss
        val_losses[i,:len(val_loss)] = val_loss
        num_iters[i] = len(train_loss)

        s_time = time.time()
        sklearn_losses[i,1], _ = alg.fa_Q.update(X_i, y_i, i, use_custom_sgd=False)
        sklearn_iters[i] = alg.fa_Q.models[i].n_iter_
        sklearn_time += time.time() - s_time

    print(f"Custom total time: {custom_time:.2f}s. sklearn total time: {sklearn_time:.2f}s")

    # plot
    plt.style.use('ggplot')
    fig, axes = plt.subplots(nrows=4)
    fig.set_size_inches(5,15)
    for i in range(n_actions):
        n_i = num_iters[i]
        axes[i].plot(np.arange(n_i), train_losses[i,:n_i], label="Train loss")
        axes[i].plot(np.arange(n_i), val_losses[i,:n_i], label="Val loss")
        axes[i].plot([0,sklearn_iters[i]-1], sklearn_losses[i], label="Sklearn pre/post train loss")
        axes[i].set(yscale="log", title=f"Action {i}", ylabel="loss", xlabel="epoch")

    axes[0].legend()
    plt.tight_layout()
    plt.show()

def test_learning_value_nn_function():
    # create enviroment
    env = gym.make("LunarLander-v2", max_episode_steps=1000)
    n_actions = utils.get_space_cardinality(env.action_space)
    sgd_n_iter = 500
    params = {"sgd_n_iter": sgd_n_iter, "fa_type": "nn", "pe_update": "adam"}

    alg = PMDGeneralStateFiniteAction(env, params)

    alg.collect_rollouts()

    (q_est, adv_est, s_visited, a_visited) = alg.rollout.get_est_stateaction_value()
    X = s_visited
    y = q_est

    custom_time = 0
    sklearn_time = 0
    train_losses = np.zeros((n_actions, sgd_n_iter), dtype=float)
    val_losses = np.zeros((n_actions, sgd_n_iter), dtype=float)
    num_iters = np.zeros(n_actions, dtype=int)
    for i in range(n_actions):
        action_i_idx = np.where(a_visited==i)[0]
        if len(action_i_idx) == 0:
            continue

        X_i = X[action_i_idx]
        y_i = y[action_i_idx]

        s_time = time.time()
        train_loss, val_loss = alg.fa_Q.update(X_i, y_i, i)
        custom_time += time.time() - s_time
        train_losses[i,:len(train_loss)] = train_loss
        val_losses[i,:len(val_loss)] = val_loss
        num_iters[i] = len(train_loss)

    print(f"Custom total time: {custom_time:.2f}s")

    # plot
    plt.style.use('ggplot')
    fig, axes = plt.subplots(nrows=4)
    fig.set_size_inches(5,15)
    for i in range(n_actions):
        n_i = num_iters[i]
        axes[i].plot(np.arange(n_i), train_losses[i,:n_i], label="Train loss")
        axes[i].plot(np.arange(n_i), val_losses[i,:n_i], label="Val loss")
        axes[i].set(yscale="log", title=f"Action {i}")

    axes[0].legend()
    plt.tight_layout()
    plt.show()

 #test_learning_value_linear_function()
test_learning_value_nn_function()
