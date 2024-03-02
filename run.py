import gymnasium as gym
import gym_examples

from rl import PMDFiniteStateAction
from rl import PMDGeneralStateFiniteAction

def main():
    # env = gym.make(
    #     "gym_examples/GridWorld-v0", 
        # "gym_examples/SimpleWorld-v0", 
    #     size=4,
    #     action_eps=0.00,
    #     max_episode_steps=1000, # can change length here!
    # )
    # env = gym.wrappers.FlattenObservation(env)
    # env = gym.wrappers.TransformReward(env, lambda r : 1-r)

    env = gym.make(
        "LunarLander-v2", 
        # render_mode="human"
    )

    env = gym.wrappers.TransformReward(env, lambda r : -r)
    output = env.reset()

    # import ipdb; ipdb.set_trace()

    params = dict({
        "gamma": 0.999,
        "verbose": False,
        "rollout_len": 1000,
        "single_trajectory": True,
        "alpha": 0.1,
        "dim": 100,
        "normalize": True,
    })
    # alg = PMDFiniteStateAction(env, params)
    alg = PMDGeneralStateFiniteAction(env, params)

    alg.learn(n_iter=100)

if __name__ == "__main__":
    main()
