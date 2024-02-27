import gymnasium as gym
import gym_examples

from rl import PMD

def main():
    env = gym.make(
        "gym_examples/GridWorld-v0", 
        size=10,
        action_eps=0.05
        # max_episode_steps=2000, # can change length here!
    )
    env = gym.wrappers.FlattenObservation(env)

    params = dict({})
    alg = PMD(env, params)

    alg.learn()

if __name__ == "__main__":
    main()
