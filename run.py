import gymnasium as gym
import gym_examples

from rl import PMDFiniteStateAction

def main():
    env = gym.make(
        # "gym_examples/GridWorld-v0", 
        "gym_examples/SimpleWorld-v0", 
        size=2,
        action_eps=0.05
        # max_episode_steps=2000, # can change length here!
    )
    # env = gym.wrappers.FlattenObservation(env)
    output = env.reset()

    params = dict({"gamma": 0.99})
    alg = PMDFiniteStateAction(env, params)

    alg.learn()

if __name__ == "__main__":
    main()
