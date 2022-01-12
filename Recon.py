import gym
env = gym.make('MountainCarContinuous-v0')
env.reset()
for _ in range(10):
    env.render()
    print(env.step(env.action_space.sample())) # take a random action
env.close()