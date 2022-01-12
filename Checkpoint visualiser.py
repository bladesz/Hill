import os
import pickle
import neat
import gym 
import numpy as np   

max_state = 0
directory = os.listdir()
for i in os.listdir()[::-1]:
    if i.startswith("neat-checkpoint-") and int(i[16:]) > max_state:
        max_state = int(i[16:])
        print("x")
print(max_state)

for i in range(max_state):    

    # load the winner
    pop = neat.Population(config)
    c.restore_checkpoint('neat-checkpoint-')
    print('Loaded checkpoint:')

    # Load the config file, which is assumed to live in
    # the same directory as this script.
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config-feedforward')
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)

    net = neat.nn.FeedForwardNetwork.create(c, config)


    env = gym.make("MountainCarContinuous-v0")
    observation = env.reset()

    done = False
    while not done:
        action = [np.argmax(net.activate(observation))]

        observation, reward, done, info = env.step(action)
        env.render()