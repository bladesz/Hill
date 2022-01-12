import os
import pickle
import neat
import gym 
import numpy as np
import random



final_generation = 0
for i in os.listdir():
    try:
        if i.startswith("best_genome") and int(i[11:]) > final_generation:
            final_generation = int(i[11:])
            print(final_generation)
    except:
        pass

for i in range(1,final_generation+1):
    # load the winner
    with open('best_genome' + str(i), 'rb') as f:
        c = pickle.load(f)

    print('Loaded genome:')
    print(c)

    # Load the config file, which is assumed to live in
    # the same directory as this script.
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config-feedforward')
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)

    net = neat.nn.FeedForwardNetwork.create(c, config)


    env = gym.make("MountainCarContinuous-v0")
    env = gym.wrappers.Monitor(env, 'best_genomex' + str(i), force=True)
    observation = env.reset()

    fitness = 0
    done = False
    while not done:
        action = [np.argmax(net.activate(observation))]

        observation, reward, done, info = env.step(action)
        env.render()
        fitness += reward
    print(fitness)