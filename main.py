import gym
'''
env = gym.make('CartPole-v0')
values = env.reset()

print(values)

for _ in range(1000):
    env.render()
    observation, reward, done, info = env.step(env.action_space.sample()) # take a random action
env.close()
'''


import multiprocessing
import os
import pickle
import gym
import neat
import numpy as np
#import cart_pole
import visualize
from neat.six_util import itervalues, iterkeys
from neat.math_util import mean, stdev


runs_per_net = 5
simulation_seconds = 60.0

def post_evaluate(self, config, population, species, best_genome):
    # pylint: disable=no-self-use
    fitnesses = [c.fitness for c in itervalues(population)]
    fit_mean = mean(fitnesses)
    fit_std = stdev(fitnesses)
    best_species_id = species.get_species_id(best_genome.key)
    print('Population\'s average fitness: {0:3.5f} stdev: {1:3.5f}'.format(fit_mean, fit_std))
    print(
        'Best fitness: {0:3.5f} - size: {1!r} - species {2} - id {3}'.format(best_genome.fitness,
                                                                             best_genome.size(),
                                                                             best_species_id,
                                                                             best_genome.key))
    with open('best_genome' + str(self.generation), 'wb') as f:
        pickle.dump(best_genome, f)


neat.StdOutReporter.post_evaluate = post_evaluate
# Use the NN network phenotype and the discrete actuator force function.
def eval_genome(genome, config):
    net = neat.nn.FeedForwardNetwork.create(genome, config)

    fitnesses = []

    for runs in range(runs_per_net):
        sim = gym.make("MountainCarContinuous-v0")

        observation = sim.reset()

        # Run the given simulation for up to num_steps time steps.
        fitness = 0.0
        done = False
        while not done and fitness >= -50:
            action = [np.argmax(net.activate(observation))]

            # Apply action to the simulated cart-pole
            observation, reward, done, info = sim.step(action)
            fitness += reward

        if fitness <= -50:
            fitness = 0
        fitnesses.append(fitness)


    # The genome's fitness is its worst performance across all runs.
    return np.mean(fitnesses)


def eval_genomes(genomes, config):
    for genome_id, genome in genomes:
        genome.fitness = eval_genome(genome, config)


def run():
    # Load the config file, which is assumed to live in
    # the same directory as this script.
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config-feedforward')
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)

    pop = neat.Population(config)
    stats = neat.StatisticsReporter()
    std = neat.StdOutReporter(True)
    pop.add_reporter(stats)
    pop.add_reporter(std)
    pop.add_reporter(neat.Checkpointer(1))


    pe = neat.ParallelEvaluator(multiprocessing.cpu_count(), eval_genome)
    winner = pop.run(pe.evaluate)

    # Save the winner.
    with open('winner-feedforward', 'wb') as f:
        pickle.dump(winner, f)

    print(winner)

if __name__ == '__main__':
    run()