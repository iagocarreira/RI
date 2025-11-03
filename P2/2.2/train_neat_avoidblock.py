import neat
import numpy as np
import visualize
from robobo_env_neat_avoidblock import RoboboEnvNeatAvoidBlock

def eval_genome(genome, config):
    try:
        env = RoboboEnvNeatAvoidBlock()
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        obs, _ = env.reset()
        total_fitness = 0.0
        done = False
        while not done:
            output = net.activate(obs)
            action = np.argmax(output)
            obs, reward, done, _, _ = env.step(action)
            total_fitness += reward
        env.close()
        return total_fitness
    except Exception as e:
        print(f"⚠️ Error evaluando genoma: {e}")
        try: env.close()
        except: pass
        return -1000.0

def eval_population(genomes, config):
    for genome_id, genome in genomes:
        try:
            fitness = eval_genome(genome, config)
            # Si es NaN o inf, sustituye por un valor seguro
            if not np.isfinite(fitness):
                fitness = -10.0
            genome.fitness = float(fitness)
        except Exception as e:
            print(f"⚠️ Error evaluando genoma {genome_id}: {e}")
            genome.fitness = -10.0  # penalización

if __name__ == "__main__":
    config_path = "config_neat_avoidblock.txt"
    config = neat.Config(
        neat.DefaultGenome, neat.DefaultReproduction,
        neat.DefaultSpeciesSet, neat.DefaultStagnation,
        config_path
    )

    pop = neat.Population(config)
    pop.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    pop.add_reporter(stats)

    winner = pop.run(eval_population, 5)

    print("\nMejor genoma encontrado:\n", winner)
    import pickle
    with open("winner_avoidblock.pkl", "wb") as f:
        pickle.dump(winner, f)

    visualize.plot_stats(stats, filename="fitness_avoidblock.svg")
    visualize.plot_species(stats, filename="species_avoidblock.svg")
    visualize.draw_net(config, winner, filename="network_avoidblock")
