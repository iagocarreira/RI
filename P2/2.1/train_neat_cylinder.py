import neat
import numpy as np
import visualize
from robobo_env_neat_cylinder import RoboboEnvNeatCylinder

def eval_genome(genome, config):
    """Evalúa un genoma en el entorno Robobo y devuelve su fitness acumulado."""
    try:
        env = RoboboEnvNeatCylinder()
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        obs, _ = env.reset()
        total_fitness = 0.0
        done = False
        while not done:
            output = net.activate(obs)
            action = int(np.argmax(output))
            obs, reward, done, _, _ = env.step(action)
            total_fitness += reward
        env.close()
        return float(total_fitness)
    except Exception as e:
        print(f"⚠️ Error evaluando genoma: {e}")
        try: env.close()
        except: pass
        return -1000.0

def eval_population(genomes, config):
    for _, genome in genomes:
        fitness = eval_genome(genome, config)
        if fitness is None or np.isnan(fitness):
            fitness = -1000.0
        genome.fitness = float(fitness)

if __name__ == "__main__":
    config_path = "config_neat_cylinder.txt"
    config = neat.Config(
        neat.DefaultGenome, neat.DefaultReproduction,
        neat.DefaultSpeciesSet, neat.DefaultStagnation,
        config_path
    )

    pop = neat.Population(config)
    pop.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    pop.add_reporter(stats)

    winner = pop.run(eval_population,5)  # empieza con 10 generaciones
    print("\nMejor genoma encontrado:\n", winner)

    import pickle
    with open("winner_cylinder.pkl", "wb") as f:
        pickle.dump(winner, f)

    visualize.plot_stats(stats, filename="fitness_cylinder.svg")
    visualize.plot_species(stats, filename="species_cylinder.svg")
    visualize.draw_net(config, winner, filename="network_cylinder")
