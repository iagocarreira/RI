import os
import neat
import graphviz
import matplotlib.pyplot as plt
import numpy as np

def draw_net(config, genome, view=False, filename=None, node_names=None, show_disabled=True, prune_unused=False, node_colors=None, fmt='png'):
    """Draws a neural network with graphviz."""
    if graphviz is None:
        raise ImportError("This function requires the graphviz package.")

    dot = graphviz.Digraph(format=fmt, node_attr={'shape': 'circle', 'fontsize': '9', 'height': '0.2', 'width': '0.2'})

    inputs = set(config.genome_config.input_keys)
    outputs = set(config.genome_config.output_keys)
    used_nodes = set(genome.nodes.keys())

    if prune_unused:
        used_nodes = set()
        for cg in genome.connections.values():
            if cg.enabled or show_disabled:
                used_nodes.add(cg.key[0])
                used_nodes.add(cg.key[1])

    for n in inputs:
        name = node_names.get(n, str(n)) if node_names else str(n)
        dot.node(name, _attributes={'style': 'filled', 'fillcolor': 'lightgray'})

    for n in outputs:
        name = node_names.get(n, str(n)) if node_names else str(n)
        dot.node(name, _attributes={'style': 'filled', 'fillcolor': 'lightblue'})

    for n in used_nodes:
        if n in inputs or n in outputs:
            continue
        name = node_names.get(n, str(n)) if node_names else str(n)
        dot.node(name, _attributes={'style': 'filled', 'fillcolor': 'white'})

    for cg in genome.connections.values():
        if cg.enabled or show_disabled:
            input, output = cg.key
            a = node_names.get(input, str(input)) if node_names else str(input)
            b = node_names.get(output, str(output)) if node_names else str(output)
            style = 'solid' if cg.enabled else 'dotted'
            color = 'green' if cg.weight > 0 else 'red'
            width = str(0.1 + abs(cg.weight / 5.0))
            dot.edge(a, b, _attributes={'style': style, 'color': color, 'penwidth': width})

    if filename is not None:
        dot.render(filename, view=view)
    elif view:
        dot.view()

    return dot


def plot_stats(statistics, ylog=False, view=False, filename='avg_fitness.svg'):
    generation = range(len(statistics.most_fit_genomes))
    best_fitness = [c.fitness for c in statistics.most_fit_genomes]
    avg_fitness = statistics.get_fitness_mean()
    stdev_fitness = statistics.get_fitness_stdev()

    plt.figure()
    plt.plot(generation, best_fitness, "b-", label="Best")
    plt.plot(generation, avg_fitness, "g-", label="Average")
    plt.fill_between(generation,
                     np.subtract(avg_fitness, stdev_fitness),
                     np.add(avg_fitness, stdev_fitness),
                     color="g", alpha=0.2)
    plt.title("Population's fitness evolution")
    plt.xlabel("Generations")
    plt.ylabel("Fitness")
    if ylog:
        plt.yscale('log')
    plt.grid()
    plt.legend()
    plt.savefig(filename)
    if view:
        plt.show()
    plt.close()


def plot_species(statistics, view=False, filename='speciation.svg'):
    species_sizes = statistics.get_species_sizes()
    num_generations = len(species_sizes)
    plt.figure()
    for s in range(len(species_sizes[0])):
        sizes = [species_sizes[g][s] if s < len(species_sizes[g]) else 0 for g in range(num_generations)]
        plt.plot(sizes)
    plt.title("Speciation")
    plt.xlabel("Generations")
    plt.ylabel("Size per species")
    plt.savefig(filename)
    if view:
        plt.show()
    plt.close()
