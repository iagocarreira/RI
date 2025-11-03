import neat, pickle, numpy as np, matplotlib.pyplot as plt
from robobo_env_neat_avoidblock import RoboboEnvNeatAvoidBlock

config_path = "config_neat_avoidblock.txt"
config = neat.Config(
    neat.DefaultGenome, neat.DefaultReproduction,
    neat.DefaultSpeciesSet, neat.DefaultStagnation,
    config_path
)

with open("winner_avoidblock.pkl", "rb") as f:
    winner = pickle.load(f)

net = neat.nn.FeedForwardNetwork.create(winner, config)
env = RoboboEnvNeatAvoidBlock()
obs, _ = env.reset()
robot_path = []
done = False

while not done:
    output = net.activate(obs)
    action = np.argmax(output)
    obs, reward, done, _, _ = env.step(action)
    try:
        pos = env.sim.getRobotLocation(0)["position"]
        robot_path.append((pos["x"], pos["z"]))
    except:
        pass

env.close()

if robot_path:
    rx, rz = zip(*robot_path)
    plt.figure(figsize=(6, 6))
    plt.plot(rx, rz, 'b-', label="Trayectoria Robobo")
    plt.scatter(rx[0], rz[0], c='green', s=80, label="Inicio")
    plt.scatter(rx[-1], rz[-1], c='red', s=80, label="Fin")
    plt.xlabel("X")
    plt.ylabel("Z")
    plt.legend()
    plt.grid(True)
    plt.axis("equal")
    plt.title("Trayectoria Robobo – Validación NEAT (Avoid the Block)")
    plt.savefig("trajectory_neat_avoidblock.png")
    plt.close()
    print("✅ Trayectoria guardada como trajectory_neat_avoidblock.png")
