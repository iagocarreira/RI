import neat, pickle, numpy as np, matplotlib.pyplot as plt
from robobo_env_neat_avoidblock import RoboboEnvNeatAvoidBlock
from stable_baselines3 import PPO

# ----------------------------
# 1) Cargar NEAT (AE)
# ----------------------------
config_path = "config_neat_avoidblock.txt"
config = neat.Config(
    neat.DefaultGenome, neat.DefaultReproduction,
    neat.DefaultSpeciesSet, neat.DefaultStagnation,
    config_path
)

with open("winner_avoidblock.pkl", "rb") as f:
    winner = pickle.load(f)

ae_net = neat.nn.FeedForwardNetwork.create(winner, config)

# ----------------------------
# 2) Cargar política RL (P1)
# ----------------------------
rl_model = PPO.load("ppo_robobo_simple_v1.zip")

# ----------------------------
# 3) Crear entorno
# ----------------------------
env = RoboboEnvNeatAvoidBlock()
obs, _ = env.reset()

robot_path = []
done = False
use_rl = False

print("\n### VALIDACIÓN P2.3 — AE → RL ###\n")

while not done:

    visible = obs[0]
    x_pos = obs[1]
    size   = obs[2]
    frontC = obs[3]

    # ----------------------------------------
    # Condición REAL de cambio AE → RL
    # ----------------------------------------
    if (not use_rl) and (visible == 1) :
        use_rl = True
        print("\n>>> OBJETIVO DETECTADO — CAMBIO A POLÍTICA RL (P1)\n")

    # ----------------------------------------
    # Selección de acción
    # ----------------------------------------
    if not use_rl:
        nn_out = ae_net.activate(obs)
        action = int(np.argmax(nn_out))
        controller = "AE"
    else:
        obs_rl = np.array([x_pos, size, frontC], dtype=np.float32)
        action, _ = rl_model.predict(obs_rl, deterministic=True)
        
        # ✅ asegurar acción escalar entero
        action = int(np.asarray(action).item())
        
        controller = "RL"

    obs, reward, done, _, _ = env.step(action)

    # Guardar trayectoria
    try:
        pos = env.sim.getRobotLocation(0)["position"]
        robot_path.append((pos["x"], pos["z"], controller))
    except:
        pass

env.close()

# ----------------------------
# 4) Generar gráfico
# ----------------------------
if robot_path:
    rx = [p[0] for p in robot_path]
    rz = [p[1] for p in robot_path]
    ctrl = [p[2] for p in robot_path]

    colors = ["blue" if c=="AE" else "red" for c in ctrl]

    plt.figure(figsize=(6, 6))
    plt.scatter(rx, rz, c=colors, s=12)
    plt.plot(rx, rz, 'k-', alpha=0.4)
    plt.scatter(rx[0], rz[0], c='green', s=100, label="Inicio")
    plt.scatter(rx[-1], rz[-1], c='black', s=100, label="Fin")
    plt.xlabel("X")
    plt.ylabel("Z")
    plt.title("Trayectoria Robobo – P2.3 (AE → RL)")
    plt.grid(True)
    plt.axis("equal")
    plt.legend()
    plt.savefig("trajectory_p23_aerl.png")
    plt.close()

    print("✅ Trayectoria guardada como: trajectory_p23_aerl.png")

print("\nFin de la validación.\n")
