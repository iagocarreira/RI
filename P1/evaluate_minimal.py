import time
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from robobo_env_simple_discrete import RoboboEnvMinimal

# -----------------------------
# Configuración de evaluación
# -----------------------------
MODEL_PATH = "ppo_robobo_simple_v1.zip"
N_EPISODES = 5

print("\n===== EVALUACIÓN DEL AGENTE PPO =====")

env = RoboboEnvMinimal(train=False)
model = PPO.load(MODEL_PATH, env=env)
print(f"Modelo cargado desde: {MODEL_PATH}")

total_rewards = []
success_count, failure_count = 0, 0
episodes_x, success_y, failure_y = [0], [0], [0]

for ep in range(N_EPISODES):
    obs, _ = env.reset()
    done = False
    ep_reward = 0
    print(f"\nEpisodio {ep + 1}/{N_EPISODES}")

    robot_path = []
    cylinder_path = []
    cylinder_moves = 0

    try:
        robot_loc = env.sim.getRobotLocation(0)["position"]
        cyl_loc = env.sim.getObjectLocation(env.cylinder_id)["position"]
        robot_path.append((robot_loc["x"], robot_loc["z"]))
        cylinder_path.append((cyl_loc["x"], cyl_loc["z"]))
    except Exception:
        pass

    last_cyl_loc = None

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        ep_reward += reward

        try:
            robot_loc = env.sim.getRobotLocation(0)["position"]
            cyl_loc = env.sim.getObjectLocation(env.cylinder_id)["position"]

            robot_path.append((robot_loc["x"], robot_loc["z"]))

            if last_cyl_loc is None or \
               abs(cyl_loc["x"] - last_cyl_loc[0]) > 1 or abs(cyl_loc["z"] - last_cyl_loc[1]) > 1:
                cylinder_path.append((cyl_loc["x"], cyl_loc["z"]))
                cylinder_moves += 1
                last_cyl_loc = (cyl_loc["x"], cyl_loc["z"])
        except Exception:
            pass

        print(f"→ Acción: {int(action)} | Estado: {np.round(obs, 2)} | Recompensa: {reward:+.2f}")
        time.sleep(0.2)

  
    plt.figure(figsize=(6, 6))
    plt.title(f"Trayectoria episodio {ep+1}")

    # Robot
    if robot_path:
        rx, rz = zip(*robot_path)
        plt.plot(rx, rz, 'b-', label='Trayectoria Robot')
        plt.scatter(rx[0], rz[0], color='green', s=100, label='Inicio Robot')
        plt.scatter(rx[-1], rz[-1], color='red', s=100, label='Fin Robot')

    # Cilindro
    if cylinder_path:
        cx, cz = zip(*cylinder_path)
        plt.plot(cx, cz, 'orange', linestyle='--', label='Cilindro')
        plt.scatter(cx, cz, color='orange', s=60)
        for i, (x, z) in enumerate(cylinder_path):
            plt.text(x, z, str(i+1), fontsize=9, ha='center', va='center', color='black')

    plt.xlabel("Posición X")
    plt.ylabel("Posición Z")
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.tight_layout()
    fname = f"trayectoria_ep{ep+1}.png"
    plt.savefig(fname)
    print(f"📸 Gráfica de trayectoria guardada: {fname}")
    plt.close()


    print(f"\nEpisodio {ep + 1} finalizado | Recompensa total: {ep_reward:.2f}")
    if info.get("success"):
        success_count += 1
        print("Resultado: ÉXITO")
    elif info.get("failure"):
        failure_count += 1
        print("Resultado: FRACASO")
    else:
        print("Finalizó por límite de pasos")

    episodes_x.append(ep + 1)
    success_y.append(success_count)
    failure_y.append(failure_count)
    total_rewards.append(ep_reward)
    time.sleep(1.0)


avg_reward = np.mean(total_rewards)
print("\n===== RESUMEN DE EVALUACIÓN =====")
print(f"Episodios evaluados: {N_EPISODES}")
print(f"Recompensa media: {avg_reward:.2f}")
print(f"Recompensas individuales: {np.round(total_rewards, 2)}")
print(f"Éxitos: {success_count} | Fracasos: {failure_count}")

env.close()
print("Evaluación completada.")


plt.figure(figsize=(8, 5))
plt.plot(episodes_x, success_y, marker='o', color='green', linewidth=2, label="Éxitos acumulados")
plt.plot(episodes_x, failure_y, marker='o', color='red', linewidth=2, label="Fracasos acumulados")
plt.title("Progresión de éxitos y fracasos por episodio")
plt.xlabel("Episodios")
plt.ylabel("Número acumulado")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig("evaluation_success_failure.png")
print("Gráfica de evaluación guardada: evaluation_success_failure.png")
plt.close()
