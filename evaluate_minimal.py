import time
import numpy as np
from stable_baselines3 import PPO
from robobo_env_simple_discrete import RoboboEnvMinimal


# -----------------------------
# Configuración de evaluación
# -----------------------------
MODEL_PATH = "ppo_robobo_simple_v1.zip"
N_EPISODES = 5

print("\n===== EVALUACIÓN DEL AGENTE PPO =====")

# Cargar modelo y entorno
env = RoboboEnvMinimal()
model = PPO.load(MODEL_PATH, env=env)
print(f"✅ Modelo cargado desde: {MODEL_PATH}")

# -----------------------------
# Bucle de evaluación
# -----------------------------
total_rewards = []

for ep in range(N_EPISODES):
    obs, _ = env.reset()
    done = False
    ep_reward = 0
    print(f"\n🌍 Episodio {ep + 1}/{N_EPISODES}")

    while not done:
        # Obtener acción del modelo (modo determinista)
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)

        done = terminated or truncated
        ep_reward += reward

        # Mostrar información
        print(f"→ Acción: {int(action)} | Estado: {np.round(obs,2)} | Recompensa: {reward:+.2f}")
        time.sleep(0.2)

    print(f"\n🏁 Episodio {ep + 1} finalizado | Recompensa total: {ep_reward:.2f}")
    if info.get("success"):
        print("✅ Resultado: ÉXITO")
    elif info.get("failure"):
        print("❌ Resultado: FRACASO")
    else:
        print("⚪ Finalizó por límite de pasos")
    total_rewards.append(ep_reward)
    time.sleep(1.0)

# -----------------------------
# Resumen final
# -----------------------------
avg_reward = np.mean(total_rewards)
print("\n===== RESUMEN DE EVALUACIÓN =====")
print(f"Episodios evaluados: {N_EPISODES}")
print(f"Recompensa media: {avg_reward:.2f}")
print(f"Recompensas individuales: {np.round(total_rewards,2)}")

env.close()
print("✅ Evaluación completada.")
