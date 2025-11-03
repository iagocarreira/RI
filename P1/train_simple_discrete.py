import os
import time
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from robobo_env_simple_discrete import RoboboEnvMinimal


log_dir = "ppo_robobo_simple_logs"
os.makedirs(log_dir, exist_ok=True)

monitor_path = os.path.join(log_dir, "monitor.csv")

# Crear entorno y registrar métricas
env = RoboboEnvMinimal()
env = Monitor(env, filename=monitor_path)
env = DummyVecEnv([lambda: env])


model = PPO(
    "MlpPolicy",
    env,
    learning_rate=3e-4,
    n_steps=512,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.01,
    verbose=1,
    tensorboard_log=log_dir,
)


print("\n===== ENTRENANDO AGENTE PPO (entorno simple) =====")
start = time.time()
model.learn(total_timesteps=1000, progress_bar=True)
print(f"\nEntrenamiento completado en {time.time() - start:.1f}s")

model.save("ppo_robobo_simple_v1")
print("Modelo guardado: ppo_robobo_simple_v1.zip")


if os.path.exists(monitor_path):
    df = pd.read_csv(monitor_path, skiprows=1)
    df["Episode"] = range(1, len(df) + 1)

    sns.set(style="whitegrid", font_scale=1.2)
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df, x="Episode", y="r", label="Recompensa por episodio", color="blue")
    plt.title("Evolución de la recompensa por episodio (PPO - Robobo)")
    plt.xlabel("Episodios")
    plt.ylabel("Recompensa total")
    plt.legend()
    plt.tight_layout()
    plt.savefig("training_rewards_curve.png")
    plt.close()
    print("Gráfica de recompensas guardada: training_rewards_curve.png")

    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df, x="Episode", y="l", label="Duración del episodio (pasos)", color="orange")
    plt.title("Duración de los episodios durante el entrenamiento")
    plt.xlabel("Episodios")
    plt.ylabel("Número de pasos")
    plt.legend()
    plt.tight_layout()
    plt.savefig("training_episode_length.png")
    plt.close()
    print(" Gráfica de duración de episodios guardada: training_episode_length.png")

else:
    print("No se encontró el archivo monitor.csv. Verifica que el entorno esté envuelto con Monitor correctamente.")
