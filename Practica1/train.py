import os
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor
from robobo_env import RoboboSimEnv

# Crear entorno
env = RoboboSimEnv()
check_env(env, warn=True)

# Monitoriza recompensas
log_dir = "./logs/"
os.makedirs(log_dir, exist_ok=True)
env = Monitor(env, log_dir)

# Modelo PPO
model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    learning_rate=3e-4,
    n_steps=512,
    batch_size=64,
    gamma=0.99,
    tensorboard_log="./tb_robobo/"
)

# Entrenamiento
TIMESTEPS = 10000
model.learn(total_timesteps=TIMESTEPS)

# Guardar modelo
model.save("ppo_robobo")

env.close()

# ----------------------------
# VISUALIZACIÓN DE RESULTADOS
# ----------------------------
import pandas as pd
import seaborn as sns

# El monitor genera "monitor.csv"
monitor_file = os.path.join(log_dir, [f for f in os.listdir(log_dir) if f.endswith('.csv')][0])
data = pd.read_csv(monitor_file, skiprows=1)
data['timestep'] = np.arange(len(data))
sns.lineplot(data=data, x="timestep", y="r")
plt.title("Evolución de la recompensa por episodio")
plt.xlabel("Episodios")
plt.ylabel("Recompensa")
plt.tight_layout()
plt.savefig("training_rewards.png")
plt.show()
