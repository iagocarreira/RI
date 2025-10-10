import os
import time
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from robobo_env_simple_discrete import RoboboEnvMinimal

# Crear entorno silencioso
env = RoboboEnvMinimal()
env = Monitor(env)
env = DummyVecEnv([lambda: env])

log_dir = "ppo_robobo_simple_logs"
os.makedirs(log_dir, exist_ok=True)

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

print("\n===== ENTRENANDO AGENTE PPO  =====")
start = time.time()
model.learn(total_timesteps=5000, progress_bar=True)
print(f"\nEntrenamiento completado en {time.time()-start:.1f}s")

model.save("ppo_robobo_simple_v1")
print(" Modelo guardado: ppo_robobo_simple_v1.zip")
