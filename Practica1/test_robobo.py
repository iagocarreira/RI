import time
from robobo_env import RoboboSimEnv

def test_env():
    env = RoboboSimEnv()
    obs = env.reset()
    print("Observación inicial:", obs)

    for step in range(20):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"Paso {step+1}, Acción: {action}, Observación: {obs}, Recompensa: {reward:.2f}")
        if terminated:
            print("Episodio terminado en paso", step+1)
            break
        time.sleep(0.5)  # para ver lo que hace el robot con calma

    env.close()

if __name__ == "__main__":
    test_env()
