import gymnasium as gym
from gymnasium import spaces
import numpy as np
import time
from collections import deque

from robobopy.Robobo import Robobo
from robobopy.utils.IR import IR
from robobopy.utils.Color import Color
from robobopy.utils.StatusFrequency import StatusFrequency
from robobosim.RoboboSim import RoboboSim


class RoboboEnvMinimal(gym.Env):
    """
    Entorno Robobo simplificado para RL discreto.
    Estado: [blob_x, blob_size, frontC]
    Acciones:
        0 = avanzar
        1 = girar izquierda
        2 = girar derecha
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, ip="localhost", max_steps=200):
        super().__init__()

        # --- Conexión con Robobo y simulador ---
        self.rob = Robobo(ip)
        self.rob.connect()
        self.rob.setStatusFrequency(StatusFrequency.High)
        self.rob.startCamera()
        self.rob.setActiveBlobs(True, False, False, False)
        self.rob.movePanTo(0, 30)
        self.rob.moveTiltTo(120, 30)
        self.rob.wait(0.5)

        self.sim = RoboboSim(ip)
        self.sim.connect()
        time.sleep(0.2)

        # --- Espacios Gym ---
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0]),
            high=np.array([100, 500, 6000]),
            dtype=np.float32
        )
        self.action_space = spaces.Discrete(3)

        # --- Parámetros ---
        self.max_steps = max_steps
        self.episode = 0
        self.step_count = 0
        self.last_size = 0.0
        self.size_history = deque(maxlen=10)
        self.steps_sin_verlo = 0
        self.max_sin_verlo = 10
        self.visible_streak = 0

    # ------------------------------------------------------------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.sim.resetSimulation()
        time.sleep(0.3)
        self.sim.setRobotLocation(
            0, position={"x": 0, "y": 0, "z": 0},
            rotation={"x": 0, "y": -90, "z": 0}
        )
        self.rob.moveTiltTo(120, 30)

        # Colocar cilindro aleatoriamente
        objects = self.sim.getObjects()
        if objects:
            cid = "cylinder" if "cylinder" in objects else list(objects)[0]
            self.sim.setObjectLocation(
                cid,
                position={"x": np.random.uniform(-800, 800),
                          "y": 0,
                          "z": np.random.uniform(-800, 800)},
                rotation={"x": 0, "y": 0, "z": 0}
            )

        self.size_history.clear()
        obs = self._get_obs()
        self.size_history.append(obs[1])
        self.step_count = 0
        self.last_size = obs[1]
        self.last_frontC = obs[2]

        print(f"\n===== EPISODIO =====")
        print(f"Estado inicial: {np.round(obs, 2)}")
        return obs, {}

    # ------------------------------------------------------------
    def step(self, action):
        self.step_count += 1

        # --- Ejecutar acción ---
        if action == 0:
            self.rob.moveWheelsByTime(60, 60, 0.25)   # avanzar
        elif action == 1:
            self.rob.moveWheelsByTime(-30, 30, 0.25)  # girar izquierda
        elif action == 2:
            self.rob.moveWheelsByTime(30, -30, 0.25)  # girar derecha

        # --- Leer nuevo estado ---
        obs = self._get_obs()
        reward = self._compute_reward(obs)
        terminated, success, failure = self._check_termination(obs)
        truncated = self.step_count >= self.max_steps

        self.last_size = obs[1]
        self.size_history.append(obs[1])

        # --- Mostrar información paso a paso ---
        print(f"Paso {self.step_count:03d} | Acción: {action} | Estado: {np.round(obs,2)} | Rew: {reward:+.2f}")

        return obs, float(reward), bool(terminated), bool(truncated), {"success": success, "failure": failure}

    # ------------------------------------------------------------
    def _get_obs(self):
        blob = self.rob.readColorBlob(Color.RED)
        front = self.rob.readIRSensor(IR.FrontC)
        return np.array([blob.posx, blob.size, front], dtype=np.float32)

    # ------------------------------------------------------------
    def _compute_reward(self, obs):
        x, size, front = obs
        r = 0.0

        # --- Control de visibilidad ---
        if size > 0:
            self.steps_sin_verlo = 0
            self.visible_streak += 1
        else:
            self.steps_sin_verlo += 1
            self.visible_streak = 0
            r -= 1.5  # penaliza perder el blob

        # ✅ Recompensa por acercarse
        if size > self.last_size and size > 0:
            r += 2.0

        # ❌ Penalización por alejarse
        if 0 < size < self.last_size:
            r -= 1.0

        # ✅ Recompensa por mantener visión estable
        if self.visible_streak > 3:  # 3 pasos seguidos viéndolo
            r += 0.3 * self.visible_streak

        # ✅ Recompensa por centrado
        if 40 < x < 60 and size > 0:
            r += 1.0

        # ❌ Penalización por choque
        if front > 100:
            r -= 3.0

        # ❌ Penalización por estar muchos pasos sin verlo
        if self.steps_sin_verlo > self.max_sin_verlo:
            r -= 0.5
            self.steps_sin_verlo = 0

        return r

    # ------------------------------------------------------------
    def _check_termination(self, obs):
        x, size, front = obs
        if size > 300:
            print("--- ✅ ÉXITO ---")
            return True, True, False
        if front > 70 and all(s == 0 for s in self.size_history):
            print("--- ❌ FRACASO ---")
            return True, False, True
        return False, False, False

    # ------------------------------------------------------------
    def close(self):
        try:
            self.rob.stopMotors()
        except:
            pass
        try:
            self.rob.disconnect()
        except:
            pass
        try:
            self.sim.disconnect()
        except:
            pass
