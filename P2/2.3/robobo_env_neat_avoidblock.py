import gymnasium as gym
from gymnasium import spaces
import numpy as np
import time
from collections import deque
from robobopy.Robobo import Robobo
from robobosim.RoboboSim import RoboboSim
from robobopy.utils.IR import IR
from robobopy.utils.Color import Color
from robobopy.utils.StatusFrequency import StatusFrequency


class RoboboEnvNeatAvoidBlock(gym.Env):
    """
    Robobo NEAT – Evitar bloque y buscar cilindro rojo.
    ---------------------------------------------------
    Entradas: [visible, x_pos, size, frontC, frontLL, frontRR]
    Acciones:
        0 - Avanzar recto
        1 - Girar izquierda
        2 - Girar derecha
        3 - Curva suave derecha
        4 - Curva suave izquierda

    ÉXITO:  Ha visto el blob en los últimos 15 pasos y ahora:
             size > 110 y frontC > 60
    FRACASO: frontC > 60 sin haber visto blob reciente.
    """

    def __init__(self, ip="localhost", max_steps=150):
        super().__init__()

        # --- Conexión con Robobo y Simulador ---
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
        time.sleep(0.3)

        # --- Espacios Gym ---
        self.observation_space = spaces.Box(
            low=np.array([0, -100, 0, 0, 0, 0], dtype=np.float32),
            high=np.array([1, 100, 200, 4095, 4095, 4095], dtype=np.float32),
            dtype=np.float32
        )
        self.action_space = spaces.Discrete(5)

        # --- Variables internas ---
        self.max_steps = max_steps
        self.step_count = 0
        self.last_phase = None
        self.prev_frontC = 0.0
        self.same_action = None
        self.same_action_count = 0

        # --- Memoria de visión (últimos 15 pasos) ---
        self.recent_seen = deque(maxlen=15)  # 1 si visible, 0 si no
        self.recent_sizes = deque(maxlen=15) # tamaños vistos

    # ------------------------------------------------------------------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.sim.resetSimulation()
        time.sleep(0.3)
        self.sim.setRobotLocation(
            0,
            rotation={"x": 0, "y": 90, "z": 0},
        )        
        self.step_count = 0
        self.same_action = None
        self.same_action_count = 0
        self.last_phase = None
        self.recent_seen.clear()
        self.recent_sizes.clear()
        self.rob.movePanTo(0, 30)
        self.rob.moveTiltTo(90, 30)

        obs = self._get_obs()
        return obs, {}

    # ------------------------------------------------------------------
    def _get_obs(self):
        blob = self.rob.readColorBlob(Color.RED)
        visible = 1.0 if blob.size > 0 else 0.0
        x_pos = blob.posx - 50 if visible else 0.0
        size = blob.size

        frontC = self.rob.readIRSensor(IR.FrontC)
        frontLL = self.rob.readIRSensor(IR.FrontLL)
        frontRR = self.rob.readIRSensor(IR.FrontRR)

        # Guardamos memoria visual
        self.recent_seen.append(visible)
        self.recent_sizes.append(size)

        obs = np.array([visible, x_pos, size, frontC, frontLL, frontRR], dtype=np.float32)
        return obs

    # ------------------------------------------------------------------
    def step(self, action):
        self.step_count += 1
        action = int(np.clip(round(action), 0, 4))

        # Control de repetición de acción
        if self.same_action == action:
            self.same_action_count += 1
        else:
            self.same_action = action
            self.same_action_count = 0

        # --- Ejecutar acción ---
        act_name = ["↑", "↺", "↻", "↗", "↖"][action]
        if action == 0:
            self.rob.moveWheelsByTime(60, 60, 0.4)
        elif action == 1:
            self.rob.moveWheelsByTime(-10, 10, 0.5)
        elif action == 2:
            self.rob.moveWheelsByTime(10, -10, 0.5)
        elif action == 3:
            self.rob.moveWheelsByTime(50, 30, 0.3)
        elif action == 4:
            self.rob.moveWheelsByTime(30, 50, 0.3)
        time.sleep(0.1)

        # --- Leer sensores ---
        obs = self._get_obs()
        reward, phase = self._compute_reward(obs)
        visible, x_pos, size, frontC, frontLL, frontRR = obs

        # --- Criterios de éxito y fracaso ---
        seen_recently = any(v == 1 for v in self.recent_seen)
        avg_recent_size = np.mean([s for s in self.recent_sizes if s > 0]) if any(self.recent_sizes) else 0

        success = False
        fail = False
        done = False

        # Éxito: ha visto en últimos 15 pasos, blob grande y cerca
        if seen_recently and avg_recent_size > 100 and frontC > 30:
            success = True
            reward += 80.0
            done = True

        # Fracaso: no ve blob y choca
        elif visible == 0 and frontC > 60:
            fail = True
            reward -= 40.0
            done = True

        # Fin por pasos o repetición
        elif self.step_count >= self.max_steps or self.same_action_count > 40:
            done = True

        print(f"[Paso {self.step_count:03d}] {act_name} | {phase:<9} | "
              f"Vis={int(visible)} | Size={size:.1f} | x={x_pos:+.1f} | "
              f"Front(L={frontLL:.0f},C={frontC:.0f},R={frontRR:.0f}) | Rew={reward:+6.2f}")

        if done:
            if success:
                print(f"✅ ÉXITO: Blob grande y cerca tras {self.step_count} pasos\n")
            elif fail:
                print(f"❌ FRACASO: Colisión sin haber visto el blob\n")
            else:
                print(f"⏹️ Fin del episodio ({self.step_count} pasos)\n")

        return obs, reward, done, False, {}

    # ------------------------------------------------------------------
    def _compute_reward(self, obs):
        visible, x_pos, size, frontC, frontLL, frontRR = obs
        reward = 0.0
        phase = "explora"

        # --- Exploración y rodeo según IR ---
        if frontC > 40:
            reward -= 5.0
            phase = "evita"
        elif frontC < 20 and frontRR > 20 and frontLL < 15:
            reward += 1.5
            phase = "bordeo-izq"
        elif frontC < 20 and frontLL > 20 and frontRR < 15:
            reward += 1.5
            phase = "bordeo-der"
        elif frontC < 10 and frontLL < 10 and frontRR < 10:
            reward += 0.5
            phase = "libre"
        else:
            reward -= 0.2

        # --- Cambio de bordeo (rodeo completo) ---
        if self.last_phase and "bordeo" in self.last_phase and "bordeo" in phase and self.last_phase != phase:
            reward += 3.0
            phase = "rodea"
        self.last_phase = phase

        # --- Detección visual ---
        if visible == 1:
            reward += 10.0 + 0.2 * size
            phase = "acerca"

        # --- Penalizaciones ---
        if self.same_action_count > 20:
            reward -= 3.0
        if abs(frontC - self.prev_frontC) < 2:
            reward -= 0.2
        self.prev_frontC = frontC

        reward -= 0.05  # coste temporal
        return reward, phase

    # ------------------------------------------------------------------
    def close(self):
        try:
            self.rob.stopMotors()
            self.rob.disconnect()
            self.sim.disconnect()
        except:
            pass
