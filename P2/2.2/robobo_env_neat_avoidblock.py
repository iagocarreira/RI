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
    Práctica 2.1 – Acercamiento al cilindro rojo inmóvil
    ----------------------------------------------------
    Objetivo:
        El Robobo debe aprender a acercarse al cilindro rojo inmóvil.

    Particularidades:
        - Posición inicial fija: {'x': -1000.0, 'y': 39.0, 'z': -400.0}
        - Física simplificada y velocidad de simulación x4 (ajustar en RoboboSim).
        - No usar modelos de mundo aleatorios.

    Estado (observación):
        [visible, x_pos, delta_x, frontC]

    Acciones:
        0 - Avanzar
        1 - Girar izquierda
        2 - Girar derecha

    Éxito:
        - Si ha visto el cilindro en los últimos 10 pasos y frontC > 60.

    Fracaso:
        - Si no lo ve en 10 pasos consecutivos.
        - Si frontC > 40 y visible == 0 (colisión ciega).
        - Si se agota max_steps.
    """

    def __init__(self, ip="localhost", max_steps=40):
        super().__init__()

        # --- Inicializa Robobo y Sim ---
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
        self.observation_space = spaces.Box(low=-1, high=1, shape=(4,), dtype=np.float32)
        self.action_space = spaces.Discrete(3)

        self.max_steps = max_steps
        self.step_count = 0
        self.last_x = 0.0
        self.recent_visibility = deque(maxlen=10)

    def reset(self, seed=None, options=None):
        """Reinicia el entorno. Escenario seleccionado manualmente."""
        super().reset(seed=seed)
        self.sim.resetSimulation()
        time.sleep(0.3)

        # Posición inicial exacta solicitada
        self.sim.setRobotLocation(
            0,
            position={"x": -1000.0, "y": 39.0, "z": -400.0},
            # rotation={"x": 0, "y": -90, "z": 0},
        )

        # Reinicia contadores
        self.step_count = 0
        self.recent_visibility.clear()

        obs = self._get_obs()
        self.last_x = obs[1]
        return obs, {}

    def _get_obs(self):
        """Devuelve [visible, x_pos, delta_x, frontC_norm]."""
        blob = self.rob.readColorBlob(Color.RED)
        visible = 1.0 if blob.size > 0 else 0.0
        x_pos_norm = (blob.posx - 50) / 50.0 if visible else 0.0
        delta_x = x_pos_norm - self.last_x
        frontC = self.rob.readIRSensor(IR.FrontC) / 4095.0

        obs = np.array([visible, x_pos_norm, delta_x, frontC], dtype=np.float32)
        self.last_x = x_pos_norm
        return obs

    def step(self, action):
        self.step_count += 1

        # --- Ejecutar acción ---
        if action == 0:
            self.rob.moveWheelsByTime(60, 60, 0.3)   # Avanzar
        elif action == 1:
            self.rob.moveWheelsByTime(-30, 30, 0.3)  # Girar izquierda
        elif action == 2:
            self.rob.moveWheelsByTime(30, -30, 0.3)  # Girar derecha

        time.sleep(0.1)  # permitir actualización sensores

        obs = self._get_obs()
        reward = self._compute_fitness(obs)

        # Actualiza historial de visibilidad
        visible = int(obs[0])
        self.recent_visibility.append(visible)

        # Condiciones de éxito y fracaso
        seen_recently = any(self.recent_visibility)
        frontC_real = obs[3] * 4095

        success = seen_recently and frontC_real > 60
        failure = (not seen_recently and self.step_count > 15) or (
            frontC_real > 40 and visible == 0
        )
        done = success or failure or self.step_count >= self.max_steps

        return obs, reward, done, False, {}

    def _compute_fitness(self, obs):
        """Recompensa basada en centrado, estabilidad y acercamiento."""
        visible, x_pos, delta_x, frontC_norm = obs
        frontC = frontC_norm * 4095
        fitness = 0.0

        if visible > 0:
            # centrado visual (más centrado, mayor fitness)
            fitness += (1 - abs(x_pos)) * 2.0

            # estabilidad (mantener el cilindro centrado)
            if abs(delta_x) < 0.05:
                fitness += 1.0
        else:
            fitness -= 2.0  # penalización por perderlo de vista

        # acercamiento
        fitness += frontC_norm * 10.0

        # contacto con el cilindro (éxito)
        if frontC > 60:
            fitness += 25.0
        # colisión sin verlo
        elif frontC > 40 and visible == 0:
            fitness -= 10.0

        # penalización por tiempo
        fitness -= 0.05
        return fitness

    def close(self):
        try:
            self.rob.stopMotors()
            self.rob.disconnect()
            self.sim.disconnect()
        except:
            pass
