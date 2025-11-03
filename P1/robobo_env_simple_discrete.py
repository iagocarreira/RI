import gymnasium as gym
from gymnasium import spaces
import numpy as np
import time
import threading
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

    En modo evaluación (train=False), el cilindro se moverá
    automáticamente a una posición aleatoria cada 4 segundos.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, ip="localhost", max_steps=200, train=True):
        super().__init__()

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

        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0]),
            high=np.array([100, 500, 6000]),
            dtype=np.float32
        )
        self.action_space = spaces.Discrete(3)

        self.max_steps = max_steps
        self.train = train
        self.episode = 0
        self.step_count = 0
        self.last_size = 0.0
        self.size_history = deque(maxlen=10)
        self.steps_sin_verlo = 0
        self.max_sin_verlo = 10
        self.visible_streak = 0

        self.move_interval = 4.0  
        self._stop_thread = False
        self._thread = None
        self.cylinder_id = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.sim.resetSimulation()
        time.sleep(0.3)
        self.sim.setRobotLocation(
            0, position={"x": 0, "y": 0, "z": 0},
            rotation={"x": 0, "y": -90, "z": 0}
        )
        self.rob.moveTiltTo(120, 30)

        objects = self.sim.getObjects()
        if objects:
            cid = "cylinder" if "cylinder" in objects else list(objects)[0]
            self.cylinder_id = cid
            self.sim.setObjectLocation(
                cid,
                position={"x": np.random.uniform(-800, 800),
                          "y": 0,
                          "z": np.random.uniform(-800, 800)},
                rotation={"x": 0, "y": 0, "z": 0}
            )

        self._stop_thread = False
        if not self.train:
            if self._thread is None or not self._thread.is_alive():
                self._thread = threading.Thread(target=self._move_cylinder_loop, daemon=True)
                self._thread.start()

        self.size_history.clear()
        obs = self._get_obs()
        self.size_history.append(obs[1])
        self.step_count = 0
        self.last_size = obs[1]
        self.last_frontC = obs[2]

        print(f"\n===== EPISODIO =====")
        print(f"Estado inicial: {np.round(obs, 2)}")
        return obs, {}

    def step(self, action):
        self.step_count += 1

        if action == 0:
            self.rob.moveWheelsByTime(60, 60, 0.25)  
        elif action == 1:
            self.rob.moveWheelsByTime(-30, 30, 0.25)  
        elif action == 2:
            self.rob.moveWheelsByTime(30, -30, 0.25) 

        obs = self._get_obs()
        reward = self._compute_reward(obs)
        terminated, success, failure = self._check_termination(obs)
        truncated = self.step_count >= self.max_steps

        self.last_size = obs[1]
        self.size_history.append(obs[1])

        print(f"Paso {self.step_count:03d} | Acción: {action} | Estado: {np.round(obs,2)} | Rew: {reward:+.2f}")
        return obs, float(reward), bool(terminated), bool(truncated), {"success": success, "failure": failure}

    
    def _get_obs(self):
        blob = self.rob.readColorBlob(Color.RED)
        front = self.rob.readIRSensor(IR.FrontC)
        return np.array([blob.posx, blob.size, front], dtype=np.float32)

    def _compute_reward(self, obs):
        x, size, front = obs
        r = 0.0
        if size > 0:
            self.steps_sin_verlo = 0
            self.visible_streak += 1
        else:
            self.steps_sin_verlo += 1
            self.visible_streak = 0
            r -= 1.5

        if size > self.last_size and size > 0:
            r += 2.0
        if 0 < size < self.last_size:
            r -= 1.0
        if self.visible_streak > 3:
            r += 0.3 * self.visible_streak
        if 40 < x < 60 and size > 0:
            r += 1.0
        if front > 100:
            r -= 3.0
        if self.steps_sin_verlo > self.max_sin_verlo:
            r -= 0.5
            self.steps_sin_verlo = 0
        return r

    def _check_termination(self, obs):
        x, size, front = obs
        if size > 300:
            self.stop_cylinder_motion()
            self.rob.moveWheelsByTime(20, 20, 2)
            print("--- ÉXITO ---")
            return True, True, False
        if front > 70 and all(s == 0 for s in self.size_history):
            self.stop_cylinder_motion()
            print("--- FRACASO ---")
            return True, False, True
        return False, False, False

    def _move_cylinder_loop(self):
        """Hilo que mueve el cilindro cada 4 segundos mientras esté en evaluación."""
        while not self._stop_thread:
            try:
                objects = self.sim.getObjects()
                if not objects:
                    time.sleep(self.move_interval)
                    continue

                cid = "cylinder" if "cylinder" in objects else list(objects)[0]
                new_x = np.random.uniform(-800, 800)
                new_z = np.random.uniform(-800, 800)
                self.sim.setObjectLocation(
                    cid,
                    position={"x": new_x, "y": 0, "z": new_z},
                    rotation={"x": 0, "y": 0, "z": 0}
                )
                print(f"Cilindro movido a ({new_x:.1f}, {new_z:.1f})")
            except Exception as e:
                print(f"⚠️ Error moviendo cilindro: {e}")
            time.sleep(self.move_interval)

    def stop_cylinder_motion(self):
        """Detiene el hilo del cilindro (al finalizar el episodio)."""
        if not self.train:
            self._stop_thread = True
            print("Movimiento del cilindro detenido.")

    
    def close(self):
        """Cierra conexiones y detiene el movimiento automático."""
        self._stop_thread = True
        if self._thread:
            self._thread.join(timeout=1)
            self._thread = None
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
