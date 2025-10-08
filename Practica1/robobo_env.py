import gymnasium as gym
from gymnasium import spaces
import numpy as np
from robobopy.Robobo import Robobo
from robobopy.utils.BlobColor import BlobColor
from robobopy.utils.IR import IR
import time

class RoboboSimEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.robobo = Robobo('localhost')
        self.robobo.connect()
        self.robobo.moveTiltTo(90, 100)
        self.robobo.movePanTo(0, 100)
        self.robobo.setActiveBlobs(True, False, False, False)

        # Observación: posición horizontal del blob rojo y distancia IR frontal
        self.observation_space = spaces.Box(low=0, high=100, shape=(2,), dtype=np.float32)
        # Acciones discretas: 0=avanzar, 1=girar izq, 2=girar dcha, 3=detenerse
        self.action_space = spaces.Discrete(4)

    def reset(self):
        self.robobo.stopMotors()
        time.sleep(1)
        blob = self.robobo.readColorBlob(BlobColor.RED)
        posx = blob.posx if blob else 50
        distancia = self.robobo.readIRSensor(IR.FrontC)
        return np.array([posx, distancia], dtype=np.float32)

    def step(self, action):
        if action == 0:
            self.robobo.moveWheels(50, 50)
        elif action == 1:
            self.robobo.moveWheels(-30, 30)
        elif action == 2:
            self.robobo.moveWheels(30, -30)
        elif action == 3:
            self.robobo.stopMotors()
        
        time.sleep(0.2)

        blob = self.robobo.readColorBlob(BlobColor.RED)
        posx = blob.posx if blob else 50
        distancia = self.robobo.readIRSensor(IR.FrontC)

        # reward = (distancia + abs(posx - 50) * 0.5)
        reward = distancia

        terminated = distancia < 1

        obs = np.array([posx, distancia], dtype=np.float32)
        return obs, reward, terminated, False, {}

    def render(self):
        pass

    def close(self):
        self.robobo.stopMotors()
        self.robobo.disconnect()
