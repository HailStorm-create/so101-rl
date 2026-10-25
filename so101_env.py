import numpy as np 
import mujoco
import gymnasium

class SO101ReachEnv(gymnasium.Env): 
    def __init__(self): 
        self.model = mujoco.MjModel.from_xml_path("third_party/SO-ARM100/Simulation/SO101/scene.xml")
        self.data = mujoco.MjData(self.model)
        self.action_space = gymnasium.spaces.Box(low = -1, high = 1, shape = (5,), dtype=np.float32)
        self.observation_space = gymnasium.spaces.Box(low = -np.inf , high = np.inf, shape = (16,), dtype = np.float32)

env = SO101ReachEnv()
print(env.action_space, env.observation_space)