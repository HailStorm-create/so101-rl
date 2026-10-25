import numpy as np 
import mujoco
import gymnasium

class SO101ReachEnv(gymnasium.Env): 

    """
    created init environment 
    creating env type + arm load + live state + RL algorithm # controls
    """

    def __init__(self): 
        self.model = mujoco.MjModel.from_xml_path("third_party/SO-ARM100/Simulation/SO101/scene.xml")
        self.data = mujoco.MjData(self.model)
        self.action_space = gymnasium.spaces.Box(low = -1, high = 1, shape = (5,), dtype=np.float32)
        self.observation_space = gymnasium.spaces.Box(low = -np.inf , high = np.inf, shape = (16,), dtype = np.float32)

    def reset(self, seed=None, options=None): 
        super().reset(seed = seed)
        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)
        self.np_random.uniform(low = [0, 0, 0])

    

env = SO101ReachEnv()
print(env.action_space, env.observation_space)