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
        self.action_space = gymnasium.spaces.Box(low = -1, high = 1, shape = (1,), dtype=np.float32)
        self.observation_space = gymnasium.spaces.Box(low = -np.inf , high = np.inf, shape = (2,), dtype = np.float32)

    def reset(self, seed=None, options=None): 
        super().reset(seed = seed)
        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)
        self.target = self.np_random.uniform(-1.5, 1.5) 
        obs = np.array([self.data.qpos[0], self.target], dtype=np.float32)
        self.stepcounter = 0
        return obs, {}

    def step(self, action): 
        self.data.ctrl[0] = np.clip(self.data.ctrl[0] + action[0] * 0.1 , -1.9, 1.9, out=None)
        for i in range(10): mujoco.mj_step(self.model, self.data) 
        self.stepcounter += 1
        obs = np.array([self.data.qpos[0], self.target], dtype=np.float32)
        distance = abs(self.target - self.data.qpos[0])
        reward = -(distance)
        terminated = distance < 0.05 
        truncated = self.stepcounter >= 100
        info = {}
        return obs, reward, terminated, truncated, info

    
if __name__ == "__main__":         
    env = SO101ReachEnv()
    print(env.action_space, env.observation_space)      