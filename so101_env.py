import numpy as np 
import mujoco
import gymnasium

class SO101ReachEnv(gymnasium.Env): 

    """
    created init environment 
    creating env type + arm load + live state + RL algorithm # controls
    """

    def __init__(self): 
        self.model = mujoco.MjModel.from_xml_path("so101/scene_reach.xml")
        self.data = mujoco.MjData(self.model)
        self.action_space = gymnasium.spaces.Box(low = -1, high = 1, shape = (5,), dtype=np.float32)
        self.observation_space = gymnasium.spaces.Box(low = -np.inf , high = np.inf, shape = (11,), dtype = np.float32)

    def reset(self, seed=None, options=None): 
        super().reset(seed = seed)
        mujoco.mj_resetData(self.model, self.data)
        self.target = self.np_random.uniform(low=[0.15,-0.15,.05], high=[0.30,.15,.20]) 
        self.data.mocap_pos[0] = self.target
        mujoco.mj_forward(self.model, self.data)
        obs = self._get_obs()
        self.stepcounter = 0
        return obs, {}

    def _get_obs(self): 
        sliced = self.data.qpos[:5]
        gripper_pos = self.data.site("gripperframe").xpos
        obs = np.concatenate([sliced, gripper_pos, self.target])
        return obs.astype(np.float32)
                
    def step(self, action): 
        self.data.ctrl[:5] = np.clip(self.data.ctrl[:5] + action * 0.1 , self.model.actuator_ctrlrange[:5, 0], self.model.actuator_ctrlrange[:5, 1], out=None)
        for i in range(10): mujoco.mj_step(self.model, self.data) 
        self.stepcounter += 1
        obs = self._get_obs()
        distance = np.linalg.norm(self.target - self.data.site("gripperframe").xpos)
        reward = -(distance)
        terminated = distance < 0.02 
        truncated = self.stepcounter >= 100
        info = {}
        return obs, reward, terminated, truncated, info

    
if __name__ == "__main__":         
    env = SO101ReachEnv()
    print(env.action_space, env.observation_space)      