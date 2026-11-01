import numpy as np 
import mujoco
import gymnasium

class SO101PushEnv(gymnasium.Env): 

    """
    created init environment 
    creating env type + arm load + live state + RL algorithm # controls
    """

    def __init__(self): 
        self.model = mujoco.MjModel.from_xml_path("so101/scene_push.xml")
        self.data = mujoco.MjData(self.model)
        self.action_space = gymnasium.spaces.Box(low = -1, high = 1, shape = (5,), dtype=np.float32)
        self.observation_space = gymnasium.spaces.Box(low = -np.inf , high = np.inf, shape = (14,), dtype = np.float32)

    def reset(self, seed=None, options=None): 
        super().reset(seed = seed)
        mujoco.mj_resetData(self.model, self.data)
        self.goal = self.np_random.uniform(low=[0.15,-0.15,0], high=[0.30,.15,0]) 
        self.cup = self.np_random.uniform(low=[0.15, -0.15, 0.04], high=[0.30, .15, .04])
        self.data.mocap_pos[0] = self.goal
        self.data.mocap_pos[1] = self.cup
        mujoco.mj_forward(self.model, self.data)
        obs = self._get_obs()
        self.stepcounter = 0
        return obs, {}

    def _get_obs(self): 
        sliced = self.data.qpos[:5]
        gripper_pos = self.data.site("gripperframe").xpos
        obs = np.concatenate([sliced, gripper_pos, self.goal, self.cup])
        return obs.astype(np.float32)
                
    def step(self, action): 
        self.data.ctrl[:5] = np.clip(self.data.ctrl[:5] + action * 0.1 , self.model.actuator_ctrlrange[:5, 0], self.model.actuator_ctrlrange[:5, 1], out=None)
        for i in range(10): mujoco.mj_step(self.model, self.data) 
        self.stepcounter += 1
        obs = self._get_obs()
        distance = np.linalg.norm(self.goal - self.data.site("gripperframe").xpos)
        reward = -(distance)
        terminated = distance < 0.02 
        truncated = self.stepcounter >= 100
        info = {}
        return obs, reward, terminated, truncated, info

    
if __name__ == "__main__":         
    env = SO101PushEnv()
    print(env.action_space, env.observation_space)      