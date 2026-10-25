from so101_env import SO101ReachEnv
from stable_baselines3 import PPO
import mujoco.viewer
import time

env = SO101ReachEnv()
model = PPO.load("ppo_reach_v1.zip")

obs, info = env.reset()
with mujoco.viewer.launch_passive(env.model, env.data ) as viewer:
    while viewer.is_running(): 
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            obs, info = env.reset()
        viewer.sync()
        time.sleep(0.02)