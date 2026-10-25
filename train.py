from so101_env import SO101ReachEnv
from stable_baselines3 import PPO

env = SO101ReachEnv()

model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=20_000)
model.save("ppo_reach_v1")

