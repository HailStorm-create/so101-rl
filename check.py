from gymnasium.utils.env_checker import check_env
from so101_env import SO101ReachEnv

check_env(SO101ReachEnv())
print("check passed")