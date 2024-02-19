from stable_baselines3.common.env_checker import check_env
from PdeVecEnv import PdeVecEnv

env = PdeVecEnv()
check_env(env, warn=True)