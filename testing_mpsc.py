import numpy as np
import torch
# from env.Envsimple_third import Env_simple
from ppo.layerclass import Actor
from env.EnvV3 import Env_version3

# 这个部分属于MPSC所需要的数据库
# from mpsc.mpsc_com import MPSC
from mpsc.mpc_unit import *
import matplotlib.pyplot as plt
from mpsc.mpsc_sim_ver2 import MPSC_SIM
from mpsc.mpsc_com2 import MPSC
#测试强化学习部分
from ppo.PPO import PPO

# env = Env_simple()
env = Env_version3()

# Testing Environment
# while True:
#     env.render()
#     action = env.action_space.sample()
#     observation, reward, done,_ = env.step(action)
#     print(observation)

# RL train for easy environment
# def train(env):
# 	# Create a model for PPO.
# 	model = PPO(env=env)
# 	model.learn(200,600)
#
# model = train(env)
# obs = env.reset()
# rl_model = Actor(env.observation_space.shape[0],env.action_space.shape[0])
# rl_model.load_state_dict(torch.load("ppo_actor.pth"))

# Testing Pure RL
# obs = env.reset()
# for i in range(500):
#     env.render()
#     act = rl_model(obs)
#     obs, reward, done, ref = env.step(act)

# Testing easy MPSC
# model = MPSC_SIM(env,[0.8],[1],10)
# results_dict = model.run(rl_model)
# print(results_dict["actions"])
# print(results_dict["feasible"])
# plt.plot(results_dict["actions"])
# plt.show()

# Testing complex MPSC
model = MPSC(env,[2],[0.8],10)
results_dict = model.run(np.array([1,0]))
# print(results_dict["obs"])
print(results_dict["actions"])
print(results_dict["feasible"])
print(results_dict["obs"])
plt.plot(results_dict["actions"])
plt.show()




