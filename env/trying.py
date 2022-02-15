import pybullet as p
import gym
from EnvV2 import Env_version2
from Boundaries import obstacles
from Ball import Ball
# from Goal import Goal
import numpy as np
from pybullet_utils import bullet_client
from time import sleep
import math

"""Testing the building environment"""
env = Env_version2()
# # print(env.step([0,0]))
env.reset()
# #
# for i in range(1000):
while True:
    env.render()
    # if i > 500:
    #     i = i - math.floor(i/500)*500
    # print(i)
    action = env.action_space.sample()
    # A,B,constraint = env.get_constraint()
    # print(A,B,constraint)
    observation, reward, done,_ = env.step(action)
#     # print(observation)
    # print(reward)

# """Here is to test if my model can be print out in GUI"""
# bc = bullet_client.BulletClient(connection_mode=p.GUI)
# # client = p.connect(p.GUI)
# goal =Ball(bc)
# print(goal.get_observation())
# sleep(10)
# # obstacles = obstacles(client)
# ball_agent = Ball(client)
# while True:
#     ball_agent.apply_action([10,10])
#     p.stepSimulation()
#     Ball_ob = ball_agent.get_observation()
# k = ball_agent.get_observation()
# print(k)
# ball_agent.reset()
# ball_agent.get_observation()
# p.getJointInfo(ball_agent,)
# print("agent is")




