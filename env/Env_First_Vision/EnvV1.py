import gym
from Ball import Ball
from Boundaries import obstacles
from Goal import Goal
from gym.spaces import Box
import numpy as np
from numpy import linalg
import pybullet as p
import time
from pybullet_utils import bullet_client

class Env_version1(gym.Env):
    def __init__(self):
        # Set initial pybullet environment
        self.init_xyz = [3, 3.5, 0.5]
        bc = bullet_client.BulletClient(connection_mode=p.GUI)
        self.bc = bc
        # self.client = p.connect(p.GUI)

        bc.resetDebugVisualizerCamera(cameraDistance=10, cameraYaw=15, cameraPitch=-80,
                                     cameraTargetPosition=[0.55, -0.35, 0.2])

        # Define the outer and inner constriants of the car
        self.length = 7
        self.width = self.length
        self.innerlength = 3
        self.innerwidth = self.innerlength

        # Set the min,max and target_radius of defining if the car arrives the final position
        self.target_radius = 10**(-2)

        # Set the properties for action and observation spaces
        self.action_space = Box(low=-2, high=2, shape=(2,), dtype=np.float32)
        self.observation_space = Box(
            low=np.array([-10, -10, 0, -20, -20, -20, -30, -30, -30], dtype=np.float32),
            high=np.array([10, 10, 0, 20, 20, 20, 30, 30, 30], dtype=np.float32))

        self.bc.setGravity(0, 0, -10)
        # Connect the agent from pybullet
        self.plane = self.bc.loadURDF("plane/plane10.urdf")
        self.Ball = Ball(bc)
        self.obstacles = obstacles(bc)
        self.goal = Goal(bc)
        self.goal_position = np.array([-3,-4])

        # Set the radio and set the position of ball and goal
        self.Ball_radios = np.ones((2,))*0.5
        self.agent_position = np.array([3,3.5])

        # self.Outside_boundary = False

    def outside_boundary(self):
        """Here we want to let the agent know if it is outside of the boundary or not"""
        if linalg.norm(self.agent_position) < 3:
            outside_boundary = True
        elif linalg.norm(self.agent_position) > 5:
            outside_boundary = True
        else:
            outside_boundary = False

        return outside_boundary

    # """The parts of setting constraints will be updated later"""
    # def get_constraint_values(self):
    #     # There a a total of 4 constraints
    #     # a lower and upper bound for each dim
    #     # We define all the constraints such that C_i = 0
    #     # _agent_position > 0 + _agent_slack => -_agent_position + _agent_slack < 0
    #     min_constraints = self.agent_slack - self.agent_position
    #     # _agent_position < np.asarray([_width, length]) - agent_slack
    #     # => _agent_position + agent_slack - np.asarray([_width, length]) < 0
    #     max_constraint = self.agent_position + self.agent_slack - np.asarray([self.width, self.length])
    #
    #     return np.concatenate([min_constraints, max_constraint])

    # def get_constraint_number(self):
    #     return 4

    def reward(self,position):
        # ball_ob = self.ball.get_observation()
        if self.outside_boundary():
            reward = -500
        elif linalg.norm(position-self.goal_position) < self.target_radius:
            reward = 100
        else:
            reward = -1
        return reward

    def step(self, action):
        # Defining the step option
        self.Ball.apply_action(action)
        # print(self.Ball.apply_action(action))
        p.stepSimulation()
        Ball_ob = self.Ball.get_observation()
        self.agent_position = Ball_ob[:2]
        # Getting rewards
        reward = self.reward(self.agent_position)

        if linalg.norm(Ball_ob[:2] - self.goal_position) < self.target_radius:
            done = True
        else:
            done = False

        observation = Ball_ob
        return observation[:3], reward, done, {}

    def reset(self):
        # Reset environement
        # p.setGravity(0, 0, -10)
        self.Ball.reset()

        # Reset the ball agent
        Ball_ob = self.Ball.get_observation()

        # Return the distance of ball and goal
        reach_goal = linalg.norm(Ball_ob[:2] - self.goal_position)

        return Ball_ob,reach_goal

    def render(self, mode='human'):
        width = 300
        height = 10
        if mode == 'human':
            time.sleep(1/60)
            return None
        elif mode == 'rgb_array':
            # Set view,proj matrix, which mainly copy from
            # https://github.com/SvenGronauer/Bullet-Safety-Gym/blob/master/bullet_safety_gym/envs/builder.py
            view_matrix = p.computeViewMatrixFromYawPitchRoll(
                cameraTargetPosition=[10,10,10],
                distance=10,
                yaw=20,
                pitch=20,
                roll=20,
                upAxisIndex=2
            )

            proj_matrix = p.computeProjectionMatrixFOV(
                fov=80,
                aspect=width/height,
                nearVal=0.01,
                farVal=100
            )
            # print("matrixes are")
            # print(proj_matrix)
            # print(view_matrix)

            (_, _, px, _, _) = p.getCameraImage(
                width=width,
                height=height,
                viewMatrix=view_matrix,
                projectionMatrix=proj_matrix,
                renderer=p.ER_BULLET_HARDWARE_OPENGL
            )

            new_shape = (height,width,-1)
            rgb_array = np.reshape(np.array(px), new_shape)
            rgb_array = rgb_array[:, :, :3]
            return rgb_array

    def close(self):
        self.bc.disconnect()