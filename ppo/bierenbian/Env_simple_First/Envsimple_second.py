import gym
from env.Ballsimple import Ball
import math
from gym.spaces import Box
import numpy as np
from numpy import linalg
import pybullet as p
import time
from pybullet_utils import bullet_client

class Env_simple(gym.Env):
    def __init__(self):
        # Set initial pybullet environment
        target_xyz = [0, 0, 0]
        init_xyz = [-3,0.,0.5]
        self.init_xyz = init_xyz
        self.target_position = target_xyz

        self.aggregate_phy_steps = 1
        # self.iteration = 5

        bc = bullet_client.BulletClient(connection_mode=p.GUI)
        self.bc = bc
        # self.client = p.connect(p.GUI)

        bc.resetDebugVisualizerCamera(cameraDistance=10, cameraYaw=15, cameraPitch=-80,
                                     cameraTargetPosition=[0.55, -0.35, 0.2])


        # Set the properties for action and observation spaces
        self.action_space = Box(low=-4, high=4, shape=(1,), dtype=np.float32)
        self.observation_space = Box(
            low=np.array([-20, -180], dtype=np.float32),
            high=np.array([20, 180], dtype=np.float32))

        # Connect the agent from pybullet
        self.bc.setGravity(0, 0, -10)
        # ospath =
        self.plane = self.bc.loadURDF("env/plane/plane100.urdf")
        self.Ball = Ball(bc,self.init_xyz)
        self.reference_trajectory = self._setup_task_specifics()

        # Set the radio and set the position of ball and goal
        self.Ball_radios = np.ones((2,))*0.5

    def _setup_task_specifics(self):
        """Initialize task specifics. Called by _setup_simulation()."""
        # print(f'Spawn target pos at:', self.target_pos)
        target_visual = self.bc.createVisualShape(
            self.bc.GEOM_BOX,
            halfExtents = [8,1,0.1],
            rgbaColor=[0.5, 0.5, 0.6, 0.6],
        )
        # Spawn visual without collision shape
        self.target_body_id = self.bc.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=-1,
            baseVisualShapeIndex=target_visual,
            basePosition = self.target_position
        )

    def outside_boundary(self):
        """Here we want to let the agent know if it is outside of the boundary or not"""
        if self.agent_position[0] - 0.5 < -8:
            outside_boundary = True
        elif self.agent_position[0] + 0.5 > 8:
            outside_boundary = True
        else:
            outside_boundary = False

        return outside_boundary

    def compute_done(self):
        """Compute end of episode if dist(drone - ref) > d."""
        dist = np.linalg.norm(self.Ball.get_observation()[0] - 0)
        if dist < 0.5:
            done = True
        elif self.outside_boundary():
            done = True
        else:
            done = False
        return done

    def get_constraint(self):
        X_state_constraint = [lambda x: x[0]-7.5, lambda x: -7.5-x[0]]
        return X_state_constraint

    def get_constraint_values(self):
        # _agent_position > 0.5 + _agent_slack => -_agent_position + _agent_slack < 0
        min_constraints = self.agent_position[0] - 0.5 - 8
        # _agent_position < 1 - _agent_slack => _agent_position + agent_slack- 1 < 0
        max_constraint = self.agent_position[0] + 0.5 + 8
        return np.concatenate([min_constraints,max_constraint])


    def reward(self):
        # ball_ob = self.ball.get_observation()
        if self.outside_boundary():
            reward_outside = -500
        else:
            reward_outside = -1
        # L2 norm
        dist = np.linalg.norm(self.Ball.get_position()[0] - 0)
        reward = -dist + reward_outside # + all_consider
        return reward


    def step(self, action):
        self.last_action = self.Ball.apply_action(action)
        p.stepSimulation()
        Ball_ob = self.Ball.get_observation()
        self.agent_position = Ball_ob[:3]
        # Getting rewards
        reward = self.reward()
        # Compute done
        done = self.compute_done()
        observation = Ball_ob
        return observation[[0,3]], reward, done, {}

    def reset(self):
        # Reset environement
        p.setGravity(0, 0, -10)
        self.Ball.reset()

        # Reset the ball agent
        Ball_ob = self.Ball.get_observation()

        return Ball_ob[[0,3]]

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