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
        set_xyz = [0, 0, 0]
        init_xyz = [-3,0.,0.5]
        target_xyz = [-2,0,0.5]
        self.init_xyz = init_xyz
        self.set_position = set_xyz
        self.target_position = target_xyz

        self.aggregate_phy_steps = 1
        # self.iteration = 5

        bc = bullet_client.BulletClient(connection_mode=p.GUI)
        self.bc = bc

        bc.resetDebugVisualizerCamera(cameraDistance=10, cameraYaw=15, cameraPitch=-80,
                                     cameraTargetPosition=[0.55, -0.35, 0.2])


        # Set the properties for action and observation spaces
        self.action_space = Box(low=-4, high=4, shape=(1,), dtype=np.float32)
        self.observation_space = Box(
            low=np.array([-20, -180], dtype=np.float32),
            high=np.array([20, 180], dtype=np.float32))

        # Connect the agent from pybullet
        self.bc.setGravity(0, 0, -10)
        self.bc.setTimeStep(0.01)
        self.plane = self.bc.loadURDF("./plane/plane100.urdf")
        self.Ball = Ball(bc,self.init_xyz)
        self.reference_trajectory = self._setup_task_specifics()

        # Set the radio and set the position of ball and goal
        self.Ball_radios = np.ones((2,))*0.5

        # Setting the reference point
        N = 400
        self.N = N
        self.t = 0
        self.ref_point = self.reference(self.N)

    def reference(self,N):
        # Set a simple version of ball moving
        self.ref = np.zeros((3, N))
        self.ref[2,:] = 0.5  # z-postion is always equal to the radius of the ball
        self.ref[1,:] = 0  # y-position
        self.ref[0, :] = np.linspace(self.init_xyz[0],7.5,num=N)  # x-position
        return self.ref

    def _setup_task_specifics(self):
        """Initialize task specifics. Called by _setup_simulation()."""
        # print(f'Spawn target pos at:', self.target_pos)
        set_visual = self.bc.createVisualShape(
            self.bc.GEOM_BOX,
            halfExtents = [8,1,0.1],
            rgbaColor=[0.5, 0.5, 0.6, 0.6],
        )
        # Spawn visual without collision shape
        self.set_body_id = self.bc.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=-1,
            baseVisualShapeIndex=set_visual,
            basePosition = self.set_position
        )

        target_visual = self.bc.createVisualShape(
            self.bc.GEOM_SPHERE,
            radius=0.5,
            rgbaColor=[0.01, 0.01, 0.01, 0.8],
        )
        # Spawn visual without collision shape
        self.target_body_id = self.bc.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=-1,
            baseVisualShapeIndex=target_visual,
            basePosition=self.target_position
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
        #  For this part it is design for safety layer and I copy this part from the github and adjust to my own version
        # https://github.com/AgrawalAmey/safe-explorer/blob/master/safe_explorer/env/spaceship.py
        #  And also what is agent_slack?
        # _agent_position > 0.5 + _agent_slack => -_agent_position + _agent_slack < 0
        min_constraints = self.agent_position[0] - 0.5 - 8
        # _agent_position < 1 - _agent_slack => _agent_position + agent_slack- 1 < 0
        max_constraint = self.agent_position[0] + 0.5 + 8
        return np.concatenate([min_constraints,max_constraint])


    def reward(self):
        if self.outside_boundary():
            reward_outside = -500
        else:
            reward_outside = -0
        # L2 norm
        dist = np.linalg.norm(self.Ball.get_position()[0] - 7.5)
        reward = -dist + reward_outside # + all_consider
        return reward


    def step(self, action):
        # self.target_position = self.ref_point[:,self.t]
        # Here I want to make the ball move forward if it doesnt arrive the 400 step and backward after 400 step
        # should be updated later
        # if self.t > self.N-1:
        #     self.t = 0
        # else:
        #     self.t = self.t+1
        self.bc.resetBasePositionAndOrientation(
            self.target_body_id,
            posObj=self.target_position[:3],
            ornObj=(0, 0, 0, 1)
        )
        self.last_action = self.Ball.apply_action(action)
        p.stepSimulation()
        Ball_ob = self.Ball.get_observation()
        self.agent_position = Ball_ob[:3]
        # Getting rewards
        reward = self.reward()
        # Compute done
        done = self.compute_done()
        observation = Ball_ob
        return observation[[0,3]], reward, done, self.target_position[:2]

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