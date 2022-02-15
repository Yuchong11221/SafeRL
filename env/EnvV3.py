import gym
from env.Ball import Ball
from env.Boundaries import obstacles
import math
from gym.spaces import Box
import numpy as np
from numpy import linalg
import pybullet as p
import time
from pybullet_utils import bullet_client

class Env_version3(gym.Env):
    def __init__(self):
        # Set initial pybullet environment
        init_xyz = [0, 0,0.5]
        init_rpy = np.zeros(3)
        init_xyz_dot = np.zeros(3)
        init_rpy_dot = np.zeros(3)

        self.init_xyz = init_xyz
        self.target_position = self.init_xyz
        self.done_dist_threshold = 0.5

        self.aggregate_phy_steps = 1
        self.iteration = 5

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

        # Set the properties for action and observation spaces
        self.action_space = Box(low=-4, high=4, shape=(2,), dtype=np.float32)
        self.observation_space = Box(
            low=np.array([-20, -20,180,180], dtype=np.float32),
            high=np.array([20, 20,-180,-180], dtype=np.float32))

        # Connect the agent from pybullet
        self.bc.setGravity(0, 0, -10)
        self.bc.setTimeStep(0.01)
        self.plane = self.bc.loadURDF("env/plane/plane100.urdf")
        self.Ball = Ball(bc,self.init_xyz)
        self.obstacles = obstacles(bc)
        self.reference_ball = self._setup_task_specifics()

        # Set the radio and set the position of ball and goal
        self.Ball_radios = np.ones((2,))*0.5
        self.agent_position = self.target_position

        # Set the reference spaces and step
        self.t = 0
        observation_frequency = 100
        N = 301
        self.N = N
        circle_time = 3
        circle_radius = 5
        episode_time = N / observation_frequency  # [s]
        ts = np.arange(N) / N * 2 * np.pi * episode_time / circle_time

        self.ref = np.zeros((N,3))
        self.ref[:,2] = 0.5 # z-postion is always equal to the radius of the ball
        self.ref[:,1] = circle_radius * np.sin(ts) # y-position
        self.ref[:,0] = circle_radius * (1-np.cos(ts)) # x-position

    def _setup_task_specifics(self):
        """Initialize task specifics. Called by _setup_simulation()."""
        # print(f'Spawn target pos at:', self.target_pos)
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
            basePosition = self.target_position
        )

    def outside_boundary(self):
        """Here we want to let the agent know if it is outside of the boundary or not"""
        if linalg.norm(self.agent_position[:2] - np.array([4,0])) < 4:
            outside_boundary = True
        elif linalg.norm(self.agent_position[:2] - np.array([4,0])) > 6:
            outside_boundary = True
        else:
            outside_boundary = False

        return outside_boundary

    def compute_done(self):
        """Compute end of episode if dist(drone - ref) > d."""
        dist = np.linalg.norm(self.Ball.get_observation()[:3] - self.target_position)
        if dist < self.done_dist_threshold or self.outside_boundary():
            done = True
        else:
            done = False

        return done

    def num_constraint(self):
        return 2

    def get_constraint_value(self, pos):
        # Here the constraints are devided into two part for the simple environment
        # One is the left one, and the other is the right one
        inner_constraints = linalg.norm(self.agent_position[:2] - np.array([3,0])) - 4
        outter_constraint = -6 + linalg.norm(self.agent_position[:2] - np.array([3,0]))
        return np.array([inner_constraints, outter_constraint])

    def get_constraint(self):
        X_state_constraint = [lambda x: (x[0]-4)**2 -36,
                              lambda x: 16-(x[0]-4)**2,
                              lambda x: x[1]**2 - 36,
                              lambda x: 16 - x[1]**2,
                              ]
        # X_state_constraint = [lambda x: (x[:3] - np.array([4, 0, 0.5])) ** 2 - 36]
        return X_state_constraint

    def reward(self):
        # ball_ob = self.ball.get_observation()
        if self.outside_boundary():
            reward_outside = -500
        else:
            reward_outside = 0

        dist = np.linalg.norm(self.Ball.get_position() - self.target_position)
        reward = -dist + reward_outside
        return reward


    def step(self, action):
        # Defining the step option
        if self.t > self.N-1:
            self.t = self.t - math.floor(self.t/self.N)* self.N

        self.target_position = self.ref[self.t]
        self.bc.resetBasePositionAndOrientation(
            self.target_body_id,
            posObj=self.target_position,
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
        self.t = self.t+1
        # if self.t > self.N:
        #     self.t = self.t - math.floor(self.t/self.N)* self.N
        return observation[[0,1,3,4]], reward, done, np.concatenate((self.target_position[[0,1]],np.zeros(2)),axis=0)

    def reset(self,random_reset=False):
        # Reset environement
        p.setGravity(0, 0, -10)
        self.Ball.reset(random_reset)

        # Reset the ball agent
        Ball_ob = self.Ball.get_observation()
        # print(Ball_ob)
        output = Ball_ob[[0,1,3,4]]

        return output

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