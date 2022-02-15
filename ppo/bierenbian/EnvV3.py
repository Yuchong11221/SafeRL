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
        init_xyz = [0, 0,2]
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
        self.action_space = Box(low=-2, high=2, shape=(3,), dtype=np.float32)
        self.observation_space = Box(
            low=np.array([-20, -20, -20,180,180,180], dtype=np.float32),
            high=np.array([20, 20, 20,-180,-180,-180], dtype=np.float32))

        # Connect the agent from pybullet
        self.bc.setGravity(0, 0, -10)
        # ospath =
        self.plane = self.bc.loadURDF("env/plane/plane100.urdf")
        self.Ball = Ball(bc)
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
        if linalg.norm(self.agent_position - np.array([4,0,0.5])) < 4:
            outside_boundary = True
        elif linalg.norm(self.agent_position - np.array([4,0,0.5])) > 6:
            outside_boundary = True
        else:
            outside_boundary = False

        return outside_boundary

    def compute_done(self):
        """Compute end of episode if dist(drone - ref) > d."""
        dist = np.linalg.norm(self.Ball.get_observation()[:3] - self.target_position)
        done = True if dist > self.done_dist_threshold else False
        return done

    def get_constraint(self):
        X_state_constraint = [lambda x: (x[:3] - np.array([4,0,0.5]))**2-36,
                             lambda x: (x[:3] - np.array([4,0,0.5]))**2-30]
        # X_state_constraint = [lambda x: (x[:3] - np.array([4, 0, 0.5])) ** 2 - 36]
        return X_state_constraint

    def reward(self):
        # ball_ob = self.ball.get_observation()
        if self.outside_boundary():
            reward_outside = -500
        else:
            reward_outside = 0
        # Determine penalties
        # spin_penalty = 1e-4 * np.linalg.norm(self.drone.rpy_dot)**2
        # print(action)
        # print(self.last_action)
        # act_diff = action - self.Ball.last_taken_action
        # ball_action_rate = linalg.norm(act_diff)

        # ball_rpy = linalg.norm(self.Ball.get_rpy())
        # ball_spin = linalg.norm(self.Ball.get_rpy_dot())
        # ball_velocity = linalg.norm(self.Ball.get_xyz_dot())

        # all_consider = np.sum([ball_action_rate,ball_rpy,ball_spin,ball_velocity])
        # L2 norm
        dist = np.linalg.norm(self.Ball.get_position() - self.target_position)
        reward = -dist + reward_outside # + all_consider
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
        return observation[:6], reward, done, {}

    def reset(self):
        # Reset environement
        p.setGravity(0, 0, -10)
        self.Ball.reset()

        # Reset the ball agent
        Ball_ob = self.Ball.get_observation()
        self.t = 0
        self.target_position = self.init_xyz
        # Return the distance of ball and goal
        dis_reach_goal = linalg.norm(Ball_ob[:3] - self.target_position)

        return Ball_ob[:6]

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