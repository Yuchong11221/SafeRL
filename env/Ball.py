import pybullet as p
import numpy as np

class Ball():
    def __init__(self,bc,init_xyz = False,debug=False):
        super(Ball, self).__init__()
        # Set the inital situations
        if init_xyz is False:
            init_xyz = [0, 0, 0.5]
        else:
            init_xyz = init_xyz
        max_force = 150
        self.init_xyz = init_xyz
        self.debug = debug
        self.max_force = max_force
        self.bc = bc
        self.ball = bc.loadURDF("env/ballagent/ball.urdf",basePosition = init_xyz)
        self.radius = 0.5
        self.last_taken_action = np.zeros((2))

        # act_dim = 2
        # obs_dim=7

    def get_ids(self):
        return self.ball, self.client

    def apply_action(self,action):
        self.last_taken_action = action
        x_force = self.last_taken_action[0]
        y_force = self.last_taken_action[1]
        position_now = self.get_position()
        # over-write actions with keyboard inputs
        if self.debug:
            keys = self.bc.getKeyboardEvents()
            x_force = 0
            y_force = 0
            for k, v in keys.items():
                if k == p.B3G_UP_ARROW and (v & p.KEY_IS_DOWN):
                    y_force += 1
                if k == p.B3G_LEFT_ARROW and (v & p.KEY_IS_DOWN):
                    x_force += -1
                if k == p.B3G_DOWN_ARROW and (v & p.KEY_IS_DOWN):
                    y_force += -1
                if k == p.B3G_RIGHT_ARROW and (v & p.KEY_IS_DOWN):
                    x_force += 1
        self.bc.applyExternalForce(
            self.ball,
            -1,
            np.array([x_force, y_force, 0.]) * self.max_force,
            position_now,
            p.WORLD_FRAME
        )
        # self.last_taken_action = self.apply_action(action)

    def get_observation(self):
        """Returns position and velocity of the base link."""
        xyz, ori = self.bc.getBasePositionAndOrientation(self.ball)
        # xyz = 0.2 * np.asarray(xyz)
        # print(xyz)

        xyz_dot, rpy_dot = self.bc.getBaseVelocity(self.ball)
        # xyz_dot = 0.1 * np.asarray(xyz_dot)
        # rpy_dot = 0.2 * np.asarray(rpy_dot)

        # Quaternion (a, b, c, d) to Euler (roll pitch yaw)
        rpy = p.getEulerFromQuaternion(ori)
        observation = np.concatenate([xyz, xyz_dot, rpy, rpy_dot])
        return observation

    def get_position(self):
        return self.get_observation()[:3]

    def get_xyz_dot(self):
        return self.get_observation()[3:7]

    def get_rpy(self):
        return self.get_observation()[7:10]

    def get_rpy_dot(self):
        return self.get_observation()[10:]

    def reset(self, random_reset=False):
        N = np.random.randint(1,301)
        circle_radius = 5
        init = self.init_xyz
        if random_reset:
            init[2] = 0.5  # z-postion is always equal to the radius of the ball
            init[1] = circle_radius * np.sin(N)  # y-position
            init[0] = circle_radius * (1 - np.cos(N))  # x-position
        return self.bc.resetBasePositionAndOrientation(self.ball, init,[0,0,0,1])
