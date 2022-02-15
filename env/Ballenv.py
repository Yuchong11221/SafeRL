import pybullet as pb
import numpy as np

class Ball():
    """ A spherical agent that moves on the (x,y)-plane.
    The ball is moved via external forces that are applied in world coordinates.
    Observations are in R^7 and actions are in R^2.
    """
    def __init__(
            self,
            bc,
            init_xyz=(0, 0, .5),  # ball diameter is 0.5
            debug=False
    ):
        super().__init__(
            'ballagent/ball.urdf',
            act_dim=2,
            obs_dim=7,
            init_xyz=init_xyz,
            fixed_base=False,
            global_scaling=1,
            collision_radius=0.5,  # ball has 0.5 diameter
            self_collision=False,
            velocity_constraint=2.5,
            max_force=3.5,
            max_velocity=0,  # irrelevant parameter (external force controlled)
            debug=debug
        )
        self.radius = 0.5
        self.size_violation_shape = self.global_scaling * 1.25 * self.radius
        self.last_taken_action = np.zeros(self.act_dim)

    def add_sensor(self, sensor):
        """ A sensor is added to the agent by a task.
            e.g. the goal reach tasks adds a sensor to detect obstacles.
        """
        # Avoid rotation of sensor with ball agent
        sensor.rotate_with_agent = False
        super().add_sensor(sensor)

    @property
    def alive(self) -> bool:
        """Returns "False" if the agent died, "True" otherwise.
        Ball agent has no termination criterion."""
        return True

    def apply_action(self, action):
        # check validity of action and clip into range [-1, +1]
        self.last_taken_action = super().apply_action(action)
        x, y = self.last_taken_action
        sphere_pos = self.get_position()
        # over-write actions with keyboard inputs
        if self.debug:
            keys = self.getKeyboardEvents()
            x = 0
            y = 0
            for k, v in keys.items():
                if k == pb.B3G_UP_ARROW and (v & pb.KEY_IS_DOWN):
                    y += 1
                if k == pb.B3G_LEFT_ARROW and (v & pb.KEY_IS_DOWN):
                    x += -1
                if k == pb.B3G_DOWN_ARROW and (v & pb.KEY_IS_DOWN):
                    y += -1
                if k == pb.B3G_RIGHT_ARROW and (v & pb.KEY_IS_DOWN):
                    x += 1

        self.apply_external_force(
            force=np.array([x, y, 0.]) * self.max_force,
            link_id=-1,
            position=sphere_pos,
            frame=pb.WORLD_FRAME
        )

    def get_linear_velocity(self) -> np.ndarray:
        """ over-write Agent class since Ball owns only one body."""
        return self.get_state()[3:6]

    def get_angular_velocity(self) -> np.ndarray:
        """Returns this link's rotational speed in [rad/s]."""
        return self.get_state()[9:12]

    def get_state(self) -> np.ndarray:
        """Returns the 2-dim vector (position, velocity) of this joint."""
        x, vx, *_ = self.bc.getJointState(
            self.body_id,
            self.index)
        return np.array([x, vx])

    def agent_specific_observation(self) -> np.ndarray:
        """ State of ball is of shape (7,) """
        xyz = 0.1 * self.get_position()
        xyz_dot = 0.2 * self.get_linear_velocity()
        rpy_dot = 0.1 * self.get_angular_velocity()
        obs = np.concatenate((xyz[:2], xyz_dot[:2], rpy_dot))
        return obs

    def get_orientation(self) -> np.ndarray:
        """ over-write Agent class since Ball owns only one body."""
        return self.get_state()[6:9]

    def get_position(self) -> np.ndarray:
        """ over-write Agent class since Ball owns only one body."""
        return self.get_state()[:3]

    def get_quaternion(self):
        xyz, abcd = self.bc.getBasePositionAndOrientation(self.body_id)
        return abcd

    def specific_reset(self):
        """ Reset only agent specifics such as motor joints. Do not set position
            or orientation since this is handled by task.specific_reset()."""
        self.set_position(self.init_xyz)

    def specific_reward(self) -> float:
        """ Some agents exhibit additional rewards besides the task objective,
            e.g. electricity costs, speed costs, etc. """
        return -0.5 * np.linalg.norm(self.last_taken_action)
