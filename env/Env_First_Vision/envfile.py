import gym
from gym.spaces import Box,Dict
import numpy as np
from numpy import linalg

class Car(gym.Env):
    def __init__(self,length,inner_length):
        # Set the inital timing
        self.current_time = 0
        self.frequency_radio = 1
        # self.velocity = 0
        # Define the outer constriants of the car
        self.length = length
        self.width = self.length
        # Define the inner constraints of the car
        self.innerlength = inner_length
        self.innerwidth = self.innerlength

        # Set the min,max and target_radius of defining if the car arrives the final position
        self.target_radius = 10**(-2)
        self.min = -self.length
        self.max = self.length

        # Define the start postion of the agent and the target
        self.agent_position = np.array((1, 2))
        self.target_position = np.array((4, 2))

        # Set the properties for spaces
        self.action_space = Box(low=-2, high=2, shape=(2,), dtype=np.float32)
        self.observation_space = Box(low=self.min, high=self.max, shape=(2,), dtype=np.float32)
        # self.observation_space = Dict({
        #     'agent_position': Box(low=self.min, high=self.max, shape=(2,), dtype=np.float32),
        #     'agent_velocity': Box(low=-200, high=200, shape=(2,), dtype=np.float32),
        #     'target_position': Box(low=self.min, high=self.max, shape=(2,), dtype=np.float32)
        # })

        # self.viewer = None
        # self.state = None

    def outside_boundary(self):
        # Here we want to let the agent know if it is outside of the boundary or not
        # If the agent postition is smaller than 0 or it is outside of length and width, then it is outside
        # If the agent position is inside the innerboundary, then it is also outside
        # else it isn't then it is false
        if  np.any(self.agent_position < 0) or np.any(self.agent_position > np.array([self.length, self.width])):
            outside_boundary = True
        elif np.all(self.agent_position > np.array([self.innerlength,self.innerwidth])) and np.all(self.agent_position < np.array([1.5*self.innerlength,1.5*self.innerwidth])):
            outside_boundary = True
        else:
            outside_boundary = False
        return outside_boundary


    def get_constraint_values(self):
        # There a a total of 4 constraints
        # a lower and upper bound for each dim
        # We define all the constraints such that C_i = 0
        # _agent_position > 0 + _agent_slack => -_agent_position + _agent_slack < 0
        min_constraints = self.agent_slack - self.agent_position
        # _agent_position < np.asarray([_width, length]) - agent_slack
        # => _agent_position + agent_slack - np.asarray([_width, length]) < 0
        max_constraint = self.agent_position + self.agent_slack - np.asarray([self.width, self.length])

        return np.concatenate([min_constraints, max_constraint])

    def get_constraint_number(self):
        return 4

    def _update_time(self):
        # Assume that frequency of motor is 1 (one action per second)
        self.current_time += self.frequency_ratio

    def reward(self):
        if self.outside_boundary():
            reward = -500
        elif linalg.norm(self.agent_position-self.target_position) < self.target_radius:
            reward = 100
        else:
            reward = -1
        return reward

    def step(self, action):
        self.velocity += self.frequency_radio * action
        self.agent_position += self.velocity * self.frequency_radio + 0.5 * action * self.frequency_radio ** 2
        observation = self.agent_position
        # self.velocity += self.frequency_radio * action
        rewards = self.reward()
        # observation = {
        #     "agent_position": self.agent_position,
        #     "agent_velocity": self.velocity,
        #     "target_postion": self.target_position
        # }

        if linalg.norm(self.agent_position-self.target_position) < self.target_radius:
            done = True
        else:
            done = False

        return observation,rewards,done,{}

    def reset(self):
        self.velocity = np.zeros(2, dtype=np.float32)
        self.agent_position = np.array((1, 2), dtype=np.float32)
        self.target_position = np.array((4, 2),dtype=np.float32)
        self.current_time = 0
        return self.velocity, self.agent_position, self.target_position, self.current_time

    # def render(self, mode="human"):
    #     screen_width = 600
    #     screen_height = 400
    #
    #     world_width = self.max * 2
    #     scale = screen_width / world_width
    #     carty = 100  # TOP OF CART
    #     polewidth = 10.0
    #     polelen = scale * (2 * self.length)
    #     cartwidth = 50.0
    #     cartheight = 30.0
    #
    #     if self.viewer is None:
    #         from gym.envs.classic_control import rendering
    #
    #         self.viewer = rendering.Viewer(screen_width, screen_height)
    #         l, r, t, b = -cartwidth / 2, cartwidth / 2, cartheight / 2, -cartheight / 2
    #         axleoffset = cartheight / 4.0
    #         cart = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
    #         self.carttrans = rendering.Transform()
    #         cart.add_attr(self.carttrans)
    #         self.viewer.add_geom(cart)
    #         l, r, t, b = (
    #             -polewidth / 2,
    #             polewidth / 2,
    #             polelen - polewidth / 2,
    #             -polewidth / 2,
    #         )
    #         pole = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
    #         pole.set_color(0.8, 0.6, 0.4)
    #         self.poletrans = rendering.Transform(translation=(0, axleoffset))
    #         pole.add_attr(self.poletrans)
    #         pole.add_attr(self.carttrans)
    #         self.viewer.add_geom(pole)
    #         self.axle = rendering.make_circle(polewidth / 2)
    #         self.axle.add_attr(self.poletrans)
    #         self.axle.add_attr(self.carttrans)
    #         self.axle.set_color(0.5, 0.5, 0.8)
    #         self.viewer.add_geom(self.axle)
    #         self.track = rendering.Line((0, carty), (screen_width, carty))
    #         self.track.set_color(0, 0, 0)
    #         self.viewer.add_geom(self.track)
    #
    #         self._pole_geom = pole
    #
    #     if self.state is None:
    #         return None
    #
    #     # Edit the pole polygon vertex
    #     pole = self._pole_geom
    #     l, r, t, b = (
    #         -polewidth / 2,
    #         polewidth / 2,
    #         polelen - polewidth / 2,
    #         -polewidth / 2,
    #     )
    #     pole.v = [(l, b), (l, t), (r, t), (r, b)]
    #
    #     x = self.state
    #     cartx = x[0] * scale + screen_width / 2.0  # MIDDLE OF CART
    #     self.carttrans.set_translation(cartx, carty)
    #     self.poletrans.set_rotation(-x[2])
    #
    #     return self.viewer.render(return_rgb_array=mode == "rgb_array")
    #
    # def close(self):
    #     if self.viewer:
    #         self.viewer.close()
    #         self.viewer = None