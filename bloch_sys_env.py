import math
from typing import Optional
import numpy as np

import gym
from gym import spaces
from gym.envs.classic_control import utils
from gym.error import DependencyNotInstalled


class Continuous_BlochSphere(gym.Env):
    """
    Custom Environment to find a target position on a Bloch sphere. The position is represented using two angles, phi and theta.

    ### Observation Space

    The observation is a `ndarray` with shape `(2,)` where the elements correspond to the following:

    | Num | Observation                          | Min  | Max | Unit         |
    |-----|--------------------------------------|------|-----|--------------|
    | 0   | position of the car along the x-axis | -Inf | Inf | position (m) |
    | 1   | velocity of the car                  | -Inf | Inf | position (m) |

    ### Action Space

    The action is a `ndarray` with shape `(1,)`, representing the directional force applied on the car.
    The action is clipped in the range `[-1,1]` and multiplied by a power of 0.0015.

    ### Transition Dynamics:

    Given an action, the mountain car follows the following transition dynamics:

    *velocity<sub>t+1</sub> = velocity<sub>t+1</sub> + force * self.power - 0.0025 * cos(3 * position<sub>t</sub>)*

    *position<sub>t+1</sub> = position<sub>t</sub> + velocity<sub>t+1</sub>*

    where force is the action clipped to the range `[-1,1]` and power is a constant 0.0015.
    The collisions at either end are inelastic with the velocity set to 0 upon collision with the wall.
    The position is clipped to the range [-1.2, 0.6] and velocity is clipped to the range [-0.07, 0.07].

    ### Reward

    A negative reward of *-0.1 * action<sup>2</sup>* is received at each timestep to penalise for
    taking actions of large magnitude. If the mountain car reaches the goal then a positive reward of +100
    is added to the negative reward for that timestep.

    ### Starting State

    The position of the car is assigned a uniform random value in `[-0.6 , -0.4]`.
    The starting velocity of the car is always assigned to 0.

    ### Episode End

    The episode ends if either of the following happens:
    1. Termination: The position of the car is greater than or equal to 0.45 (the goal position on top of the right hill)
    2. Truncation: The length of the episode is 999.

    """

    def __init__(self, 
                 delta_angle=0.02*np.pi,
                 target_phi=1,
                 target_theta=0.5, 
                ):

        # Actions: continuous changes in phi and theta
        self.delta_angle = delta_angle
        self.min_action = -1.0
        self.max_action = 1.0
        self.action_space = spaces.Box(
            low=self.min_action, high=self.max_action, shape=(2,), dtype=np.float32
        )

        # State: [phi, theta]    0 < phi < pi, 0 < theta < 2pi
        self.observation_space = spaces.Box(
            low = np.array([0, 0]), 
            high= np.array([np.pi, 2*np.pi]), 
            dtype=np.float32
        )
        
        self.target_phi = target_phi
        self.target_theta = target_theta
        self.threhold = 0.1
        self.max_dist = np.sqrt(np.pi**2 + np.pi**2)


    def angle_difference(self, angle1, angle2):
        # signed minimum difference between the two angles. result in [0, np.pi]
        diff = (angle2 - angle1 + np.pi) % (2*np.pi) - np.pi
        return abs(diff)

    def _calculate_reward(self, phi_diff, theta_diff):
        distance = np.sqrt(phi_diff**2 + theta_diff**2) 
        distance_portion = distance / self.max_dist
        if distance_portion <= 0.01:
            reward = 100.0
        elif distance_portion <= 0.1:
            reward = 0.8 - 0.8 * distance_portion
        else:
            reward = 0.1 - distance_portion 
        return distance, reward

    def _is_target_reached(self, phi_diff, theta_diff):
        distance = np.sqrt(phi_diff**2 + theta_diff**2) 
        distance_portion = distance / self.max_dist
        return distance_portion <= 0.02


    def step(self, action: np.ndarray):
        delta_phi = min(max(action[0], self.min_action), self.max_action) * self.delta_angle
        delta_theta = min(max(action[1], self.min_action), self.max_action) * self.delta_angle
        # delta_phi, delta_theta = action[0], action[1]
        # print(f'action: {action}, delta_phi : {delta_phi:.4f}, delta_theta : {delta_theta:.4f}')
        phi, theta = self.state
        phi   = (phi + delta_phi) % np.pi
        theta = (theta + delta_theta) % (2 * np.pi)
        self.state = np.array([phi, theta])

        phi_diff = self.angle_difference(angle1=self.target_phi, angle2=self.state[0])
        theta_diff = self.angle_difference(angle1=self.target_theta, angle2=self.state[1])
        distance, reward = self._calculate_reward(phi_diff, theta_diff)
        done = self._is_target_reached(phi_diff, theta_diff)
        info = distance

        return self.state, reward, done, False,  info


        # position = self.state[0]
        # velocity = self.state[1]
        # force = min(max(action[0], self.min_action), self.max_action)

        # velocity += force * self.power - 0.0025 * math.cos(3 * position)
        # if velocity > self.max_speed:
        #     velocity = self.max_speed
        # if velocity < -self.max_speed:
        #     velocity = -self.max_speed
        # position += velocity
        # if position > self.max_position:
        #     position = self.max_position
        # if position < self.min_position:
        #     position = self.min_position
        # if position == self.min_position and velocity < 0:
        #     velocity = 0

        # # Convert a possible numpy bool to a Python bool.
        # terminated = bool(position >= self.goal_position and velocity >= self.goal_velocity)

        # reward = 0
        # if terminated:
        #     reward = 100.0
        # reward -= math.pow(action[0], 2) * 0.1

        # self.state = np.array([position, velocity], dtype=np.float32)
        # return self.state, reward, terminated, False, {}


    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        self.state = self.observation_space.sample()
        # self.target_phi = np.random.uniform(0, np.pi)
        # self.target_theta = np.random.uniform(0, 2*np.pi)
        return np.array(self.state, dtype=np.float32), {}


    def render(self):
        pass

    def close(self):
        pass


