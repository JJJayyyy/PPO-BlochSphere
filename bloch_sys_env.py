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



    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        self.state = self.observation_space.sample()
        self.target_phi = np.random.uniform(0, np.pi)
        self.target_theta = np.random.uniform(0, 2*np.pi)
        return np.array(self.state, dtype=np.float32), {}


    def render(self):
        pass

    def close(self):
        pass



