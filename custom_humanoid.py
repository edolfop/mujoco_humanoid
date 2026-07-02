import mujoco

import gymnasium as gym
import numpy as np
from gymnasium.envs.mujoco.humanoid_v5 import HumanoidEnv
#from gymnasium.envs.mujoco.utils import mass_center
from gymnasium.envs.registration import register

from collections import deque

def mass_center(model, data):
    mass = model.body_mass.reshape(-1, 1)
    xpos = data.xipos
    return (mass * xpos).sum(axis=0) / mass.sum()

class CustomHumanoidEnv(HumanoidEnv):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        print("init")
        # Variables para guardar estado entre pasos
        self.left_knee_history = deque([0.0, 0.0, 0.0], maxlen=3)
        self.right_knee_history = deque([0.0, 0.0, 0.0], maxlen=3)
        self.left_knee_qpos_idx = None#11
        self.right_knee_qpos_idx = None#7    
        self.delta_knee_mov = 0.02 #original 0.1
        self.knee_reward_base = 1
    
    def reset(self, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        
        # Mapear joint -> índice en qpos (solo una vez)
        if self.left_knee_qpos_idx is None:
            l_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, 'left_knee')
            r_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, 'right_knee')
            self.left_knee_qpos_idx = self.model.jnt_qposadr[l_id]
            self.right_knee_qpos_idx = self.model.jnt_qposadr[r_id]
            print("left knee: ", self.left_knee_qpos_idx, " right knee: ", self.right_knee_qpos_idx)

        # Inicializar historial con la posición real para evitar sesgo de ceros
        start_l = self.data.qpos[self.left_knee_qpos_idx]
        start_r = self.data.qpos[self.right_knee_qpos_idx]
        self.left_knee_history = deque([start_l]*3, maxlen=3)
        self.right_knee_history = deque([start_r]*3, maxlen=3)
        
        return obs, info    
    
    def _get_reward(self, x_velocity: float, action):
        forward_reward = self._forward_reward_weight * x_velocity
        healthy_reward = self.healthy_reward

        curr_left  = self.data.qpos[self.left_knee_qpos_idx]
        curr_right = self.data.qpos[self.right_knee_qpos_idx]
        
        avg_left = sum(self.left_knee_history) / 3.0
        avg_right = sum(self.right_knee_history) / 3.0

        delta_left = curr_left - avg_left
        delta_right = curr_right - avg_right

        # reward for moving the knees, both
        if (delta_left > self.delta_knee_mov ) & (delta_right > self.delta_knee_mov ) :
            knee_reward = 1 * self.knee_reward_base
        else:
            knee_reward = -2 * self.knee_reward_base
            
        self.left_knee_history.append(curr_left)
        self.right_knee_history.append(curr_right)
        
        rewards = forward_reward + healthy_reward + knee_reward

        ctrl_cost = self.control_cost(action)
        contact_cost = self.contact_cost
        costs = ctrl_cost + contact_cost

        reward = rewards - costs

        reward_info = {
            "reward_survive": healthy_reward,
            "reward_forward": forward_reward,
            "reward_ctrl": -ctrl_cost,
            "reward_contact": -contact_cost,
        }

        return reward, reward_info
    
    def step(self, action):
        xy_position_before = mass_center(self.model, self.data)
        self.do_simulation(action, self.frame_skip)
        xy_position_after = mass_center(self.model, self.data)

        xy_velocity = (xy_position_after - xy_position_before) / self.dt
        x_velocity, y_velocity,_ = xy_velocity

        observation = self._get_obs()
        reward, reward_info = self._get_rew(x_velocity, action)
        terminated = (not self.is_healthy) and self._terminate_when_unhealthy
        info = {
            "x_position": self.data.qpos[0],
            "y_position": self.data.qpos[1],
            "tendon_length": self.data.ten_length,
            "tendon_velocity": self.data.ten_velocity,
            "distance_from_origin": np.linalg.norm(self.data.qpos[0:2], ord=2),
            "x_velocity": x_velocity,
            "y_velocity": y_velocity,
            **reward_info,
        }

        if self.render_mode == "human":
            self.render()
        # truncation=False as the time limit is handled by the `TimeLimit` wrapper added during `make`
        return observation, reward, terminated, False, info
    
try:
    register(
        id="CustomHumanoid-v0",
        entry_point="custom_humanoid:CustomHumanoidEnv",
        max_episode_steps=1000,
    )
except Exception:
    pass