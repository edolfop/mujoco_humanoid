import gymnasium as gym
from gymnasium.envs.mujoco.humanoid_v5 import HumanoidEnv
from gymnasium.envs.registration import register

class CustomHumanoidEnv(HumanoidEnv):
    def _get_reward(self, x_velocity: float, action):
        forward_reward = self._forward_reward_weight * (x_velocity * x_velocity)
        healthy_reward = self.healthy_reward
        rewards = forward_reward + healthy_reward

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
    
try:
    register(
        id="CustomHumanoid-v0",
        entry_point="custom_humanoid:CustomHumanoidEnv",
        max_episode_steps=1000,
    )
except Exception:
    pass