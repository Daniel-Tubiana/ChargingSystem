"""
Base environment class for EV charging system.
Provides common functionality for both continuous and discrete reward environments.
"""

import copy
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Tuple, Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class BaseEVChargingEnv(gym.Env, ABC):
    """Base class for EV charging environments."""

    def __init__(self, evs: List[Dict[str, Any]], power_limit: np.ndarray):
        """
        Initialize the EV charging environment.

        Args:
            evs: List of EV dictionaries containing EV characteristics
            power_limit: Array of power limits for each hour (24 hours)
        """
        super().__init__()

        # Environment state
        self.time: Optional[int] = None
        self.total_reward: Optional[float] = None
        self.terminated: bool = False
        self.truncated: bool = False

        # Data storage
        self.evs_const = evs  # Original EV dataset (immutable)
        self.evs: Optional[List[Dict[str, Any]]] = None  # Working copy
        self.power_limit = power_limit  # Power limits per hour

        # Environment parameters
        self.n_evs = len(evs)
        self.n_hours = 24
        self.charge_actions = [0, 3.7, 11]  # Available charging levels in kW

        # Action space: 3 actions per EV (no charge, slow charge, fast charge)
        self.action_space = spaces.MultiDiscrete([3] * self.n_evs)

        # Observation space
        self._setup_observation_space()

    def _setup_observation_space(self) -> None:
        """Setup the observation space based on EV features."""
        # Get number of features from first EV
        ev_features = len(self.evs_const[0].keys())

        self.observation_space = spaces.Dict({
            'evs': spaces.Box(
                low=0,
                high=300,
                shape=(self.n_evs, ev_features),
                dtype=np.float32
            ),
            'power_allowed': spaces.Box(
                low=0,
                high=1000,
                shape=(self.n_hours,),
                dtype=np.float32
            ),
            'time': spaces.Discrete(self.n_hours)
        })

    def reset(self, **kwargs) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """Reset the environment to initial state."""
        # Reset environment state
        self.terminated = False
        self.truncated = False
        self.time = 0
        self.total_reward = 0.0

        # Reset EV data (deep copy to avoid reference issues)
        self.evs = copy.deepcopy(self.evs_const)

        # Create initial observation
        observation = self._create_observation()
        info = self._create_info()

        return observation, info

    def step(self, actions: np.ndarray) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        """
        Execute one step in the environment.

        Args:
            actions: Array of actions for each EV

        Returns:
            observation: New state observation
            reward: Reward for this step
            terminated: Whether episode has terminated
            truncated: Whether episode was truncated
            info: Additional information
        """
        # Validate actions
        if len(actions) != self.n_evs:
            raise ValueError(f"Expected {self.n_evs} actions, got {len(actions)}")

        # Convert actions to charge values
        charge = np.array([self.charge_actions[action] for action in actions])

        # Calculate rewards
        reward = self._calculate_reward(actions, charge)

        # Update EV states
        self._update_ev_states(actions, charge)

        # Update time and check termination
        self.time += 1
        if self.time >= self.n_hours:
            self.terminated = True

        # Update total reward
        self.total_reward += reward

        # Create new observation and info
        observation = self._create_observation()
        info = self._create_info(charge)

        return observation, reward, self.terminated, self.truncated, info

    def _create_observation(self) -> Dict[str, np.ndarray]:
        """Create observation dictionary from current state."""
        # EV observations: [arrival_time, time_until_departure, battery_capacity, energy_needed, soc]
        evs_obs = np.array([
            [
                ev['Arrival_time[h]'],
                ev['TuD (int)'],
                ev['Battery capacity [KWh]'],
                ev['ENonD'],
                ev['SOC']
            ]
            for ev in self.evs
        ], dtype=np.float32)

        power_allowed_obs = np.array(self.power_limit, dtype=np.float32)
        time_obs = self.time

        return {
            'evs': evs_obs,
            'power_allowed': power_allowed_obs,
            'time': time_obs
        }

    def _create_info(self, charge: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Create info dictionary with additional debugging information."""
        info = {
            'total_reward': self.total_reward,
            'time': self.time,
            'terminated': self.terminated,
            'truncated': self.truncated
        }

        if charge is not None:
            info['charge_pwr'] = np.sum(charge)
            info['individual_charges'] = charge.copy()

        return info

    def _update_ev_states(self, actions: np.ndarray, charge: np.ndarray) -> None:
        """Update the state of each EV based on actions."""
        for i, (action, charge_power) in enumerate(zip(actions, charge)):
            ev = self.evs[i]

            # Only charge if EV is available and has time remaining
            if (ev['Arrival_time[h]'] <= self.time and ev['TuD (int)'] > 0):
                # Update SOC based on charging power
                if charge_power > 0:
                    soc_increase = charge_power / ev['Battery capacity [KWh]']
                    ev['SOC'] = min(1.0, ev['SOC'] + soc_increase)

            # Update time until departure
            if ev['TuD (int)'] > 0:
                ev['TuD (int)'] -= 1

    @abstractmethod
    def _calculate_reward(self, actions: np.ndarray, charge: np.ndarray) -> float:
        """
        Calculate reward for the current step.
        Must be implemented by subclasses for specific reward structures.

        Args:
            actions: Array of actions taken
            charge: Array of charge powers

        Returns:
            Total reward for this step
        """
        pass

    def _calculate_time_constraint_reward(self, actions: np.ndarray, charge: np.ndarray) -> float:
        """Calculate reward/penalty for time constraints."""
        time_reward = 0.0

        for i, (action, charge_power) in enumerate(zip(actions, charge)):
            ev = self.evs[i]

            # Check if EV is trying to charge when not available
            if (ev['Arrival_time[h]'] > self.time or ev['TuD (int)'] <= 0):
                if charge_power > 0:
                    time_reward -= 100000  # Large penalty for invalid charging
                else:
                    time_reward += 0  # No penalty for not charging when unavailable

        return time_reward

    def _calculate_power_constraint_reward(self, charge: np.ndarray) -> float:
        """Calculate reward/penalty for power constraints."""
        total_charge = np.sum(charge)
        available_power = self.power_limit[self.time]

        if total_charge > available_power:
            return -100000  # Large penalty for exceeding power limit
        else:
            return 0  # No penalty for staying within limits

    def _calculate_soc_reward(self) -> float:
        """Calculate reward based on SOC levels (only at episode end)."""
        if not self.terminated:
            return 0.0

        soc_reward = 0.0
        min_soc = 0.2  # Minimum SOC requirement

        for ev in self.evs:
            required_soc = (ev['ENonD'] / ev['Battery capacity [KWh]']) + min_soc

            if ev['SOC'] < required_soc:
                soc_reward -= 5000  # Penalty for not meeting SOC requirement

        return soc_reward

    def render(self, mode: str = 'human') -> Optional[np.ndarray]:
        """Render the environment (optional implementation)."""
        if mode == 'human':
            print(f"Time: {self.time}/24, Total Reward: {self.total_reward:.2f}")
            print(f"Power used: {np.sum([self.charge_actions[0]] * self.n_evs):.2f}/"
                  f"{self.power_limit[self.time]:.2f} kW")
        return None

    def close(self) -> None:
        """Clean up resources."""
        pass