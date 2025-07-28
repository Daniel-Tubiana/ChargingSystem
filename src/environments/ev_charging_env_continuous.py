"""
Continuous reward EV charging environment.
Implements smooth, continuous reward functions for better gradient flow.
"""

from typing import List, Dict, Any

import numpy as np

from .base_env import BaseEVChargingEnv


class EVChargingEnvContinuous(BaseEVChargingEnv):
    """EV charging environment with continuous reward functions."""

    def __init__(self, evs: List[Dict[str, Any]], power_limit: np.ndarray):
        """
        Initialize continuous reward environment.

        Args:
            evs: List of EV dictionaries containing EV characteristics
            power_limit: Array of power limits for each hour (24 hours)
        """
        super().__init__(evs, power_limit)

        # Continuous reward parameters
        self.exp_const_soc = 5.0
        self.penalty_const_soc = 2.0
        self.exp_const_power = 3.0
        self.penalty_const_power = 1.0

    def _calculate_reward(self, actions: np.ndarray, charge: np.ndarray) -> float:
        """
        Calculate continuous reward for the current step.

        Args:
            actions: Array of actions taken
            charge: Array of charge powers

        Returns:
            Total reward for this step
        """
        # Initialize reward components
        time_reward = self._calculate_continuous_time_reward(actions, charge)
        power_reward = self._calculate_continuous_power_reward(charge)
        soc_reward = self._calculate_continuous_soc_reward()

        # Combine rewards
        total_reward = time_reward + power_reward + soc_reward

        return total_reward

    def _calculate_continuous_time_reward(self, actions: np.ndarray, charge: np.ndarray) -> float:
        """
        Calculate continuous time constraint reward.
        Uses smooth penalty functions instead of binary penalties.
        """
        time_reward = 0.0

        for i, (action, charge_power) in enumerate(zip(actions, charge)):
            ev = self.evs[i]

            # Check if EV is available for charging
            is_available = (ev['Arrival_time[h]'] <= self.time and ev['TuD (int)'] > 0)

            if not is_available and charge_power > 0:
                # Exponential penalty for charging when unavailable
                penalty_factor = charge_power / max(self.charge_actions)  # Normalize by max charge
                time_reward -= 100000 * np.exp(penalty_factor)
            elif is_available:
                # Small positive reward for valid charging opportunities
                time_reward += 0  # Neutral reward for valid states

        return time_reward

    def _calculate_continuous_power_reward(self, charge: np.ndarray) -> float:
        """
        Calculate continuous power constraint reward using exponential function.
        """
        total_charge = np.sum(charge)
        available_power = self.power_limit[self.time]

        if available_power <= 0:
            return -100000 if total_charge > 0 else 0

        # Calculate power utilization ratio
        power_ratio = total_charge / available_power

        if power_ratio <= 1.0:
            # Reward efficient power usage (closer to limit is better)
            power_reward = 1.0 - np.exp(-self.exp_const_power * power_ratio)
            return power_reward
        else:
            # Exponential penalty for exceeding power limit
            excess_ratio = power_ratio - 1.0
            power_penalty = -self.penalty_const_power * np.exp(self.exp_const_power * excess_ratio)
            return np.clip(power_penalty, a_min=-100000, a_max=0)

    def _calculate_continuous_soc_reward(self) -> float:
        """
        Calculate continuous SOC reward.
        Provides immediate feedback during episode and final evaluation.
        """
        soc_reward = 0.0
        min_soc = 0.2

        for ev in self.evs:
            # Calculate required SOC
            required_energy_ratio = ev['ENonD'] / ev['Battery capacity [KWh]']
            required_soc = required_energy_ratio + min_soc

            # Current SOC relative to requirement
            soc_difference = required_soc - ev['SOC']

            if soc_difference <= 0:
                # SOC requirement met or exceeded
                if self.terminated:
                    soc_reward += 0  # No penalty for meeting requirements
                else:
                    # Small positive reward during episode for progress
                    soc_reward += 0.1
            else:
                # SOC requirement not met
                if self.terminated:
                    # Final penalty at episode end
                    normalized_difference = soc_difference / required_soc
                    penalty = 1.0 - np.exp(-self.exp_const_soc * normalized_difference)
                    soc_reward += self.penalty_const_soc * np.clip(penalty, a_min=-1000000, a_max=0)
                else:
                    # Gradual penalty during episode to encourage progress
                    normalized_difference = soc_difference / required_soc
                    penalty = -0.1 * normalized_difference  # Small continuous penalty
                    soc_reward += penalty

        return soc_reward

    def _calculate_efficiency_bonus(self, charge: np.ndarray) -> float:
        """
        Calculate bonus reward for efficient charging patterns.
        Encourages using available power effectively.
        """
        total_charge = np.sum(charge)
        available_power = self.power_limit[self.time]

        if available_power <= 0:
            return 0.0

        # Efficiency ratio (0 to 1)
        efficiency = total_charge / available_power

        # Bonus for high efficiency (close to 1.0)
        if 0.8 <= efficiency <= 1.0:
            return 10.0 * efficiency
        elif 0.5 <= efficiency < 0.8:
            return 5.0 * efficiency
        else:
            return 0.0

    def _calculate_charging_consistency_reward(self) -> float:
        """
        Calculate reward for consistent charging patterns.
        Encourages stable charging behavior.
        """
        # This could track charging history and reward consistency
        # For now, return 0 (can be extended based on requirements)
        return 0.0