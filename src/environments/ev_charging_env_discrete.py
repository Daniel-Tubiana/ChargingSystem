"""
Discrete reward EV charging environment.
Implements binary reward functions with clear success/failure states.
"""

from typing import List, Dict, Any

import numpy as np

from .base_env import BaseEVChargingEnv


class EVChargingEnvDiscrete(BaseEVChargingEnv):
    """EV charging environment with discrete reward functions."""

    def __init__(self, evs: List[Dict[str, Any]], power_limit: np.ndarray):
        """
        Initialize discrete reward environment.

        Args:
            evs: List of EV dictionaries containing EV characteristics
            power_limit: Array of power limits for each hour (24 hours)
        """
        super().__init__(evs, power_limit)

        # Discrete reward parameters
        self.large_penalty = -100000
        self.soc_penalty = -5000
        self.neutral_reward = 0

    def _calculate_reward(self, actions: np.ndarray, charge: np.ndarray) -> float:
        """
        Calculate discrete reward for the current step.

        Args:
            actions: Array of actions taken
            charge: Array of charge powers

        Returns:
            Total reward for this step
        """
        # Initialize reward components
        time_reward = self._calculate_discrete_time_reward(actions, charge)
        power_reward = self._calculate_discrete_power_reward(charge)
        soc_reward = self._calculate_discrete_soc_reward()

        # Combine rewards
        total_reward = time_reward + power_reward + soc_reward

        return total_reward

    def _calculate_discrete_time_reward(self, actions: np.ndarray, charge: np.ndarray) -> float:
        """
        Calculate discrete time constraint reward.
        Binary penalty/reward system.
        """
        time_reward = 0.0

        for i, (action, charge_power) in enumerate(zip(actions, charge)):
            ev = self.evs[i]

            # Check if EV is available for charging
            is_available = (ev['Arrival_time[h]'] <= self.time and ev['TuD (int)'] > 0)

            if not is_available and charge_power > 0:
                # Large penalty for invalid charging
                time_reward += self.large_penalty
            elif is_available and charge_power > 0:
                # Neutral reward for valid charging
                time_reward += self.neutral_reward
            else:
                # Neutral reward for not charging (whether valid or invalid)
                time_reward += self.neutral_reward

        return time_reward

    def _calculate_discrete_power_reward(self, charge: np.ndarray) -> float:
        """
        Calculate discrete power constraint reward.
        Binary success/failure based on power limit.
        """
        total_charge = np.sum(charge)
        available_power = self.power_limit[self.time]

        if total_charge > available_power:
            # Large penalty for exceeding power limit
            return self.large_penalty
        else:
            # Neutral reward for staying within limits
            return self.neutral_reward

    def _calculate_discrete_soc_reward(self) -> float:
        """
        Calculate discrete SOC reward.
        Only evaluated at episode end with binary success/failure.
        """
        # Only calculate SOC reward at episode termination
        if not self.terminated:
            return 0.0

        soc_reward = 0.0
        min_soc = 0.2

        for ev in self.evs:
            # Calculate required SOC
            required_energy_ratio = ev['ENonD'] / ev['Battery capacity [KWh]']
            required_soc = required_energy_ratio + min_soc

            if ev['SOC'] < required_soc:
                # Binary penalty for not meeting SOC requirement
                soc_reward += self.soc_penalty
            else:
                # Neutral reward for meeting requirement
                soc_reward += self.neutral_reward

        return soc_reward

    def _calculate_charging_success_reward(self) -> float:
        """
        Calculate reward based on successful charging sessions.
        Only applicable at episode end.
        """
        if not self.terminated:
            return 0.0

        success_reward = 0.0
        success_bonus = 100  # Bonus for each successfully charged EV

        for ev in self.evs:
            # Check if EV met its energy requirements
            min_soc = 0.2
            required_energy_ratio = ev['ENonD'] / ev['Battery capacity [KWh]']
            required_soc = required_energy_ratio + min_soc

            if ev['SOC'] >= required_soc:
                success_reward += success_bonus

        return success_reward

    def _calculate_constraint_violation_penalty(self, actions: np.ndarray, charge: np.ndarray) -> float:
        """
        Calculate additional penalties for constraint violations.
        Can be used to fine-tune the discrete reward structure.
        """
        penalty = 0.0

        # Count violations
        time_violations = 0
        for i, (action, charge_power) in enumerate(zip(actions, charge)):
            ev = self.evs[i]
            is_available = (ev['Arrival_time[h]'] <= self.time and ev['TuD (int)'] > 0)

            if not is_available and charge_power > 0:
                time_violations += 1

        # Power violations
        total_charge = np.sum(charge)
        available_power = self.power_limit[self.time]
        power_violation = max(0, total_charge - available_power)

        # Apply scaled penalties
        if time_violations > 0:
            penalty += self.large_penalty * time_violations

        if power_violation > 0:
            penalty += self.large_penalty  # Binary penalty regardless of violation magnitude

        return penalty

    def get_success_metrics(self) -> Dict[str, float]:
        """
        Get success metrics for the current episode.
        Useful for analysis and debugging.

        Returns:
            Dictionary with success metrics
        """
        if not self.terminated:
            return {}

        metrics = {
            'total_evs': len(self.evs),
            'successful_charges': 0,
            'failed_charges': 0,
            'average_final_soc': 0.0,
            'min_final_soc': 1.0,
            'max_final_soc': 0.0
        }

        min_soc = 0.2
        final_socs = []

        for ev in self.evs:
            required_energy_ratio = ev['ENonD'] / ev['Battery capacity [KWh]']
            required_soc = required_energy_ratio + min_soc
            final_soc = ev['SOC']

            final_socs.append(final_soc)

            if final_soc >= required_soc:
                metrics['successful_charges'] += 1
            else:
                metrics['failed_charges'] += 1

        if final_socs:
            metrics['average_final_soc'] = np.mean(final_socs)
            metrics['min_final_soc'] = np.min(final_socs)
            metrics['max_final_soc'] = np.max(final_socs)

        metrics['success_rate'] = metrics['successful_charges'] / metrics['total_evs']

        return metrics