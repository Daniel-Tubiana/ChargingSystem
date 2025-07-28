"""
Environment factory for creating EV charging environments.
Handles registration and creation of both continuous and discrete reward environments.
"""

from typing import List, Dict, Any, Callable

import numpy as np
from gymnasium.envs.registration import register
from omegaconf import DictConfig
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv

from .ev_charging_env_continuous import EVChargingEnvContinuous
from .ev_charging_env_discrete import EVChargingEnvDiscrete


class EnvironmentFactory:
    """Factory class for creating EV charging environments."""

    def __init__(self, config: DictConfig):
        self.config = config
        self._registered_envs = set()

    def create_vectorized_env(self, evs_data: List[Dict[str, Any]], power_limit: np.ndarray) -> DummyVecEnv:
        """
        Create a vectorized environment for training/evaluation.

        Args:
            evs_data: List of EV data dictionaries
            power_limit: Array of power limits for each hour

        Returns:
            Vectorized environment ready for training
        """
        # Register environment if not already registered
        env_id = self._register_environment(evs_data, power_limit)

        # Create vectorized environment
        vec_env = make_vec_env(
            env_id,
            n_envs=self.config.env.n_environments,
            seed=self.config.env.seed
        )

        return vec_env

    def _register_environment(self, evs_data: List[Dict[str, Any]], power_limit: np.ndarray) -> str:
        """
        Register the appropriate environment based on configuration.

        Args:
            evs_data: List of EV data dictionaries
            power_limit: Array of power limits for each hour

        Returns:
            Environment ID for gym registration
        """
        env_type = self.config.env.type.lower()
        env_id = f"EVChargingEnv-{env_type}-v0"

        # Only register if not already registered
        if env_id not in self._registered_envs:
            if env_type == "continuous":
                entry_point = self._make_continuous_env_factory(evs_data, power_limit)
            elif env_type == "discrete":
                entry_point = self._make_discrete_env_factory(evs_data, power_limit)
            else:
                raise ValueError(f"Unknown environment type: {env_type}")

            register(
                id=env_id,
                entry_point=entry_point,
                kwargs={'evs': evs_data, 'power_limit': power_limit}
            )

            self._registered_envs.add(env_id)

        return env_id

    def _make_continuous_env_factory(self, evs_data: List[Dict[str, Any]], power_limit: np.ndarray) -> Callable:
        """Create factory function for continuous reward environment."""

        def make_continuous_env(evs: List[Dict[str, Any]], power_limit: np.ndarray) -> EVChargingEnvContinuous:
            return EVChargingEnvContinuous(evs=evs, power_limit=power_limit)

        return make_continuous_env

    def _make_discrete_env_factory(self, evs_data: List[Dict[str, Any]], power_limit: np.ndarray) -> Callable:
        """Create factory function for discrete reward environment."""

        def make_discrete_env(evs: List[Dict[str, Any]], power_limit: np.ndarray) -> EVChargingEnvDiscrete:
            return EVChargingEnvDiscrete(evs=evs, power_limit=power_limit)

        return make_discrete_env

    def create_single_env(self, evs_data: List[Dict[str, Any]], power_limit: np.ndarray):
        """
        Create a single environment instance (for testing/debugging).

        Args:
            evs_data: List of EV data dictionaries
            power_limit: Array of power limits for each hour

        Returns:
            Single environment instance
        """
        env_type = self.config.env.type.lower()

        if env_type == "continuous":
            return EVChargingEnvContinuous(evs=evs_data, power_limit=power_limit)
        elif env_type == "discrete":
            return EVChargingEnvDiscrete(evs=evs_data, power_limit=power_limit)
        else:
            raise ValueError(f"Unknown environment type: {env_type}")


# Factory functions for gymnasium registration
def make_continuous_env(evs: List[Dict[str, Any]], power_limit: np.ndarray) -> EVChargingEnvContinuous:
    """Factory function for continuous reward environment."""
    return EVChargingEnvContinuous(evs=evs, power_limit=power_limit)


def make_discrete_env(evs: List[Dict[str, Any]], power_limit: np.ndarray) -> EVChargingEnvDiscrete:
    """Factory function for discrete reward environment."""
    return EVChargingEnvDiscrete(evs=evs, power_limit=power_limit)