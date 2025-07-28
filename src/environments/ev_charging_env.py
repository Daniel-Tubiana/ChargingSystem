"""
Main EV charging environment module.
Exports the environment classes and factory functions for easy importing.
"""

from typing import List, Dict, Any
import numpy as np

from .base_env import BaseEVChargingEnv
from .ev_charging_env_continuous import EVChargingEnvContinuous
from .ev_charging_env_discrete import EVChargingEnvDiscrete

# Export main classes
__all__ = [
    'BaseEVChargingEnv',
    'EVChargingEnvContinuous',
    'EVChargingEnvDiscrete',
    'make_continuous_env',
    'make_discrete_env'
]


# Factory functions for gymnasium registration
def make_continuous_env(evs: List[Dict[str, Any]], power_limit: np.ndarray) -> EVChargingEnvContinuous:
    """
    Factory function for continuous reward environment.

    Args:
        evs: List of EV data dictionaries
        power_limit: Array of power limits for each hour

    Returns:
        Continuous reward environment instance
    """
    return EVChargingEnvContinuous(evs=evs, power_limit=power_limit)


def make_discrete_env(evs: List[Dict[str, Any]], power_limit: np.ndarray) -> EVChargingEnvDiscrete:
    """
    Factory function for discrete reward environment.

    Args:
        evs: List of EV data dictionaries
        power_limit: Array of power limits for each hour

    Returns:
        Discrete reward environment instance
    """
    return EVChargingEnvDiscrete(evs=evs, power_limit=power_limit)


def get_env_info() -> Dict[str, Any]:
    """
    Get information about available environments.

    Returns:
        Dictionary with environment information
    """
    return {
        'available_envs': ['continuous', 'discrete'],
        'action_space': 'MultiDiscrete([3] * n_evs)',
        'action_meanings': {
            0: 'No charging (0 kW)',
            1: 'Slow charging (3.7 kW)',
            2: 'Fast charging (11 kW)'
        },
        'observation_space': {
            'evs': 'Box(low=0, high=300, shape=(n_evs, 5))',
            'power_allowed': 'Box(low=0, high=1000, shape=(24,))',
            'time': 'Discrete(24)'
        },
        'reward_types': {
            'continuous': 'Smooth exponential rewards/penalties',
            'discrete': 'Binary success/failure rewards'
        }
    }


def validate_env_config(env_type: str, evs: List[Dict[str, Any]], power_limit: np.ndarray) -> bool:
    """
    Validate environment configuration parameters.

    Args:
        env_type: Type of environment ('continuous' or 'discrete')
        evs: List of EV data dictionaries
        power_limit: Array of power limits

    Returns:
        True if configuration is valid

    Raises:
        ValueError: If configuration is invalid
    """
    # Validate environment type
    if env_type.lower() not in ['continuous', 'discrete']:
        raise ValueError(f"Invalid environment type: {env_type}. Must be 'continuous' or 'discrete'")

    # Validate EVs data
    if not evs:
        raise ValueError("EVs list cannot be empty")

    required_ev_keys = ['Arrival_time[h]', 'TuD (int)', 'Battery capacity [KWh]', 'ENonD', 'SOC']
    for i, ev in enumerate(evs):
        missing_keys = [key for key in required_ev_keys if key not in ev]
        if missing_keys:
            raise ValueError(f"EV {i} missing required keys: {missing_keys}")

    # Validate power limit
    if len(power_limit) != 24:
        raise ValueError(f"Power limit must have 24 values (one per hour), got {len(power_limit)}")

    if np.any(power_limit < 0):
        raise ValueError("Power limit values cannot be negative")

    return True


def create_test_environment(env_type: str = 'continuous', n_evs: int = 5) -> BaseEVChargingEnv:
    """
    Create a test environment with dummy data for testing purposes.

    Args:
        env_type: Type of environment to create
        n_evs: Number of EVs to include

    Returns:
        Test environment instance
    """
    # Create dummy EV data
    evs = []
    for i in range(n_evs):
        ev = {
            'Arrival_time[h]': np.random.randint(0, 12),
            'TuD (int)': np.random.randint(8, 24),
            'Battery capacity [KWh]': np.random.uniform(40, 80),
            'ENonD': np.random.uniform(10, 30),
            'SOC': np.random.uniform(0.2, 0.8)
        }
        evs.append(ev)

    # Create dummy power limit (sinusoidal pattern to simulate solar)
    hours = np.arange(24)
    power_limit = 50 + 30 * np.sin(np.pi * (hours - 6) / 12)  # Peak at noon
    power_limit = np.maximum(power_limit, 10)  # Minimum 10 kW

    # Validate and create environment
    validate_env_config(env_type, evs, power_limit)

    if env_type.lower() == 'continuous':
        return EVChargingEnvContinuous(evs, power_limit)
    else:
        return EVChargingEnvDiscrete(evs, power_limit)