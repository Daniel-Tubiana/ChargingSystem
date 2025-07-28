"""
RL models factory and configuration for EV charging system.
"""

from typing import Any, Dict, Optional, Union
import warnings

from omegaconf import DictConfig
from stable_baselines3 import A2C, PPO
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.vec_env import VecNormalize


class ModelFactory:
    """Factory class for creating and configuring RL models with device support."""

    def __init__(self, config: DictConfig):
        self.config = config
        self.supported_algorithms = {
            'A2C': A2C,
            'PPO': PPO
        }

    def create_model(self, env: VecNormalize, log_dir: str) -> BaseAlgorithm:
        """
        Create and configure an RL model based on configuration.

        Args:
            env: Vectorized environment
            log_dir: Directory for tensorboard logs

        Returns:
            Configured RL model
        """
        algorithm_name = self.config.model.algorithm.upper()

        if algorithm_name not in self.supported_algorithms:
            raise ValueError(f"Unsupported algorithm: {algorithm_name}. "
                           f"Supported algorithms: {list(self.supported_algorithms.keys())}")

        # Get algorithm class
        algorithm_class = self.supported_algorithms[algorithm_name]

        # Get algorithm-specific parameters
        algorithm_params = self._get_algorithm_params(algorithm_name)

        # Handle device selection
        device = self._get_device()

        # Common parameters
        common_params = {
            'policy': self.config.model.policy,
            'env': env,
            'verbose': self.config.model.verbose,
            'device': device,
            # Optional: Keep TensorBoard for SB3 built-in metrics
            'tensorboard_log': log_dir if self.config.logging.get('tensorboard', False) else None,
        }

        # Merge parameters
        model_params = {**common_params, **algorithm_params}

        # Create model
        model = algorithm_class(**model_params)

        print(f"Created {algorithm_name} model with parameters:")
        for key, value in algorithm_params.items():
            print(f"  {key}: {value}")
        print(f"  device: {device}")

        return model

    def _get_device(self) -> str:
        """Get the actual device to use, handling 'auto' detection."""
        import torch

        device_config = self.config.hardware.device.lower()

        if device_config == "auto":
            if torch.cuda.is_available():
                device = "cuda"
                gpu_name = torch.cuda.get_device_name(0)
                print(f"Auto-detected device: CUDA ({gpu_name})")
            else:
                device = "cpu"
                print("CUDA not available, using CPU")
        else:
            device = device_config
            if device.startswith("cuda") and not torch.cuda.is_available():
                print("CUDA requested but not available, falling back to CPU")
                device = "cpu"
            else:
                print(f"Using configured device: {device}")

        return device

    def _get_algorithm_params(self, algorithm_name: str) -> Dict[str, Any]:
        """
        Get algorithm-specific parameters from configuration.

        Args:
            algorithm_name: Name of the algorithm

        Returns:
            Dictionary of algorithm parameters
        """
        if algorithm_name == 'A2C':
            return self._get_a2c_params()
        elif algorithm_name == 'PPO':
            return self._get_ppo_params()
        else:
            return {}

    def _get_a2c_params(self) -> Dict[str, Any]:
        """Get A2C-specific parameters."""
        a2c_config = self.config.model.a2c

        params = {}

        # Learning rate
        if hasattr(a2c_config, 'learning_rate'):
            params['learning_rate'] = a2c_config.learning_rate

        # Number of steps
        if hasattr(a2c_config, 'n_steps'):
            params['n_steps'] = a2c_config.n_steps

        # Discount factor
        if hasattr(a2c_config, 'gamma'):
            params['gamma'] = a2c_config.gamma

        # GAE lambda
        if hasattr(a2c_config, 'gae_lambda'):
            params['gae_lambda'] = a2c_config.gae_lambda

        # Entropy coefficient
        if hasattr(a2c_config, 'ent_coef'):
            params['ent_coef'] = a2c_config.ent_coef

        # Value function coefficient
        if hasattr(a2c_config, 'vf_coef'):
            params['vf_coef'] = a2c_config.vf_coef

        # Maximum gradient norm
        if hasattr(a2c_config, 'max_grad_norm'):
            params['max_grad_norm'] = a2c_config.max_grad_norm

        return params

    def _get_ppo_params(self) -> Dict[str, Any]:
        """Get PPO-specific parameters."""
        ppo_config = self.config.model.ppo

        params = {}

        # Learning rate
        if hasattr(ppo_config, 'learning_rate'):
            params['learning_rate'] = ppo_config.learning_rate

        # Number of steps
        if hasattr(ppo_config, 'n_steps'):
            params['n_steps'] = ppo_config.n_steps

        # Batch size
        if hasattr(ppo_config, 'batch_size'):
            params['batch_size'] = ppo_config.batch_size

        # Number of epochs
        if hasattr(ppo_config, 'n_epochs'):
            params['n_epochs'] = ppo_config.n_epochs

        # Discount factor
        if hasattr(ppo_config, 'gamma'):
            params['gamma'] = ppo_config.gamma

        # GAE lambda
        if hasattr(ppo_config, 'gae_lambda'):
            params['gae_lambda'] = ppo_config.gae_lambda

        # Clip range
        if hasattr(ppo_config, 'clip_range'):
            params['clip_range'] = ppo_config.clip_range

        # Entropy coefficient
        if hasattr(ppo_config, 'ent_coef'):
            params['ent_coef'] = ppo_config.ent_coef

        # Value function coefficient
        if hasattr(ppo_config, 'vf_coef'):
            params['vf_coef'] = ppo_config.vf_coef

        # Maximum gradient norm
        if hasattr(ppo_config, 'max_grad_norm'):
            params['max_grad_norm'] = ppo_config.max_grad_norm

        return params

    def load_model(self, model_path: str, env: Optional[VecNormalize] = None) -> BaseAlgorithm:
        """
        Load a pre-trained model from file.

        Args:
            model_path: Path to the saved model
            env: Environment for the model (optional)

        Returns:
            Loaded RL model
        """
        algorithm_name = self.config.model.algorithm.upper()

        if algorithm_name not in self.supported_algorithms:
            raise ValueError(f"Unsupported algorithm: {algorithm_name}")

        algorithm_class = self.supported_algorithms[algorithm_name]

        # Get device (handle auto-detection)
        device = self._get_device()

        try:
            model = algorithm_class.load(
                model_path,
                env=env,
                device=device
            )
            print(f"Successfully loaded {algorithm_name} model from: {model_path}")
            print(f"Model loaded on device: {device}")
            return model
        except Exception as e:
            raise RuntimeError(f"Failed to load model from {model_path}: {str(e)}")

    def save_model(self, model: BaseAlgorithm, save_path: str) -> None:
        """
        Save a trained model to file.

        Args:
            model: RL model to save
            save_path: Path where to save the model
        """
        try:
            model.save(save_path)
            print(f"Model saved successfully to: {save_path}")
        except Exception as e:
            raise RuntimeError(f"Failed to save model to {save_path}: {str(e)}")

    def get_model_info(self, model: BaseAlgorithm) -> Dict[str, Any]:
        """
        Get information about a model.

        Args:
            model: RL model

        Returns:
            Dictionary with model information
        """
        info = {
            'algorithm': model.__class__.__name__,
            'policy': model.policy.__class__.__name__,
            'learning_rate': getattr(model, 'learning_rate', 'N/A'),
            'n_steps': getattr(model, 'n_steps', 'N/A'),
            'gamma': getattr(model, 'gamma', 'N/A'),
        }

        # Add algorithm-specific info
        if hasattr(model, 'gae_lambda'):
            info['gae_lambda'] = model.gae_lambda

        if hasattr(model, 'clip_range'):
            info['clip_range'] = model.clip_range

        if hasattr(model, 'batch_size'):
            info['batch_size'] = model.batch_size

        if hasattr(model, 'n_epochs'):
            info['n_epochs'] = model.n_epochs

        return info

    def compare_models(self, model1: BaseAlgorithm, model2: BaseAlgorithm) -> Dict[str, Any]:
        """
        Compare two models and return their differences.

        Args:
            model1: First model
            model2: Second model

        Returns:
            Dictionary with comparison results
        """
        info1 = self.get_model_info(model1)
        info2 = self.get_model_info(model2)

        comparison = {
            'same_algorithm': info1['algorithm'] == info2['algorithm'],
            'same_policy': info1['policy'] == info2['policy'],
            'differences': {}
        }

        # Find differences
        all_keys = set(info1.keys()) | set(info2.keys())
        for key in all_keys:
            val1 = info1.get(key, 'N/A')
            val2 = info2.get(key, 'N/A')

            if val1 != val2:
                comparison['differences'][key] = {'model1': val1, 'model2': val2}

        return comparison

    def create_custom_model(self,
                          algorithm_name: str,
                          env: VecNormalize,
                          custom_params: Dict[str, Any]) -> BaseAlgorithm:
        """
        Create a model with custom parameters (overriding config).

        Args:
            algorithm_name: Name of the algorithm
            env: Vectorized environment
            custom_params: Custom parameters to override config

        Returns:
            Configured RL model with custom parameters
        """
        if algorithm_name.upper() not in self.supported_algorithms:
            raise ValueError(f"Unsupported algorithm: {algorithm_name}")

        algorithm_class = self.supported_algorithms[algorithm_name.upper()]

        # Start with default parameters
        if algorithm_name.upper() == 'A2C':
            default_params = self._get_a2c_params()
        elif algorithm_name.upper() == 'PPO':
            default_params = self._get_ppo_params()
        else:
            default_params = {}

        # Common parameters
        model_params = {
            'policy': self.config.model.policy,
            'env': env,
            'verbose': self.config.model.verbose,
            'device': self.config.hardware.device,
        }

        # Merge with default algorithm params
        model_params.update(default_params)

        # Override with custom parameters
        model_params.update(custom_params)

        # Create model
        model = algorithm_class(**model_params)

        print(f"Created custom {algorithm_name} model with parameters:")
        for key, value in model_params.items():
            if key not in ['env', 'policy']:  # Skip complex objects
                print(f"  {key}: {value}")

        return model