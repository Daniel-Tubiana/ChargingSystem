'''"""
Main entry point for EV charging system training and evaluation.
Uses Hydra for configuration management and supports both continuous and discrete reward environments.
"""

import os
import time
from pathlib import Path
from typing import Dict, Any, Optional

import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf
from stable_baselines3 import A2C, PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize

from data.data_loader import DataLoader
from environments.environment_factory import EnvironmentFactory
from evaluation.evaluator import Evaluator
from models.rl_models import ModelFactory


class EVChargingTrainer:
    """Main trainer class for EV charging system."""

    def __init__(self, config: DictConfig):
        self.config = config
        self.timestamp = int(time.time()) if config.run.timestamp else ""
        self.run_name = f"{self.timestamp}_{config.run.name}" if self.timestamp else config.run.name

        # Initialize paths
        self.setup_directories()

        # Initialize components
        self.data_loader = DataLoader(config)
        self.env_factory = EnvironmentFactory(config)
        self.model_factory = ModelFactory(config)
        self.evaluator = Evaluator(config)

    def setup_directories(self) -> None:
        """Create necessary directories for saving models, logs, and results."""
        base_paths = {
            'models': Path(self.config.logging.models_dir) / self.run_name,
            'logs': Path(self.config.logging.log_dir) / self.run_name,
            'results': Path(self.config.logging.results_dir) / self.run_name
        }

        for path in base_paths.values():
            path.mkdir(parents=True, exist_ok=True)

        self.paths = base_paths

    def load_data(self) -> Dict[str, Any]:
        """Load training and test data."""
        print("Loading data...")
        return self.data_loader.load_all_data()

    def create_environment(self, data: Dict[str, Any], is_training: bool = True) -> VecNormalize:
        """Create vectorized and normalized environment."""
        print(f"Creating {'training' if is_training else 'evaluation'} environment...")

        # Select appropriate dataset
        dataset_key = 'df_evs_train' if is_training else 'df_evs_test'
        evs_data = data[dataset_key].to_dict('records')
        power_limit = data['df_build']['surplus_power[kw]'].to_numpy()

        # Create environment
        env = self.env_factory.create_vectorized_env(evs_data, power_limit)

        # Apply normalization if configured
        if self.config.training.normalize.norm_obs:
            env = VecNormalize(
                env,
                norm_obs=self.config.training.normalize.norm_obs,
                norm_reward=self.config.training.normalize.norm_reward,
                norm_obs_keys=self.config.training.normalize.norm_obs_keys
            )

        return env

    def create_model(self, env: VecNormalize) -> Any:
        """Create RL model."""
        print(f"Creating {self.config.model.algorithm} model...")
        return self.model_factory.create_model(env, str(self.paths['logs']))

    def setup_callbacks(self, eval_env: Optional[VecNormalize] = None) -> list:
        """Setup training callbacks."""
        callbacks = []

        if self.config.evaluation.enabled and eval_env is not None:
            eval_callback = EvalCallback(
                eval_env,
                log_path=str(self.paths['logs']),
                eval_freq=self.config.evaluation.eval_freq,
                n_eval_episodes=self.config.evaluation.n_eval_episodes,
                deterministic=self.config.evaluation.deterministic
            )
            callbacks.append(eval_callback)

        return callbacks

    def train_model(self, model: Any, env: VecNormalize, callbacks: list) -> Any:
        """Train the RL model."""
        print("Starting training...")

        total_timesteps = self.config.training.total_timesteps
        timesteps_per_save = self.config.training.timesteps_per_save

        iters = 0
        while iters * timesteps_per_save < total_timesteps:
            iters += 1
            current_timesteps = min(timesteps_per_save, total_timesteps - (iters - 1) * timesteps_per_save)

            model.learn(
                total_timesteps=current_timesteps,
                reset_num_timesteps=self.config.training.reset_num_timesteps,
                tb_log_name=self.config.model.algorithm,
                callback=callbacks
            )

            # Save model checkpoint
            if self.config.output.save_model:
                model_path = self.paths['models'] / f"{self.run_name}_{iters * timesteps_per_save}"
                model.save(str(model_path))
                print(f"Model saved: {model_path}")

        return model

    def evaluate_model(self, model: Any, eval_env: VecNormalize, data: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate trained model and collect results."""
        print("Evaluating model...")
        return self.evaluator.evaluate_model(model, eval_env, data, self.run_name)

    def save_results(self, results: Dict[str, Any]) -> None:
        """Save evaluation results to files."""
        if not self.config.output.save_results:
            return

        print("Saving results...")

        # Save to Excel if configured
        if self.config.output.export_excel:
            excel_path = self.paths['results'] / f"{self.run_name}.xlsx"

            with pd.ExcelWriter(str(excel_path), engine='xlsxwriter') as writer:
                for sheet_name in self.config.output.excel_sheets:
                    if sheet_name in results:
                        results[sheet_name].to_excel(writer, sheet_name=sheet_name, index=False)

            print(f"Results saved to: {excel_path}")

    def run_training(self) -> None:
        """Main training pipeline."""
        try:
            # Load data
            data = self.load_data()

            # Create environments
            train_env = self.create_environment(data, is_training=True)
            eval_env = self.create_environment(data, is_training=False) if self.config.evaluation.enabled else None

            # Create model
            model = self.create_model(train_env)

            # Setup callbacks
            callbacks = self.setup_callbacks(eval_env)

            # Train model
            trained_model = self.train_model(model, train_env, callbacks)

            # Evaluate model
            if eval_env is not None:
                results = self.evaluate_model(trained_model, eval_env, data)
                self.save_results(results)

            print(f"Training completed successfully! Run: {self.run_name}")

        except Exception as e:
            print(f"Training failed with error: {str(e)}")
            raise


@hydra.main(version_base=None, config_path="../config", config_name="train_config")
def main(cfg: DictConfig) -> None:
    """Main function using Hydra for configuration management."""
    print("=" * 60)
    print("EV Charging System Training")
    print("=" * 60)
    print(f"Configuration:\n{OmegaConf.to_yaml(cfg)}")

    # Initialize trainer
    trainer = EVChargingTrainer(cfg)

    # Run training
    trainer.run_training()


if __name__ == "__main__":
    main()'''

"""
Main entry point for EV charging system training and evaluation.
Updated to use MLflow for experiment tracking instead of TensorBoard.
"""

import os
import time
from pathlib import Path
from typing import Dict, Any, Optional

import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf
from stable_baselines3 import A2C, PPO
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize

from data.data_loader import DataLoader
from environments.environment_factory import EnvironmentFactory
from evaluation.evaluator import Evaluator
from models.rl_models import ModelFactory
from utils.mlflow_logger import MLflowLogger


class MLflowCallback(BaseCallback):
    """Custom callback to log training metrics to MLflow."""

    def __init__(self, mlflow_logger: MLflowLogger, log_freq: int = 1000):
        super().__init__()
        self.mlflow_logger = mlflow_logger
        self.log_freq = log_freq
        self.episode_count = 0

    def _on_step(self) -> bool:
        """Called at each training step."""
        # Log metrics periodically
        if self.n_calls % self.log_freq == 0:
            # Get current metrics from the training environment
            if hasattr(self.training_env, 'get_attr'):
                try:
                    # Get info from first environment
                    infos = self.training_env.get_attr('get_wrapper_attr', 'infos')
                    if infos and len(infos) > 0 and len(infos[0]) > 0:
                        info = infos[0][0]  # First environment, latest info

                        self.mlflow_logger.log_metrics({
                            "training_step": self.n_calls,
                            "episode_reward": info.get('total_reward', 0),
                            "charge_power": info.get('charge_pwr', 0)
                        }, step=self.n_calls)
                except:
                    pass  # Silently continue if we can't get metrics

        return True


class EVChargingTrainer:
    """Main trainer class for EV charging system with MLflow integration."""

    def __init__(self, config: DictConfig):
        self.config = config
        self.timestamp = int(time.time()) if config.run.timestamp else ""
        self.run_name = f"{self.timestamp}_{config.run.name}" if self.timestamp else config.run.name

        # Initialize MLflow logger
        self.mlflow_logger = MLflowLogger(config) if config.mlflow.enabled else None

        # Initialize paths
        self.setup_directories()

        # Initialize components
        self.data_loader = DataLoader(config)
        self.env_factory = EnvironmentFactory(config)
        self.model_factory = ModelFactory(config)
        self.evaluator = Evaluator(config)

    def setup_directories(self) -> None:
        """Create necessary directories for saving models, logs, and results."""
        base_paths = {
            'models': Path(self.config.logging.models_dir) / self.run_name,
            'logs': Path(self.config.logging.log_dir) / self.run_name,
            'results': Path(self.config.logging.results_dir) / self.run_name
        }

        for path in base_paths.values():
            path.mkdir(parents=True, exist_ok=True)

        self.paths = base_paths

    def load_data(self) -> Dict[str, Any]:
        """Load training and test data."""
        print("Loading data...")
        return self.data_loader.load_all_data()

    def create_environment(self, data: Dict[str, Any], is_training: bool = True) -> VecNormalize:
        """Create vectorized and normalized environment."""
        print(f"Creating {'training' if is_training else 'evaluation'} environment...")

        # Select appropriate dataset
        dataset_key = 'df_evs_train' if is_training else 'df_evs_test'
        evs_data = data[dataset_key].to_dict('records')
        power_limit = data['df_build']['surplus_power[kw]'].to_numpy()

        # Create environment
        env = self.env_factory.create_vectorized_env(evs_data, power_limit)

        # Apply normalization if configured
        if self.config.training.normalize.norm_obs:
            env = VecNormalize(
                env,
                norm_obs=self.config.training.normalize.norm_obs,
                norm_reward=self.config.training.normalize.norm_reward,
                norm_obs_keys=self.config.training.normalize.norm_obs_keys
            )

        return env

    def create_model(self, env: VecNormalize) -> Any:
        """Create RL model."""
        print(f"Creating {self.config.model.algorithm} model...")

        # Use MLflow for logging instead of TensorBoard
        log_dir = None  # Disable SB3's built-in tensorboard logging
        if self.config.logging.tensorboard:
            log_dir = str(self.paths['logs'])

        return self.model_factory.create_model(env, log_dir)

    def setup_callbacks(self, eval_env: Optional[VecNormalize] = None) -> list:
        """Setup training callbacks including MLflow logging."""
        callbacks = []

        # Add MLflow callback for training metrics
        if self.mlflow_logger:
            mlflow_callback = MLflowCallback(
                self.mlflow_logger,
                log_freq=self.config.get('mlflow_log_freq', 1000)
            )
            callbacks.append(mlflow_callback)

        # Add evaluation callback
        if self.config.evaluation.enabled and eval_env is not None:
            eval_callback = EvalCallback(
                eval_env,
                log_path=str(self.paths['logs']),
                eval_freq=self.config.evaluation.eval_freq,
                n_eval_episodes=self.config.evaluation.n_eval_episodes,
                deterministic=self.config.evaluation.deterministic
            )
            callbacks.append(eval_callback)

        return callbacks

    def train_model(self, model: Any, env: VecNormalize, callbacks: list) -> Any:
        """Train the RL model with MLflow tracking."""
        print("Starting training...")

        total_timesteps = self.config.training.total_timesteps
        timesteps_per_save = self.config.training.timesteps_per_save

        iters = 0
        while iters * timesteps_per_save < total_timesteps:
            iters += 1
            current_timesteps = min(timesteps_per_save, total_timesteps - (iters - 1) * timesteps_per_save)

            # Train model
            model.learn(
                total_timesteps=current_timesteps,
                reset_num_timesteps=self.config.training.reset_num_timesteps,
                callback=callbacks
            )

            # Log training progress to MLflow
            if self.mlflow_logger:
                progress_metrics = {
                    "training_progress": (iters * timesteps_per_save) / total_timesteps,
                    "total_timesteps_trained": iters * timesteps_per_save,
                    "training_iteration": iters
                }
                self.mlflow_logger.log_metrics(progress_metrics)

            # Save model checkpoint
            if self.config.output.save_model:
                model_path = self.paths['models'] / f"{self.run_name}_{iters * timesteps_per_save}"
                model.save(str(model_path))
                print(f"Model saved: {model_path}")

                # Log model to MLflow
                if self.mlflow_logger:
                    self.mlflow_logger.log_model(model, f"checkpoint_{iters}")

        return model

    def evaluate_model(self, model: Any, eval_env: VecNormalize, data: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate trained model and collect results."""
        print("Evaluating model...")
        results = self.evaluator.evaluate_model(model, eval_env, data, self.run_name)

        # Log evaluation results to MLflow
        if self.mlflow_logger:
            self.mlflow_logger.log_evaluation_results(results)

        return results

    def save_results(self, results: Dict[str, Any]) -> None:
        """Save evaluation results to files."""
        if not self.config.output.save_results:
            return

        print("Saving results...")

        # Save to Excel if configured
        if self.config.output.export_excel:
            excel_path = self.paths['results'] / f"{self.run_name}.xlsx"

            with pd.ExcelWriter(str(excel_path), engine='xlsxwriter') as writer:
                for sheet_name in self.config.output.excel_sheets:
                    if sheet_name in results:
                        results[sheet_name].to_excel(writer, sheet_name=sheet_name, index=False)

            print(f"Results saved to: {excel_path}")

    def run_training(self) -> None:
        """Main training pipeline with MLflow tracking."""
        # Start MLflow run
        if self.mlflow_logger:
            self.mlflow_logger.start_run(self.run_name)

        try:
            # Load data
            data = self.load_data()

            # Log data summary to MLflow
            if self.mlflow_logger:
                data_metrics = {
                    "n_training_evs": len(data['df_evs_train']),
                    "n_test_evs": len(data['df_evs_test']),
                    "max_surplus_power": float(data['df_build']['surplus_power[kw]'].max()),
                    "avg_surplus_power": float(data['df_build']['surplus_power[kw]'].mean())
                }
                self.mlflow_logger.log_metrics(data_metrics)

            # Create environments
            train_env = self.create_environment(data, is_training=True)
            eval_env = self.create_environment(data, is_training=False) if self.config.evaluation.enabled else None

            # Create model
            model = self.create_model(train_env)

            # Log model info to MLflow
            if self.mlflow_logger:
                model_info = self.model_factory.get_model_info(model)
                # Convert model info to metrics (flatten nested dicts)
                flat_info = {}
                for k, v in model_info.items():
                    if isinstance(v, (int, float)):
                        flat_info[f"model_{k}"] = v
                self.mlflow_logger.log_metrics(flat_info)

            # Setup callbacks
            callbacks = self.setup_callbacks(eval_env)

            # Train model
            trained_model = self.train_model(model, train_env, callbacks)

            # Log final model
            if self.mlflow_logger:
                self.mlflow_logger.log_model(trained_model, "final_model")

            # Evaluate model
            if eval_env is not None:
                results = self.evaluate_model(trained_model, eval_env, data)
                self.save_results(results)

            print(f"Training completed successfully! Run: {self.run_name}")

            # Print MLflow run info
            if self.mlflow_logger:
                # TODO: solve this  call, get run info is not available in MLflowLogger
                run_info = self.mlflow_logger.get_run_info()
                print(f"MLflow run ID: {run_info.get('run_id', 'N/A')}")
                print(f"View results at: {self.config.mlflow.tracking_uri}")

        except Exception as e:
            print(f"Training failed with error: {str(e)}")
            raise
        finally:
            # Always end MLflow run
            if self.mlflow_logger:
                self.mlflow_logger.end_run()


@hydra.main(version_base=None, config_path="../config", config_name="train_config")
def main(cfg: DictConfig) -> None:
    """Main function using Hydra for configuration management."""
    print("=" * 60)
    print("EV Charging System Training with MLflow")
    print("=" * 60)
    print(f"Configuration:\n{OmegaConf.to_yaml(cfg)}")

    # Initialize trainer
    trainer = EVChargingTrainer(cfg)

    # Run training
    trainer.run_training()


if __name__ == "__main__":
    main()