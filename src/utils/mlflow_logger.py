"""
MLflow logging utility for EV charging system.
Updated to handle cleaner run/experiment naming strategy.
"""

import os
import tempfile
from pathlib import Path
from typing import Dict, Any, Optional, Union
import json
import pickle

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf
from stable_baselines3.common.base_class import BaseAlgorithm


class MLflowLogger:
    """MLflow experiment tracking and logging utility."""

    def __init__(self, config: DictConfig):
        self.config = config
        self.mlflow_config = config.mlflow
        self.run_config = config.run
        self.run_id = None
        self.experiment_id = None

        if self.mlflow_config.enabled:
            self._setup_mlflow()

    def _setup_mlflow(self):
        """Initialize MLflow tracking."""
        # Set tracking URI
        mlflow.set_tracking_uri(self.mlflow_config.tracking_uri)

        # Create or get experiment
        try:
            experiment = mlflow.get_experiment_by_name(self.mlflow_config.experiment_name)
            if experiment is None:
                self.experiment_id = mlflow.create_experiment(
                    name=self.mlflow_config.experiment_name,
                    tags={
                        "project": "ev_charging_system",
                        "version": "1.0",
                        "framework": "stable-baselines3"
                    }
                )
                print(f"Created new MLflow experiment: {self.mlflow_config.experiment_name}")
            else:
                self.experiment_id = experiment.experiment_id
                print(f"Using existing MLflow experiment: {self.mlflow_config.experiment_name}")
        except Exception as e:
            print(f"Warning: Could not setup MLflow experiment: {e}")
            self.experiment_id = "0"  # Default experiment

    def _generate_run_name(self) -> str:
        """Generate run name based on configuration strategy."""
        # Strategy 1: Use run.name from config (recommended)
        if self.mlflow_config.get('use_run_name_from_config', True):
            base_name = self.run_config.name

            # Add timestamp if configured
            if self.run_config.get('timestamp', False):
                import time
                timestamp = int(time.time())
                return f"{timestamp}_{base_name}"
            else:
                return base_name

        # Strategy 2: Use custom name from MLflow config
        elif hasattr(self.mlflow_config, 'custom_run_name'):
            return self.mlflow_config.custom_run_name

        # Strategy 3: Auto-generate descriptive name
        else:
            env_type = self.config.env.type
            algorithm = self.config.model.algorithm
            n_envs = self.config.env.n_environments
            return f"{algorithm}_{env_type}_{n_envs}envs"

    def start_run(self, custom_run_name: Optional[str] = None) -> str:
        """Start a new MLflow run with improved naming."""
        if not self.mlflow_config.enabled:
            return None

        try:
            # Generate run name
            run_name = custom_run_name or self._generate_run_name()

            # Create comprehensive tags
            tags = {
                "environment_type": self.config.env.type,
                "algorithm": self.config.model.algorithm,
                "n_environments": str(self.config.env.n_environments),
                "env_seed": str(self.config.env.seed),
                "framework": "stable-baselines3",
                "project_phase": "training"
            }

            # Add description if available
            if hasattr(self.run_config, 'description'):
                tags["description"] = self.run_config.description

            # Start run
            mlflow.start_run(
                experiment_id=self.experiment_id,
                run_name=run_name,
                tags=tags
            )

            self.run_id = mlflow.active_run().info.run_id
            print(f" Started MLflow run: '{run_name}'")
            print(f"   Run ID: {self.run_id}")
            print(f"   Experiment: {self.mlflow_config.experiment_name}")
            print(f"   Tracking URI: {self.mlflow_config.tracking_uri}")

            # Log configuration
            if self.mlflow_config.get('log_params', True):
                self._log_config()

            return self.run_id

        except Exception as e:
            print(f"Warning: Could not start MLflow run: {e}")
            return None

    def _log_config(self):
        """Log configuration parameters with better organization."""
        try:
            # Organize config into logical groups
            config_groups = {
                "environment": {
                    "env_type": self.config.env.type,
                    "n_environments": self.config.env.n_environments,
                    "seed": self.config.env.seed
                },
                "model": {
                    "algorithm": self.config.model.algorithm,
                    "policy": self.config.model.policy,
                    "verbose": self.config.model.verbose
                },
                "training": {
                    "total_timesteps": self.config.training.total_timesteps,
                    "timesteps_per_save": self.config.training.timesteps_per_save,
                    "normalize_obs": self.config.training.normalize.norm_obs,
                    "normalize_reward": self.config.training.normalize.norm_reward
                },
                "data": {
                    "data_path": self.config.data.path,
                    "train_file": self.config.data.files.df_evs_train,
                    "test_file": self.config.data.files.df_evs_test,
                    "building_file": self.config.data.files.df_build
                }
            }

            # Add algorithm-specific parameters
            algorithm = self.config.model.algorithm.lower()
            if algorithm in self.config.model:
                algo_params = OmegaConf.to_container(self.config.model[algorithm])
                config_groups[f"{algorithm}_params"] = algo_params

            # Flatten and log all parameters
            flat_params = {}
            for group_name, group_params in config_groups.items():
                for key, value in group_params.items():
                    param_name = f"{group_name}.{key}"
                    flat_params[param_name] = str(value)

            mlflow.log_params(flat_params)
            print(f" Logged {len(flat_params)} configuration parameters")

        except Exception as e:
            print(f"Warning: Could not log config: {e}")

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """Log metrics to MLflow with better error handling."""
        if not self.mlflow_config.enabled or not self.mlflow_config.get('log_metrics', True):
            return

        try:
            logged_count = 0
            for key, value in metrics.items():
                if isinstance(value, (int, float, np.number)) and not np.isnan(value):
                    mlflow.log_metric(key, float(value), step=step)
                    logged_count += 1

            if logged_count > 0:
                print(f" Logged {logged_count} metrics" + (f" at step {step}" if step else ""))

        except Exception as e:
            print(f"Warning: Could not log metrics: {e}")

    def log_training_metrics(self,
                             episode: int,
                             reward: float,
                             power_usage: float,
                             soc_success_rate: float,
                             violations: int):
        """Log training-specific metrics with descriptive names."""
        metrics = {
            "training/episode_reward": reward,
            "training/power_usage_kw": power_usage,
            "training/soc_success_rate": soc_success_rate,
            "training/constraint_violations": violations,
            "training/episode": episode
        }
        self.log_metrics(metrics, step=episode)

    def log_evaluation_results(self, results: Dict[str, Any]):
        """Log comprehensive evaluation results with better organization."""
        if not self.mlflow_config.enabled:
            return

        try:
            # Organize metrics by category
            final_metrics = {
                # Performance metrics
                "evaluation/final_mean_reward": results.get("mean_total_reward", 0),
                "evaluation/reward_std": results.get("std_total_reward", 0),
                "evaluation/min_reward": results.get("min_total_reward", 0),
                "evaluation/max_reward": results.get("max_total_reward", 0),

                # SOC metrics
                "evaluation/soc_success_rate": results.get("soc_success_rate", 0),
                "evaluation/mean_final_soc": results.get("mean_final_soc", 0),
                "evaluation/soc_deficit": results.get("mean_soc_deficit", 0),
                "evaluation/evs_with_deficit": results.get("evs_with_deficit", 0),

                # Power metrics
                "evaluation/power_utilization": results.get("power_utilization_ratio", 0),
                "evaluation/power_violations": results.get("power_violations", 0),
                "evaluation/violation_rate": results.get("power_violation_rate", 0),
                "evaluation/charge_variance": results.get("charge_power_variance", 0),

                # Energy metrics
                "evaluation/total_energy_delivered": results.get("total_energy_delivered", 0),
                "evaluation/mean_soc_increase": results.get("mean_soc_increase", 0)
            }

            self.log_metrics(final_metrics)

            # Log summary statistics
            summary_stats = {
                "summary/total_evs": results.get("total_evs", 0),
                "summary/successful_evs": results.get("evs_meeting_requirements", 0),
                "summary/mean_power_usage": results.get("mean_power_usage", 0),
                "summary/mean_available_power": results.get("mean_available_power", 0)
            }

            self.log_metrics(summary_stats)

            # Log detailed results as artifacts
            if self.mlflow_config.get('log_artifacts', True):
                self._log_evaluation_artifacts(results)

        except Exception as e:
            print(f"Warning: Could not log evaluation results: {e}")

    def _log_evaluation_artifacts(self, results: Dict[str, Any]):
        """Log evaluation results as artifacts with better organization."""
        try:
            with tempfile.TemporaryDirectory() as tmp_dir:
                tmp_path = Path(tmp_dir)

                # Create organized subdirectories
                (tmp_path / "metrics").mkdir()
                (tmp_path / "data").mkdir()

                # Save summary metrics as JSON
                summary_metrics = {
                    "performance": {
                        "mean_reward": results.get("mean_total_reward", 0),
                        "std_reward": results.get("std_total_reward", 0),
                        "success_rate": results.get("soc_success_rate", 0)
                    },
                    "power": {
                        "utilization_ratio": results.get("power_utilization_ratio", 0),
                        "violations": results.get("power_violations", 0),
                        "avg_usage": results.get("mean_power_usage", 0)
                    }
                }

                with open(tmp_path / "metrics" / "summary.json", 'w') as f:
                    json.dump(summary_metrics, f, indent=2)

                # Save DataFrames as CSV with descriptive names
                dataframe_mapping = {
                    'charge_pwr': 'charging_power_timeseries.csv',
                    'SOC_mat': 'soc_matrix.csv',
                    'df_evs_out': 'ev_final_states.csv',
                    'total_reward': 'environment_rewards.csv'
                }

                for key, filename in dataframe_mapping.items():
                    if key in results and isinstance(results[key], pd.DataFrame):
                        results[key].to_csv(tmp_path / "data" / filename, index=False)

                # Log all artifacts
                mlflow.log_artifacts(str(tmp_path), "evaluation_results")
                print(f" Logged evaluation artifacts to MLflow")

        except Exception as e:
            print(f"Warning: Could not log evaluation artifacts: {e}")

    def log_model(self, model: BaseAlgorithm, model_name: str = "rl_model"):
        """Log trained model to MLflow with versioning."""
        if not self.mlflow_config.enabled or not self.mlflow_config.get('log_models', True):
            return

        try:
            with tempfile.TemporaryDirectory() as tmp_dir:
                model_path = Path(tmp_dir) / f"{model_name}.zip"
                model.save(str(model_path))

                # Log model as artifact with metadata
                mlflow.log_artifact(str(model_path), "models")

                # Log model metadata
                model_metadata = {
                    "model_type": model.__class__.__name__,
                    "policy_type": model.policy.__class__.__name__,
                    "total_timesteps": getattr(model, 'num_timesteps', 0)
                }

                with open(Path(tmp_dir) / "model_metadata.json", 'w') as f:
                    json.dump(model_metadata, f, indent=2)

                mlflow.log_artifact(str(Path(tmp_dir) / "model_metadata.json"), "models")

                print(f" Logged model '{model_name}' to MLflow")

                # Register model if configured
                if (self.mlflow_config.get('model_registry', {}).get('enabled', False)):
                    self._register_model(model_path, model_name)

        except Exception as e:
            print(f"Warning: Could not log model: {e}")

    def end_run(self):
        """End the current MLflow run with summary."""
        if not self.mlflow_config.enabled:
            return

        try:
            if self.run_id:
                # Log final run summary
                run_summary = {
                    "run_status": "FINISHED",
                    "total_duration_minutes": 0  # Could calculate actual duration
                }
                self.log_metrics(run_summary)

                print(f" Completed MLflow run: {self.run_id}")
                print(f"   View at: {self.mlflow_config.tracking_uri}")

            mlflow.end_run()

        except Exception as e:
            print(f"Warning: Could not end MLflow run: {e}")

    def get_experiment_runs(self) -> pd.DataFrame:
        """Get all runs from the current experiment."""
        if not self.mlflow_config.enabled:
            return pd.DataFrame()

        try:
            client = mlflow.MlflowClient()
            runs = client.search_runs(
                experiment_ids=[self.experiment_id],
                max_results=100
            )

            # Convert to DataFrame for easy analysis
            runs_data = []
            for run in runs:
                run_data = {
                    "run_id": run.info.run_id,
                    "run_name": run.info.run_name,
                    "status": run.info.status,
                    "start_time": run.info.start_time,
                    "end_time": run.info.end_time
                }

                # Add key metrics
                for key, value in run.data.metrics.items():
                    run_data[f"metric_{key}"] = value

                # Add key parameters
                for key, value in run.data.params.items():
                    run_data[f"param_{key}"] = value

                runs_data.append(run_data)

            return pd.DataFrame(runs_data)

        except Exception as e:
            print(f"Warning: Could not get experiment runs: {e}")
            return pd.DataFrame()