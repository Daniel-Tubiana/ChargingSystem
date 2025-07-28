"""
Evaluator for EV charging system models.
Handles model evaluation, metrics calculation, and results analysis.
"""

from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import pandas as pd
from omegaconf import DictConfig
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.vec_env import VecNormalize


class Evaluator:
    """Handles evaluation of trained RL models."""

    def __init__(self, config: DictConfig):
        self.config = config
        self.n_hours = 24

    def evaluate_model(self,
                       model: BaseAlgorithm,
                       env: VecNormalize,
                       data: Dict[str, pd.DataFrame],
                       run_name: str) -> Dict[str, Any]:
        """
        Evaluate a trained model and collect comprehensive results.

        Args:
            model: Trained RL model
            env: Evaluation environment
            data: Dictionary containing test data
            run_name: Name of the current run

        Returns:
            Dictionary containing evaluation results
        """
        print("Starting model evaluation...")

        # Initialize result containers
        n_envs = self.config.env.n_environments
        charge_power_matrix = np.zeros((self.n_hours - 1, n_envs))  # 23 hours
        soc_matrix = np.zeros((len(data['df_evs_test']), n_envs))
        total_rewards = []
        episode_info = []

        # Reset environment
        obs = env.reset()
        original_obs = env.get_original_obs()

        # Run evaluation for 23 hours (24 steps total, but we collect 23 hours of data)
        for hour in range(self.n_hours - 1):
            # Get action from model
            actions, _states = model.predict(original_obs, deterministic=True)

            # Step environment
            obs, rewards, dones, infos = env.step(actions)
            original_obs = env.get_original_obs()

            # Collect data for each environment
            for env_idx in range(n_envs):
                # Store charge power
                charge_power_matrix[hour, env_idx] = infos[env_idx].get('charge_pwr', 0)

                # Store episode info for first hour only (to avoid duplication)
                if hour == 0:
                    total_rewards.append(infos[env_idx].get('total_reward', 0))

            # Log progress
            if hour % 5 == 0 or hour == self.n_hours - 2:
                avg_charge = np.mean(charge_power_matrix[hour, :])
                available_power = original_obs['power_allowed'][0][hour] if hour < len(
                    original_obs['power_allowed'][0]) else 0
                print(f"Hour {hour + 1:2d}: Avg charge power: {avg_charge:6.2f} kW, "
                      f"Available: {available_power:6.2f} kW")

        # Collect final SOC values
        final_obs = env.get_original_obs()
        for env_idx in range(n_envs):
            soc_matrix[:, env_idx] = final_obs['evs'][env_idx][:, 4]  # SOC is the 5th column (index 4)

        # Update total rewards with final values
        total_rewards = []
        for env_idx in range(n_envs):
            total_rewards.append(infos[env_idx].get('total_reward', 0))

        # Create results dictionary
        results = self._compile_results(
            charge_power_matrix=charge_power_matrix,
            soc_matrix=soc_matrix,
            total_rewards=total_rewards,
            test_data=data['df_evs_test'],
            building_data=data['df_build'],
            run_name=run_name
        )

        # Calculate and display summary metrics
        summary = self._calculate_summary_metrics(results)
        self._display_summary(summary)

        return results

    def _compile_results(self,
                         charge_power_matrix: np.ndarray,
                         soc_matrix: np.ndarray,
                         total_rewards: List[float],
                         test_data: pd.DataFrame,
                         building_data: pd.DataFrame,
                         run_name: str) -> Dict[str, pd.DataFrame]:
        """
        Compile evaluation results into structured DataFrames.

        Args:
            charge_power_matrix: Matrix of charge powers [hours x environments]
            soc_matrix: Matrix of final SOC values [evs x environments]
            total_rewards: List of total rewards per environment
            test_data: Original test dataset
            building_data: Building power data
            run_name: Name of the current run

        Returns:
            Dictionary of result DataFrames
        """
        results = {}

        # 1. Charge power DataFrame
        charge_df = pd.DataFrame(charge_power_matrix)
        # Add surplus power column for comparison
        surplus_power = building_data['surplus_power[kw]'].iloc[:len(charge_df)].values
        charge_df['surplusPWR'] = surplus_power
        results['charge_pwr'] = charge_df

        # 2. SOC matrix DataFrame
        results['SOC_mat'] = pd.DataFrame(soc_matrix)

        # 3. EV output DataFrame (combining initial data with final results)
        df_evs_out = pd.DataFrame()

        # Add initial SOC
        df_evs_out['initial_SOC'] = test_data['SOC'].values

        # Add EV characteristics from first environment (they should be the same across environments)
        ev_features = ['Arrival_time[h]', 'TuD (int)', 'Battery capacity [KWh]', 'ENonD', 'SOC']
        for i, feature in enumerate(ev_features):
            if i == 4:  # SOC is the final SOC, not initial
                df_evs_out['final_SOC'] = soc_matrix[:, 0]  # Use first environment as representative
            else:
                df_evs_out[feature] = test_data[feature].values

        # Calculate additional metrics
        df_evs_out['SOC_increase'] = df_evs_out['final_SOC'] - df_evs_out['initial_SOC']

        # Calculate if energy requirements were met
        min_soc = 0.2
        df_evs_out['required_SOC'] = (df_evs_out['ENonD'] / df_evs_out['Battery capacity [KWh]']) + min_soc
        df_evs_out['requirement_met'] = df_evs_out['final_SOC'] >= df_evs_out['required_SOC']
        df_evs_out['SOC_deficit'] = np.maximum(0, df_evs_out['required_SOC'] - df_evs_out['final_SOC'])

        results['df_evs_out'] = df_evs_out

        # 4. Total rewards DataFrame
        total_reward_df = pd.DataFrame({'total_reward': total_rewards})
        results['total_reward'] = total_reward_df

        return results

    def _calculate_summary_metrics(self, results: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """
        Calculate summary metrics from evaluation results.

        Args:
            results: Dictionary of result DataFrames

        Returns:
            Dictionary of summary metrics
        """
        charge_df = results['charge_pwr']
        soc_df = results['SOC_mat']
        evs_df = results['df_evs_out']
        rewards_df = results['total_reward']

        summary = {}

        # Reward metrics
        summary['mean_total_reward'] = rewards_df['total_reward'].mean()
        summary['std_total_reward'] = rewards_df['total_reward'].std()
        summary['min_total_reward'] = rewards_df['total_reward'].min()
        summary['max_total_reward'] = rewards_df['total_reward'].max()

        # Power utilization metrics
        charge_columns = [col for col in charge_df.columns if col != 'surplusPWR']
        mean_charge_power = charge_df[charge_columns].mean(axis=1)
        surplus_power = charge_df['surplusPWR']

        summary['mean_power_usage'] = mean_charge_power.mean()
        summary['mean_available_power'] = surplus_power.mean()
        summary['power_utilization_ratio'] = summary['mean_power_usage'] / summary['mean_available_power'] if summary[
                                                                                                                  'mean_available_power'] > 0 else 0

        # Power constraint violations
        power_violations = (mean_charge_power > surplus_power).sum()
        summary['power_violations'] = power_violations
        summary['power_violation_rate'] = power_violations / len(mean_charge_power)

        # Charge power variance (consistency metric)
        summary['charge_power_variance'] = charge_df[charge_columns].var(axis=1).mean()

        # SOC achievement metrics
        summary['evs_meeting_requirements'] = evs_df['requirement_met'].sum()
        summary['total_evs'] = len(evs_df)
        summary['soc_success_rate'] = summary['evs_meeting_requirements'] / summary['total_evs']

        # SOC statistics
        summary['mean_final_soc'] = evs_df['final_SOC'].mean()
        summary['std_final_soc'] = evs_df['final_SOC'].std()
        summary['min_final_soc'] = evs_df['final_SOC'].min()
        summary['max_final_soc'] = evs_df['final_SOC'].max()

        # SOC deficit metrics
        summary['mean_soc_deficit'] = evs_df['SOC_deficit'].mean()
        summary['evs_with_deficit'] = (evs_df['SOC_deficit'] > 0).sum()

        # Energy efficiency metrics
        summary['mean_soc_increase'] = evs_df['SOC_increase'].mean()
        summary['total_energy_delivered'] = (evs_df['SOC_increase'] * evs_df['Battery capacity [KWh]']).sum()

        return summary

    def _display_summary(self, summary: Dict[str, Any]) -> None:
        """
        Display evaluation summary in a formatted way.

        Args:
            summary: Dictionary of summary metrics
        """
        print("\n" + "=" * 60)
        print("EVALUATION SUMMARY")
        print("=" * 60)

        print(f"\n📊 REWARD METRICS:")
        print(f"  Mean Total Reward:     {summary['mean_total_reward']:10.2f}")
        print(f"  Std Total Reward:      {summary['std_total_reward']:10.2f}")
        print(f"  Reward Range:          [{summary['min_total_reward']:8.2f}, {summary['max_total_reward']:8.2f}]")

        print(f"\n⚡ POWER UTILIZATION:")
        print(f"  Mean Power Usage:      {summary['mean_power_usage']:10.2f} kW")
        print(f"  Mean Available Power:  {summary['mean_available_power']:10.2f} kW")
        print(f"  Utilization Ratio:     {summary['power_utilization_ratio']:10.2%}")
        print(f"  Power Violations:      {summary['power_violations']:10d} ({summary['power_violation_rate']:.1%})")
        print(f"  Charge Variance:       {summary['charge_power_variance']:10.2f}")

        print(f"\n🔋 SOC PERFORMANCE:")
        print(
            f"  EVs Meeting Req.:      {summary['evs_meeting_requirements']:4d}/{summary['total_evs']:4d} ({summary['soc_success_rate']:.1%})")
        print(f"  Mean Final SOC:        {summary['mean_final_soc']:10.2%}")
        print(f"  SOC Range:             [{summary['min_final_soc']:8.2%}, {summary['max_final_soc']:8.2%}]")
        print(f"  Mean SOC Deficit:      {summary['mean_soc_deficit']:10.4f}")
        print(f"  EVs with Deficit:      {summary['evs_with_deficit']:10d}")

        print(f"\n🔄 ENERGY EFFICIENCY:")
        print(f"  Mean SOC Increase:     {summary['mean_soc_increase']:10.2%}")
        print(f"  Total Energy Delivered: {summary['total_energy_delivered']:9.2f} kWh")

        print("=" * 60 + "\n")

    def save_detailed_analysis(self, results: Dict[str, pd.DataFrame], save_path: str) -> None:
        """
        Save detailed analysis results to files.

        Args:
            results: Dictionary of result DataFrames
            save_path: Base path for saving files
        """
        try:
            # Save each DataFrame to a separate sheet in Excel file
            with pd.ExcelWriter(f"{save_path}_detailed_analysis.xlsx", engine='xlsxwriter') as writer:
                for sheet_name, df in results.items():
                    df.to_excel(writer, sheet_name=sheet_name, index=False)

                # Add summary sheet
                summary = self._calculate_summary_metrics(results)
                summary_df = pd.DataFrame(list(summary.items()), columns=['Metric', 'Value'])
                summary_df.to_excel(writer, sheet_name='Summary', index=False)

            print(f"Detailed analysis saved to: {save_path}_detailed_analysis.xlsx")

        except Exception as e:
            print(f"Failed to save detailed analysis: {str(e)}")

    def compare_runs(self, results1: Dict[str, Any], results2: Dict[str, Any],
                     run_name1: str, run_name2: str) -> Dict[str, Any]:
        """
        Compare results from two different runs.

        Args:
            results1: Results from first run
            results2: Results from second run
            run_name1: Name of first run
            run_name2: Name of second run

        Returns:
            Dictionary with comparison results
        """
        summary1 = self._calculate_summary_metrics(results1)
        summary2 = self._calculate_summary_metrics(results2)

        comparison = {
            'run_names': [run_name1, run_name2],
            'metrics_comparison': {},
            'improvements': {},
            'regressions': {}
        }

        # Compare each metric
        for metric in summary1.keys():
            if metric in summary2:
                val1 = summary1[metric]
                val2 = summary2[metric]
                comparison['metrics_comparison'][metric] = {
                    run_name1: val1,
                    run_name2: val2,
                    'difference': val2 - val1,
                    'percent_change': ((val2 - val1) / val1 * 100) if val1 != 0 else float('inf')
                }

                # Determine if it's an improvement or regression
                # (This is context-dependent, so we'll use simple heuristics)
                if 'reward' in metric.lower() or 'success' in metric.lower() or 'utilization' in metric.lower():
                    # Higher is better
                    if val2 > val1:
                        comparison['improvements'][metric] = comparison['metrics_comparison'][metric]
                    elif val2 < val1:
                        comparison['regressions'][metric] = comparison['metrics_comparison'][metric]
                elif 'violation' in metric.lower() or 'deficit' in metric.lower() or 'variance' in metric.lower():
                    # Lower is better
                    if val2 < val1:
                        comparison['improvements'][metric] = comparison['metrics_comparison'][metric]
                    elif val2 > val1:
                        comparison['regressions'][metric] = comparison['metrics_comparison'][metric]

        return comparison