"""
Data loader for EV charging system.
Handles loading and preprocessing of training and test datasets.
"""

import os
from pathlib import Path
from typing import Dict, Any, List, Optional

import pandas as pd
import numpy as np
from omegaconf import DictConfig


class DataLoader:
    """Handles loading and preprocessing of EV charging data."""

    def __init__(self, config: DictConfig):
        self.config = config
        self.data_path = Path(config.data.path)
        self.drop_columns = config.data.get('drop_columns', [])

    def load_all_data(self) -> Dict[str, pd.DataFrame]:
        """
        Load all required datasets for training and evaluation.

        Returns:
            Dictionary containing all loaded dataframes
        """
        data = {}

        # Load EV training data
        train_file = self.data_path / self.config.data.files.df_evs_train
        data['df_evs_train'] = self._load_excel_file(train_file, 'EV training data')

        # Load EV test data
        test_file = self.data_path / self.config.data.files.df_evs_test
        data['df_evs_test'] = self._load_excel_file(test_file, 'EV test data')

        # Load building data
        building_file = self.data_path / self.config.data.files.df_build
        data['df_build'] = self._load_excel_file(building_file, 'Building data')

        # Preprocess data
        data = self._preprocess_data(data)

        # Validate data
        self._validate_data(data)

        return data

    def _load_excel_file(self, file_path: Path, description: str) -> pd.DataFrame:
        """
        Load a single Excel file with error handling.

        Args:
            file_path: Path to the Excel file
            description: Description for logging

        Returns:
            Loaded DataFrame
        """
        try:
            if not file_path.exists():
                raise FileNotFoundError(f"{description} file not found: {file_path}")

            df = pd.read_excel(file_path)
            print(f"Loaded {description}: {df.shape} rows x {df.shape[1]} columns")
            return df

        except Exception as e:
            raise RuntimeError(f"Failed to load {description} from {file_path}: {str(e)}")

    def _preprocess_data(self, data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """
        Preprocess loaded data (remove unnecessary columns, handle missing values, etc.).

        Args:
            data: Dictionary of loaded DataFrames

        Returns:
            Dictionary of preprocessed DataFrames
        """
        preprocessed_data = {}

        for key, df in data.items():
            # Create a copy to avoid modifying original
            processed_df = df.copy()

            # Remove specified columns if they exist
            columns_to_drop = [col for col in self.drop_columns if col in processed_df.columns]
            if columns_to_drop:
                processed_df = processed_df.drop(columns=columns_to_drop)
                print(f"Dropped columns from {key}: {columns_to_drop}")

            # Handle missing values
            processed_df = self._handle_missing_values(processed_df, key)

            # Apply specific preprocessing based on data type
            processed_df = self._apply_specific_preprocessing(processed_df, key)

            preprocessed_data[key] = processed_df

        return preprocessed_data

    def _handle_missing_values(self, df: pd.DataFrame, data_key: str) -> pd.DataFrame:
        """
        Handle missing values in the dataset.

        Args:
            df: DataFrame to process
            data_key: Key identifying the dataset type

        Returns:
            DataFrame with missing values handled
        """
        if df.isnull().sum().sum() == 0:
            return df

        print(f"Handling missing values in {data_key}:")

        # Log missing values
        missing_info = df.isnull().sum()
        missing_info = missing_info[missing_info > 0]

        for col, count in missing_info.items():
            print(f"  {col}: {count} missing values")

        # Apply different strategies based on data type
        if data_key in ['df_evs_train', 'df_evs_test']:
            # For EV data, fill numeric columns with median, categorical with mode
            for col in df.columns:
                if df[col].dtype in ['float64', 'int64']:
                    df[col].fillna(df[col].median(), inplace=True)
                else:
                    df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 'Unknown', inplace=True)

        elif data_key == 'df_build':
            # For building data, use forward fill for time series
            df.fillna(method='ffill', inplace=True)
            df.fillna(method='bfill', inplace=True)  # Handle any remaining NaN at the beginning

        return df

    def _apply_specific_preprocessing(self, df: pd.DataFrame, data_key: str) -> pd.DataFrame:
        """
        Apply dataset-specific preprocessing.

        Args:
            df: DataFrame to process
            data_key: Key identifying the dataset type

        Returns:
            Preprocessed DataFrame
        """
        if data_key in ['df_evs_train', 'df_evs_test']:
            # EV data preprocessing
            df = self._preprocess_ev_data(df)

        elif data_key == 'df_build':
            # Building data preprocessing
            df = self._preprocess_building_data(df)

        return df

    def _preprocess_ev_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess EV data specifically.

        Args:
            df: EV DataFrame

        Returns:
            Preprocessed EV DataFrame
        """
        # Ensure SOC is within valid range [0, 1]
        if 'SOC' in df.columns:
            df['SOC'] = df['SOC'].clip(0, 1)

        # Ensure arrival time is within valid range [0, 23]
        if 'Arrival_time[h]' in df.columns:
            df['Arrival_time[h]'] = df['Arrival_time[h]'].clip(0, 23)

        # Ensure time until departure is positive
        if 'TuD (int)' in df.columns:
            df['TuD (int)'] = df['TuD (int)'].clip(lower=1)

        # Ensure battery capacity is positive
        if 'Battery capacity [KWh]' in df.columns:
            df['Battery capacity [KWh]'] = df['Battery capacity [KWh]'].clip(lower=1)

        # Ensure energy needed is non-negative
        if 'ENonD' in df.columns:
            df['ENonD'] = df['ENonD'].clip(lower=0)

        return df

    def _preprocess_building_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess building data specifically.

        Args:
            df: Building DataFrame

        Returns:
            Preprocessed building DataFrame
        """
        # Ensure surplus power is non-negative
        if 'surplus_power[kw]' in df.columns:
            df['surplus_power[kw]'] = df['surplus_power[kw]'].clip(lower=0)

        return df

    def _validate_data(self, data: Dict[str, pd.DataFrame]) -> None:
        """
        Validate loaded and preprocessed data.

        Args:
            data: Dictionary of DataFrames to validate
        """
        required_datasets = ['df_evs_train', 'df_evs_test', 'df_build']

        # Check all required datasets are present
        for dataset in required_datasets:
            if dataset not in data:
                raise ValueError(f"Required dataset missing: {dataset}")

        # Validate EV data structure
        self._validate_ev_data(data['df_evs_train'], 'Training EV data')
        self._validate_ev_data(data['df_evs_test'], 'Test EV data')

        # Validate building data structure
        self._validate_building_data(data['df_build'])

        print("Data validation completed successfully!")

    def _validate_ev_data(self, df: pd.DataFrame, name: str) -> None:
        """
        Validate EV data structure and content.

        Args:
            df: EV DataFrame to validate
            name: Name for logging
        """
        required_columns = ['Arrival_time[h]', 'TuD (int)', 'Battery capacity [KWh]', 'ENonD', 'SOC']

        # Check required columns
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"{name} missing required columns: {missing_columns}")

        # Check data ranges
        if 'SOC' in df.columns:
            if not df['SOC'].between(0, 1).all():
                raise ValueError(f"{name}: SOC values must be between 0 and 1")

        if 'Arrival_time[h]' in df.columns:
            if not df['Arrival_time[h]'].between(0, 23).all():
                raise ValueError(f"{name}: Arrival time must be between 0 and 23")

        print(f"{name} validation passed")

    def _validate_building_data(self, df: pd.DataFrame) -> None:
        """
        Validate building data structure and content.

        Args:
            df: Building DataFrame to validate
        """
        required_columns = ['surplus_power[kw]']

        # Check required columns
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Building data missing required columns: {missing_columns}")

        # Check that we have 24 hours of data
        if len(df) != 24:
            raise ValueError(f"Building data must have exactly 24 hours, got {len(df)}")

        # Check surplus power is non-negative
        if 'surplus_power[kw]' in df.columns:
            if (df['surplus_power[kw]'] < 0).any():
                raise ValueError("Surplus power cannot be negative")

        print("Building data validation passed")

    def get_data_summary(self, data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """
        Get summary statistics for loaded data.

        Args:
            data: Dictionary of DataFrames

        Returns:
            Dictionary with summary statistics
        """
        summary = {}

        for key, df in data.items():
            summary[key] = {
                'shape': df.shape,
                'columns': list(df.columns),
                'dtypes': df.dtypes.to_dict(),
                'missing_values': df.isnull().sum().to_dict(),
                'numeric_summary': df.describe().to_dict() if len(df.select_dtypes(include=[np.number]).columns) > 0 else {}
            }

        return summary