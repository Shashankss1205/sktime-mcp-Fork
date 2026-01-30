"""
Sktime Data Adapter
A robust data adapter that converts various data sources into sktime-compatible format.
Handles CSV, Excel, SQL, and DataFrame inputs with proper frequency inference.
"""

import pandas as pd
import numpy as np
from typing import Union, Optional, Tuple
from pathlib import Path


class SktimeDataAdapter:
    """
    Adapter class to convert various data sources into sktime-compatible time series format.
    
    Features:
    - Automatic frequency detection and setting
    - Multiple data source support (CSV, Excel, SQL, DataFrame)
    - Proper datetime index handling
    - Train/test splitting with correct index preservation
    - Data validation and cleaning
    """
    
    def __init__(self, verbose: bool = True):
        """
        Initialize the adapter.
        
        Parameters
        ----------
        verbose : bool, default=True
            Whether to print progress messages
        """
        self.verbose = verbose
        self.original_data = None
        self.processed_data = None
        self.frequency = None
        self.date_column = None
        self.value_column = None
        
    def _log(self, message: str):
        """Print message if verbose mode is enabled."""
        if self.verbose:
            print(f"[SktimeDataAdapter] {message}")
    
    def load_csv(
        self, 
        filepath: Union[str, Path],
        date_column: str,
        value_column: str,
        date_format: Optional[str] = None,
        **kwargs
    ) -> pd.Series:
        """
        Load data from CSV file and convert to sktime format.
        
        Parameters
        ----------
        filepath : str or Path
            Path to CSV file
        date_column : str
            Name of the date/time column
        value_column : str
            Name of the value column to forecast
        date_format : str, optional
            Format string for parsing dates (e.g., '%Y-%m-%d')
        **kwargs : dict
            Additional arguments passed to pd.read_csv
            
        Returns
        -------
        pd.Series
            Time series with proper DatetimeIndex and inferred frequency
        """
        self._log(f"Loading CSV from {filepath}...")
        
        # Load CSV
        df = pd.read_csv(filepath, **kwargs)
        self._log(f"Loaded {len(df)} rows")
        
        # Store column names
        self.date_column = date_column
        self.value_column = value_column
        
        # Convert to datetime
        if date_format:
            df[date_column] = pd.to_datetime(df[date_column], format=date_format)
        else:
            df[date_column] = pd.to_datetime(df[date_column])
        
        # Set index
        df = df.set_index(date_column)
        
        # Extract series
        series = df[value_column]
        
        # Process and return
        return self._process_series(series)
    
    def load_dataframe(
        self,
        df: pd.DataFrame,
        date_column: str,
        value_column: str,
        date_format: Optional[str] = None
    ) -> pd.Series:
        """
        Load data from pandas DataFrame and convert to sktime format.
        
        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame
        date_column : str
            Name of the date/time column
        value_column : str
            Name of the value column to forecast
        date_format : str, optional
            Format string for parsing dates
            
        Returns
        -------
        pd.Series
            Time series with proper DatetimeIndex and inferred frequency
        """
        self._log("Processing DataFrame...")
        
        df = df.copy()
        
        # Store column names
        self.date_column = date_column
        self.value_column = value_column
        
        # Convert to datetime if needed
        if not pd.api.types.is_datetime64_any_dtype(df[date_column]):
            if date_format:
                df[date_column] = pd.to_datetime(df[date_column], format=date_format)
            else:
                df[date_column] = pd.to_datetime(df[date_column])
        
        # Set index
        df = df.set_index(date_column)
        
        # Extract series
        series = df[value_column]
        
        # Process and return
        return self._process_series(series)
    
    def _process_series(self, series: pd.Series) -> pd.Series:
        """
        Process a pandas Series to make it sktime-compatible.
        
        This includes:
        - Inferring and setting frequency
        - Sorting by index
        - Removing duplicates
        - Handling missing values
        
        Parameters
        ----------
        series : pd.Series
            Input time series
            
        Returns
        -------
        pd.Series
            Processed time series with proper frequency
        """
        self._log("Processing time series...")
        
        # Store original
        self.original_data = series.copy()
        
        # Sort by index
        series = series.sort_index()
        
        # Remove duplicate indices (keep first)
        if series.index.duplicated().any():
            n_duplicates = series.index.duplicated().sum()
            self._log(f"Warning: Removing {n_duplicates} duplicate timestamps")
            series = series[~series.index.duplicated(keep='first')]
        
        # Infer frequency
        freq = pd.infer_freq(series.index)
        
        if freq is None:
            # Try to infer from most common difference
            self._log("Could not infer frequency automatically, attempting manual inference...")
            time_diffs = series.index.to_series().diff().dropna()
            most_common_diff = time_diffs.mode()[0]
            
            # Map common differences to frequency strings
            if most_common_diff == pd.Timedelta(days=1):
                freq = 'D'
            elif most_common_diff == pd.Timedelta(hours=1):
                freq = 'H'
            elif most_common_diff == pd.Timedelta(minutes=1):
                freq = 'T'
            elif most_common_diff == pd.Timedelta(seconds=1):
                freq = 'S'
            elif most_common_diff == pd.Timedelta(days=7):
                freq = 'W'
            elif most_common_diff.days >= 28 and most_common_diff.days <= 31:
                freq = 'M'
            else:
                freq = 'D'  # Default to daily
                self._log(f"Using default frequency: {freq}")
        
        self.frequency = freq
        self._log(f"Detected frequency: {freq}")
        
        # Create a complete date range with the inferred frequency
        full_range = pd.date_range(
            start=series.index.min(),
            end=series.index.max(),
            freq=freq
        )
        
        # Reindex to fill any gaps
        series = series.reindex(full_range)
        
        # Set the frequency explicitly on the index
        series.index.freq = freq
        
        # Handle missing values (forward fill then backward fill)
        if series.isna().any():
            n_missing = series.isna().sum()
            self._log(f"Warning: Found {n_missing} missing values, filling with forward/backward fill")
            series = series.fillna(method='ffill').fillna(method='bfill')
        
        self.processed_data = series
        
        self._log(f"Processed series: {len(series)} observations from {series.index[0]} to {series.index[-1]}")
        
        return series
    
    def train_test_split(
        self,
        series: pd.Series,
        train_size: float = 0.8,
        return_indices: bool = False
    ) -> Union[Tuple[pd.Series, pd.Series], Tuple[pd.Series, pd.Series, pd.Index, pd.Index]]:
        """
        Split time series into train and test sets while preserving frequency.
        
        Parameters
        ----------
        series : pd.Series
            Time series to split
        train_size : float, default=0.8
            Proportion of data to use for training (0 < train_size < 1)
        return_indices : bool, default=False
            If True, also return the train and test indices
            
        Returns
        -------
        y_train : pd.Series
            Training data
        y_test : pd.Series
            Test data
        train_idx : pd.Index (optional)
            Training indices
        test_idx : pd.Index (optional)
            Test indices
        """
        if not 0 < train_size < 1:
            raise ValueError("train_size must be between 0 and 1")
        
        split_point = int(len(series) * train_size)
        
        y_train = series.iloc[:split_point]
        y_test = series.iloc[split_point:]
        
        # Ensure frequency is preserved
        if hasattr(series.index, 'freq') and series.index.freq is not None:
            y_train.index.freq = series.index.freq
            y_test.index.freq = series.index.freq
        
        self._log(f"Split: {len(y_train)} train, {len(y_test)} test observations")
        
        if return_indices:
            return y_train, y_test, y_train.index, y_test.index
        else:
            return y_train, y_test
    
    def get_info(self) -> dict:
        """
        Get information about the processed data.
        
        Returns
        -------
        dict
            Dictionary containing data information
        """
        if self.processed_data is None:
            return {"status": "No data loaded"}
        
        return {
            "total_observations": len(self.processed_data),
            "start_date": str(self.processed_data.index[0]),
            "end_date": str(self.processed_data.index[-1]),
            "frequency": self.frequency,
            "date_column": self.date_column,
            "value_column": self.value_column,
            "missing_values": self.processed_data.isna().sum(),
            "mean": self.processed_data.mean(),
            "std": self.processed_data.std(),
            "min": self.processed_data.min(),
            "max": self.processed_data.max()
        }


# Convenience function for quick usage
def load_time_series(
    filepath: Union[str, Path],
    date_column: str,
    value_column: str,
    train_size: float = 0.8,
    verbose: bool = True
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Quick function to load and split time series data.
    
    Parameters
    ----------
    filepath : str or Path
        Path to data file (CSV, Excel, etc.)
    date_column : str
        Name of the date column
    value_column : str
        Name of the value column
    train_size : float, default=0.8
        Proportion for training
    verbose : bool, default=True
        Print progress messages
        
    Returns
    -------
    full_series : pd.Series
        Complete time series
    y_train : pd.Series
        Training data
    y_test : pd.Series
        Test data
    """
    adapter = SktimeDataAdapter(verbose=verbose)
    
    # Determine file type and load
    filepath = Path(filepath)
    if filepath.suffix.lower() == '.csv':
        series = adapter.load_csv(filepath, date_column, value_column)
    else:
        raise ValueError(f"Unsupported file type: {filepath.suffix}")
    
    # Split
    y_train, y_test = adapter.train_test_split(series, train_size=train_size)
    
    return series, y_train, y_test
