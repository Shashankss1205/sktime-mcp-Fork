"""
Pandas adapter for in-memory DataFrames.

Supports loading data from pandas DataFrames with automatic
time index detection and validation.
"""

import pandas as pd
from typing import Any, Dict, Tuple
from ..base import DataSourceAdapter


class PandasAdapter(DataSourceAdapter):
    """
    Adapter for in-memory pandas DataFrames.
    
    Config example:
    {
        "type": "pandas",
        "data": <DataFrame object or dict>,
        "time_column": "date",  # optional, will try to detect
        "target_column": "value",  # optional, defaults to first column
        "exog_columns": ["feature1", "feature2"],  # optional
        "frequency": "D"  # optional, will try to infer
    }
    """
    
    def load(self) -> pd.DataFrame:
        """Load from in-memory DataFrame or dict."""
        data = self.config.get("data")
        
        if data is None:
            raise ValueError("Config must contain 'data' key")
        
        # Convert dict to DataFrame if needed
        if isinstance(data, dict):
            df = pd.DataFrame(data)
        elif isinstance(data, pd.DataFrame):
            df = data.copy()
        else:
            raise ValueError(f"Data must be a pandas DataFrame or dict, got {type(data)}")
        
        # Set time index
        time_col = self.config.get("time_column")
        
        if time_col:
            # User specified time column
            if time_col not in df.columns:
                raise ValueError(f"Time column '{time_col}' not found in data")
            df = df.set_index(time_col)
        elif not isinstance(df.index, pd.DatetimeIndex):
            # Try to detect time column
            time_col = self._detect_time_column(df)
            if time_col:
                df = df.set_index(time_col)
        
        # Ensure datetime index
        if not isinstance(df.index, pd.DatetimeIndex):
            try:
                df.index = pd.to_datetime(df.index)
            except Exception as e:
                raise ValueError(f"Could not convert index to datetime: {e}")
        
        # Sort by time
        df = df.sort_index()
        
        # Infer or set frequency
        freq = self.config.get("frequency")
        if freq:
            df = df.asfreq(freq)
        elif df.index.freq is None:
            # Try to infer frequency
            inferred_freq = pd.infer_freq(df.index)
            if inferred_freq:
                df = df.asfreq(inferred_freq)
        
        self._data = df
        self._metadata = {
            "source": "pandas",
            "rows": len(df),
            "columns": list(df.columns),
            "frequency": str(df.index.freq) if df.index.freq else pd.infer_freq(df.index),
            "start_date": str(df.index.min()),
            "end_date": str(df.index.max()),
            "missing_values": df.isnull().sum().to_dict(),
        }
        
        return df
    
    def _detect_time_column(self, df: pd.DataFrame) -> str:
        """
        Try to detect which column is the time column.
        
        Looks for columns with datetime-like names or types.
        """
        # Common time column names
        time_names = ['date', 'time', 'datetime', 'timestamp', 'ds', 'period']
        
        for col in df.columns:
            # Check by name
            if col.lower() in time_names:
                return col
            
            # Check by type
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                return col
        
        return None
    
    def validate(self, data: pd.DataFrame) -> Tuple[bool, Dict[str, Any]]:
        """Validate pandas DataFrame for time series forecasting."""
        errors = []
        warnings = []
        
        # Check for datetime index
        if not isinstance(data.index, pd.DatetimeIndex):
            errors.append("Index must be DatetimeIndex for time series forecasting")
        
        # Check for empty data
        if len(data) == 0:
            errors.append("DataFrame is empty")
        
        # Check for missing values
        missing_counts = data.isnull().sum()
        if missing_counts.any():
            missing_pct = (missing_counts / len(data) * 100).round(2)
            warnings.append(
                f"Missing values detected: {missing_pct[missing_pct > 0].to_dict()}"
            )
        
        # Check for duplicate indices
        if data.index.duplicated().any():
            dup_count = data.index.duplicated().sum()
            errors.append(f"Duplicate time indices found: {dup_count} duplicates")
        
        # Check for monotonic index
        if not data.index.is_monotonic_increasing:
            warnings.append("Time index is not sorted (will be sorted automatically)")
        
        # Check for sufficient data
        if len(data) < 10:
            warnings.append(f"Very small dataset ({len(data)} rows). Consider using more data for reliable forecasting.")
        
        # Check for constant values
        for col in data.columns:
            if data[col].nunique() == 1:
                warnings.append(f"Column '{col}' has constant values")
        
        # Check frequency
        if isinstance(data.index, pd.DatetimeIndex):
            freq = pd.infer_freq(data.index)
            if freq is None:
                warnings.append("Could not infer frequency. Time series may have irregular intervals.")
        
        is_valid = len(errors) == 0
        
        return is_valid, {
            "valid": is_valid,
            "errors": errors,
            "warnings": warnings,
        }
