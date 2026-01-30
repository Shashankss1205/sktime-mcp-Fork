"""
Data source adapters.

Available adapters:
- PandasAdapter: In-memory DataFrames
- SQLAdapter: SQL databases
- FileAdapter: CSV, Excel, Parquet files
"""

from .pandas_adapter import PandasAdapter
from .sql_adapter import SQLAdapter
from .file_adapter import FileAdapter

__all__ = [
    "PandasAdapter",
    "SQLAdapter",
    "FileAdapter",
]
