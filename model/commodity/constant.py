"""Constants that will help commodity price model"""
# Column Descripton
id_columns: list[str] = ['commodity', 'province']
target_columns: list[str] = ['price']
timestamp_column: str = "Date"
DEFAULT_PARAMETER="default_parameter"

column_specifiers = {
    "timestamp_column": timestamp_column,
    "id_columns": id_columns,
    "target_columns": target_columns,
    "control_columns": [],
}

# Configuration
split_config = {
    "train": 0.8,
    "test": 0.1
}

TTM_MODEL_PATH = "ibm-granite/granite-timeseries-ttm-r2"
feature_to_scale = ['GlobalOpen', 'GlobalHigh', 'GlobalVol.', 'GlobalPrice', 'CE_Close', 'CE_High', 'CE_Low', 'CE_Open']
