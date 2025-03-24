"""Constants that will help commodity price model"""
# Column Descripton
id_columns: list[str] = ['commodity', 'province']
target_columns: list[str] = ['price']
timestamp_column: str = "Date"

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
