import pandas as pd
from typing import Dict

# 存储每个股票的完整数据（包含原始数据、因子和目标值）
AAPL: pd.DataFrame = None
AMZN: pd.DataFrame = None
GOOG: pd.DataFrame = None
INTC: pd.DataFrame = None
MSFT: pd.DataFrame = None

# 所有股票的DataFrame字典
stocks: dict = {
    'AAPL': None,
    'AMZN': None,
    'GOOG': None,
    'INTC': None,
    'MSFT': None
}

# 存储所有原始数据的DataFrame
raw_data: pd.DataFrame = None

# 存储所有带目标变量的DataFrame
targets_data: pd.DataFrame = None

# 存储原始数据的DataFrame
AAPL_df: pd.DataFrame = None
AMZN_df: pd.DataFrame = None
GOOG_df: pd.DataFrame = None
INTC_df: pd.DataFrame = None
MSFT_df: pd.DataFrame = None

# 存储带目标变量的DataFrame
AAPL_targets: pd.DataFrame = None
AMZN_targets: pd.DataFrame = None
GOOG_targets: pd.DataFrame = None
INTC_targets: pd.DataFrame = None
MSFT_targets: pd.DataFrame = None

# 所有股票的DataFrame字典
raw_data: Dict[str, pd.DataFrame] = {
    'AAPL': None,
    'AMZN': None,
    'GOOG': None,
    'INTC': None,
    'MSFT': None
}

# 所有带目标变量的DataFrame字典
targets_data: Dict[str, pd.DataFrame] = {
    'AAPL': None,
    'AMZN': None,
    'GOOG': None,
    'INTC': None,
    'MSFT': None
} 