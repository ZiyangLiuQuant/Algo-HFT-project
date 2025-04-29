import pandas as pd
import os
from typing import List, Dict, Optional
import logging
import pyarrow.feather as feather
import numpy as np
from datetime import datetime

logger = logging.getLogger(__name__)

class DataLoader:
    def __init__(self, data_dir: str = "data", processed_dir: str = "processed_data"):
        """
        初始化数据加载器
        
        Args:
            data_dir: 原始数据目录路径
            processed_dir: 处理后的数据目录路径
        """
        self.data_dir = data_dir
        self.processed_dir = processed_dir
        self.symbols = ['AAPL', 'MSFT', 'INTC', 'GOOG', 'AMZN']
        
        # 确保目录存在
        os.makedirs(self.processed_dir, exist_ok=True)
        
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
    def load_data(self, symbol: str) -> Optional[pd.DataFrame]:
        """
        加载数据并计算时间相关特征
        
        Args:
            symbol: 股票代码
            
        Returns:
            pd.DataFrame: 处理后的数据框，如果加载失败则返回None
        """
        try:
            file_path = os.path.join(self.data_dir, f"{symbol}_train_data.csv")
            
            if not os.path.exists(file_path):
                self.logger.error(f"Data file not found for {symbol}")
                return None
                
            self.logger.info(f"Loading data from {file_path}")
            
            # Load data with detailed logging
            df = pd.read_csv(file_path)
            self.logger.info(f"Successfully loaded data. Shape: {df.shape}")
            self.logger.info(f"Original columns: {df.columns.tolist()}")
            
            # Rename columns to remove spaces and make them more code-friendly
            column_mapping = {
                'time': 'time',
                'ask price': 'ask_price',
                'ask volume': 'ask_size',
                'bid price': 'bid_price',
                'bid volume': 'bid_size',
                'order type (1:submission_of_new_limit, 2:partial_cancelation, 3:total_cancelation, 4:execution_of_visible, 5:execution_of_hidden, 7:trading_halt)': 'order_type',
                'order id': 'order_id',
                'size': 'trade_size',
                'order price': 'trade_price',
                'order direction (-1:sell, 1:buy)': 'order_direction',
                'mid price = (ask price + bid price) / 2': 'mid_price',
                'spread = ask price - bid price': 'spread'
            }
            
            df = df.rename(columns=column_mapping)
            self.logger.info(f"Renamed columns: {df.columns.tolist()}")
            
            # Convert time to datetime and set as index
            # First convert nanoseconds to seconds by dividing by 1e9
            df['time'] = pd.to_datetime('2023-01-01') + pd.to_timedelta(df['time'].astype(float), unit='s')
            df = df.set_index('time')
            self.logger.info("Set timestamp as index")
            
            # Sort by timestamp
            df.sort_index(inplace=True)
            
            # Check for required columns
            required_columns = ['ask_price', 'bid_price', 'ask_size', 'bid_size']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                self.logger.error(f"Missing required columns: {missing_columns}")
                return None
                
            # Log data statistics
            self.logger.info("\nData Statistics:")
            self.logger.info(f"Time range: {df.index.min()} to {df.index.max()}")
            self.logger.info(f"Number of unique timestamps: {df.index.nunique()}")
            self.logger.info(f"Missing values:\n{df.isnull().sum()}")
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error loading data for {symbol}: {str(e)}")
            return None
    
    def save_processed_data(self, symbol: str, df: pd.DataFrame) -> bool:
        """Save processed data to CSV file"""
        try:
            # Drop duplicate time indices
            df = df[~df.index.duplicated(keep='first')]
            
            # Save to CSV
            output_path = os.path.join(self.processed_dir, f"{symbol}_processed.csv")
            df.to_csv(output_path)
            self.logger.info(f"Saved processed data to {output_path}")
            return True
        except Exception as e:
            self.logger.error(f"Error saving processed data for {symbol}: {str(e)}")
            return False
    
    def load_processed_data(self, symbol: str, suffix: str = "processed") -> Optional[pd.DataFrame]:
        """
        加载处理后的数据
        
        Args:
            symbol: 股票代码
            suffix: 文件后缀
            
        Returns:
            pd.DataFrame: 数据框，如果加载失败则返回None
        """
        try:
            file_name = f"{symbol}_{suffix}"
            file_path = os.path.join(self.processed_dir, f"{file_name}.feather")
            
            if not os.path.exists(file_path):
                self.logger.warning(f"Processed data file not found for {file_name}")
                return None
            
            df = pd.read_feather(file_path)
            
            # Set time as index if it exists
            if 'time' in df.columns:
                df = df.set_index('time')
            
            self.logger.info(f"Successfully loaded processed data for {file_name}")
            return df
            
        except Exception as e:
            self.logger.error(f"Error loading processed data for {symbol}: {str(e)}")
            return None
    
    def get_processed_files(self, symbol: str) -> Dict[str, str]:
        """
        获取指定股票的所有处理后的文件
        
        Args:
            symbol: 股票代码
            
        Returns:
            Dict[str, str]: 文件名到完整路径的映射
        """
        files = {}
        for file in os.listdir(self.processed_dir):
            if file.startswith(symbol) and file.endswith('.feather'):
                file_name = file.replace('.feather', '')
                file_path = os.path.join(self.processed_dir, file)
                files[file_name] = file_path
        return files
    
    def check_data_availability(self, symbol: str) -> bool:
        """
        检查指定股票的数据是否可用
        
        Args:
            symbol: 股票代码
            
        Returns:
            bool: 数据是否可用
        """
        # 检查原始数据或处理后的数据
        raw_file = os.path.join(self.data_dir, f"{symbol}_train_data.csv")
        if not os.path.exists(raw_file):
            self.logger.warning(f"Raw data file not found for {symbol}")
            return False
        
        processed_files = self.get_processed_files(symbol)
        if not processed_files:
            self.logger.warning(f"No processed data files found for {symbol}")
            return False
        
        return True
    
    def process_all_symbols(self):
        """
        处理所有股票的数据
        """
        for symbol in self.symbols:
            print(f"Processing {symbol} data...")
            df = self.load_data(symbol)
            if df is not None:
                self.save_processed_data(symbol, df)
                print(f"Finished processing {symbol} data") 