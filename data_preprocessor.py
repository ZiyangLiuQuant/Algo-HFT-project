import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)

class DataPreprocessor:
    def __init__(self):
        """
        初始化数据预处理器
        """
        # 必需的数据列
        self.required_columns = [
            'mid_price', 'ask_price', 'bid_price', 
            'ask_size', 'bid_size', 'trade_price', 
            'trade_size', 'order direction (-1:sell, 1:buy)'
        ]
        
        # 价格相关的列
        self.price_columns = ['ask_price', 'bid_price', 'trade_price', 'mid_price']
        
        # 数量相关的列
        self.size_columns = ['ask_size', 'bid_size', 'trade_size']
        
        # 时间窗口设置
        self.time_windows = ['1s', '5s', '30s']
        
    def validate_data(self, df: pd.DataFrame) -> bool:
        """
        验证数据完整性
        
        Args:
            df: 输入数据框
            
        Returns:
            bool: 数据是否有效
            
        Raises:
            ValueError: 当数据不完整时抛出
        """
        # 检查必需列
        missing_cols = [col for col in self.required_columns if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
            
        # 检查时间索引
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("DataFrame must have datetime index")
            
        # 检查数据范围
        for col in self.price_columns:
            if df[col].min() <= 0:
                raise ValueError(f"Invalid price values in {col}")
                
        for col in self.size_columns:
            if df[col].min() < 0:
                raise ValueError(f"Invalid size values in {col}")
                
        return True
        
    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Enhanced missing value handling with sophisticated imputation
        
        Args:
            df: Input DataFrame
            
        Returns:
            pd.DataFrame: Processed DataFrame
        """
        df = df.copy()
        
        # 1. Price data imputation
        for col in self.price_columns:
            # First try forward fill with a limit
            df[col] = df[col].ffill(limit=5)
            # Then try backward fill for any remaining NaNs
            df[col] = df[col].bfill(limit=5)
            # For any remaining NaNs, use rolling median
            df[col] = df[col].fillna(df[col].rolling(window=10, min_periods=1, center=True).median())
        
        # 2. Size data imputation
        for col in self.size_columns:
            # Use 0 for missing sizes, but only if missing for less than 5 consecutive periods
            mask = df[col].isna()
            consecutive_nas = mask.astype(int).groupby(mask.ne(mask.shift()).cumsum()).cumsum()
            df.loc[consecutive_nas <= 5, col] = 0
            # For longer gaps, use rolling median
            df[col] = df[col].fillna(df[col].rolling(window=10, min_periods=1, center=True).median())
        
        # 3. Order direction imputation
        if 'order_direction' in df.columns:
            # Forward fill order direction, but only for a limited number of periods
            df['order_direction'] = df['order_direction'].ffill(limit=3)
            # For remaining NaNs, use most frequent direction in surrounding window
            window_size = 5
            df['order_direction'] = df['order_direction'].fillna(
                df['order_direction'].rolling(window=window_size, min_periods=1, center=True).apply(
                    lambda x: x.mode()[0] if not x.empty else 0
                )
            )
        
        # 4. Feature imputation
        feature_cols = [col for col in df.columns if any([
            col.startswith(prefix) for prefix in [
                'momentum__', 'spread__', 'orderflow__', 'trade__',
                'order__', 'execution__', 'minute__', 'micro__'
            ]
        ])]
        
        for col in feature_cols:
            if col in df.columns:
                # Use rolling statistics for feature imputation
                df[col] = df[col].fillna(df[col].rolling(window=10, min_periods=1, center=True).mean())
                
        return df
        
    def remove_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        去除异常值
        
        Args:
            df: 输入数据框
            
        Returns:
            pd.DataFrame: 处理后的数据框
        """
        df = df.copy()
        
        # 价格异常值处理
        for col in self.price_columns:
            # 计算滚动统计量
            rolling_mean = df[col].rolling('5min').mean()
            rolling_std = df[col].rolling('5min').std()
            
            # 设置上下限
            upper_bound = rolling_mean + 3 * rolling_std
            lower_bound = rolling_mean - 3 * rolling_std
            
            # 替换异常值
            df[col] = df[col].clip(lower_bound, upper_bound)
            
        # 数量异常值处理
        for col in self.size_columns:
            # 使用分位数方法
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            upper_bound = q3 + 3 * iqr
            lower_bound = q1 - 3 * iqr
            
            df[col] = df[col].clip(lower_bound, upper_bound)
            
        return df
        
    def add_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        添加衍生特征
        
        Args:
            df: 输入数据框
            
        Returns:
            pd.DataFrame: 添加了衍生特征的数据框
        """
        df = df.copy()
        
        # 计算spread
        df['spread'] = df['ask_price'] - df['bid_price']
        
        # 计算相对spread
        df['relative_spread'] = df['spread'] / df['mid_price']
        
        # 计算订单簿不平衡
        df['orderbook_imbalance'] = (
            (df['bid_size'] - df['ask_size']) / 
            (df['bid_size'] + df['ask_size'] + 1e-6)
        )
        
        # 添加trade_side列
        df['trade_side'] = df['order direction (-1:sell, 1:buy)'].map({-1: 'sell', 1: 'buy'})
        
        return df
        
    def standardize_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        标准化特征
        
        Args:
            df: 输入数据框
            
        Returns:
            pd.DataFrame: 标准化后的数据框
        """
        df = df.copy()
        
        # 获取所有特征列（排除目标变量和非数值列）
        feature_cols = [
            col for col in df.columns 
            if not col.startswith('target_') and 
            pd.api.types.is_numeric_dtype(df[col])
        ]
        
        # 计算滚动统计量
        for col in feature_cols:
            try:
                # 使用5分钟窗口
                rolling_mean = df[col].rolling('5min').mean()
                rolling_std = df[col].rolling('5min').std()
                
                # 处理标准差接近0的情况
                rolling_std = rolling_std.clip(lower=1e-6)
                
                # 计算z-score
                df[col] = (df[col] - rolling_mean) / rolling_std
                
            except Exception as e:
                logger.warning(f"Error standardizing feature {col}: {str(e)}")
                continue
            
        return df
        
    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        完整的数据处理流程
        
        Args:
            df: 输入数据框
            
        Returns:
            pd.DataFrame: 处理后的数据框
        """
        try:
            # 1. 验证数据
            self.validate_data(df)
            
            # 2. 处理缺失值
            df = self.handle_missing_values(df)
            
            # 3. 去除异常值
            df = self.remove_outliers(df)
            
            # 4. 添加衍生特征
            df = self.add_derived_features(df)
            
            # 5. 标准化特征
            df = self.standardize_features(df)
            
            return df
            
        except Exception as e:
            logger.error(f"Error in data preprocessing: {str(e)}")
            raise 