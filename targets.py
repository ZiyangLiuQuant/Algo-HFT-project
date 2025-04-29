import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import logging

logger = logging.getLogger(__name__)

class TargetCalculator:
    def __init__(self):
        """
        初始化目标变量计算器
        """
        # 时间窗口设置
        self.time_windows = ['1s', '5s', '30s']
        self.tick_windows = [1, 5, 50, 100]
        
    def _get_minute_groups(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        将数据按分钟分组，并计算每分钟的最后一个tick的ask/bid price作为TWAP
        
        Args:
            df: 原始DataFrame，包含timestamp索引和必要的价格列
            
        Returns:
            pd.DataFrame: 添加了分组信息的DataFrame
        """
        # 创建分钟分组
        df = df.copy()
        df['minute_group'] = df.index.floor('1min')
        
        # 计算每分钟最后一个tick的ask/bid price作为TWAP
        last_ticks_ask = df.groupby('minute_group')['ask_price'].last()
        last_ticks_bid = df.groupby('minute_group')['bid_price'].last()
        last_ticks_mid = df.groupby('minute_group')['mid_price'].last()
        
        df['twap_ask'] = df['minute_group'].map(last_ticks_ask)  # 买入时的TWAP
        df['twap_bid'] = df['minute_group'].map(last_ticks_bid)  # 卖出时的TWAP
        df['twap_mid'] = df['minute_group'].map(last_ticks_mid)  # 原有的mid price TWAP
        
        return df
        
    def calculate_targets(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算所有目标变量
        
        Args:
            df: 原始DataFrame，必须包含以下列：
               - timestamp (index)
               - mid_price
               - ask_price
               - bid_price
               - spread
               - ask_size (可选)
               - bid_size (可选)
               - trade_price (可选)
               - trade_size (可选)
               - trade_side (可选)
               
        Returns:
            pd.DataFrame: 包含所有目标变量的DataFrame
        """
        df = df.copy()
        
        try:
            # 1. 短期价格跳跃预测
            # 1.1 tick级别的跳跃（使用历史数据）
            for ticks in self.tick_windows:
                # 使用历史数据计算跳跃
                df[f'target_midprice_jump_{ticks}tick'] = (
                    df['mid_price'] - df['mid_price'].shift(ticks)
                ).fillna(np.nan)
                
                # 添加相对变化率
                df[f'target_midprice_jump_{ticks}tick_relative'] = (
                    (df['mid_price'] - df['mid_price'].shift(ticks)) / 
                    df['mid_price'].shift(ticks)
                ).fillna(np.nan)
            
            # 1.2 秒级跳跃（使用历史数据）
            for seconds in self.time_windows:
                # 将时间窗口转换为periods
                periods = int(seconds.replace('s', ''))
                
                # 使用历史数据计算跳跃
                df[f'target_midprice_jump_{seconds}'] = (
                    df['mid_price'] - df['mid_price'].shift(periods)
                ).fillna(np.nan)
                
                # 添加相对变化率
                df[f'target_midprice_jump_{seconds}_relative'] = (
                    (df['mid_price'] - df['mid_price'].shift(periods)) / 
                    df['mid_price'].shift(periods)
                ).fillna(np.nan)
            
            # 2. 微观结构变化
            # 2.1 spread变化（使用历史数据）
            for seconds in self.time_windows:
                # 将时间窗口转换为periods
                periods = int(seconds.replace('s', ''))
                
                # 连续值版本
                df[f'target_spread_narrowing_{seconds}'] = (
                    df['spread'] - df['spread'].shift(periods)
                ).fillna(np.nan)
                
                # 相对变化率
                df[f'target_spread_narrowing_{seconds}_relative'] = (
                    (df['spread'] - df['spread'].shift(periods)) / 
                    df['spread'].shift(periods)
                ).fillna(np.nan)
            
            # 2.2 订单簿不平衡变化（使用历史数据）
            if 'ask_size' in df.columns and 'bid_size' in df.columns:
                for seconds in self.time_windows:
                    # 将时间窗口转换为periods
                    periods = int(seconds.replace('s', ''))
                    
                    # 计算当前订单簿不平衡
                    current_imbalance = (df['bid_size'] - df['ask_size']) / (df['bid_size'] + df['ask_size'])
                    
                    # 计算历史订单簿不平衡
                    historical_imbalance = (
                        (df['bid_size'].shift(periods) - df['ask_size'].shift(periods)) / 
                        (df['bid_size'].shift(periods) + df['ask_size'].shift(periods))
                    )
                    
                    # 计算变化
                    df[f'target_imbalance_change_{seconds}'] = (
                        current_imbalance - historical_imbalance
                    ).fillna(np.nan)
            
            # 3. 交易方向预测
            if 'trade_side' in df.columns:
                for seconds in self.time_windows:
                    # 将时间窗口转换为periods
                    periods = int(seconds.replace('s', ''))
                    
                    # 预测未来交易方向
                    df[f'target_trade_side_{seconds}'] = (
                        df['trade_side'].shift(-periods)
                    ).fillna(np.nan)
            
            return df
            
        except Exception as e:
            logger.error(f"Error calculating targets: {str(e)}")
            raise
            
    def evaluate_targets(self, df: pd.DataFrame) -> Dict:
        """
        评估目标变量的统计特性
        
        Args:
            df: 包含目标变量的DataFrame
            
        Returns:
            Dict: 目标变量的统计信息
        """
        target_stats = {}
        
        try:
            # 获取所有目标列
            target_columns = [col for col in df.columns if col.startswith('target_')]
            
            for col in target_columns:
                # 计算基本统计量
                stats = {
                    'mean': df[col].mean(),
                    'std': df[col].std(),
                    'min': df[col].min(),
                    'max': df[col].max(),
                    'skew': df[col].skew(),
                    'kurtosis': df[col].kurtosis(),
                    'non_null_count': df[col].count(),
                    'null_count': df[col].isnull().sum()
                }
                
                target_stats[col] = stats
                
            return target_stats
            
        except Exception as e:
            logger.error(f"Error evaluating targets: {str(e)}")
            raise 