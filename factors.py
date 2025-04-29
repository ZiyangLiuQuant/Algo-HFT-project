import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from datetime import datetime, timedelta
import os
import logging
import gc
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

logger = logging.getLogger(__name__)

class FactorCalculator:
    def __init__(self):
        """
        初始化因子计算器
        """
        # tick级窗口 [短-中-长]
        self.tick_windows = [5, 20, 100]
        # 秒级窗口 [短-中-长]
        self.time_windows = [2, 5, 10, 30, 300]  # 秒
        # 分钟级窗口 [短-中-长]
        self.minute_windows = [1, 5, 10, 30, 60]  # 分钟
        # 日级窗口
        self.day_windows = ['today']  # 今日开盘以来
        
    def load_processed_data(self, symbol: str) -> pd.DataFrame:
        """
        从data_processed文件夹加载处理后的数据
        
        Args:
            symbol (str): 股票代码
            
        Returns:
            pd.DataFrame: 处理后的数据
        """
        file_path = os.path.join('data_processed', f'{symbol}_processed.csv')
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Processed data file for {symbol} not found")
        
        df = pd.read_csv(file_path, index_col=0, parse_dates=True)
        return df
        
    def calculate_factors(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算所有因子，使用并行计算加速
        
        Args:
            df (pd.DataFrame): 包含行情数据的DataFrame
            
        Returns:
            pd.DataFrame: 添加了因子的DataFrame
        """
        try:
            print("\n=== Starting Factor Calculation ===")
            
            # 复制DataFrame以避免修改原始数据
            df = df.copy()
            
            # 验证输入数据
            if df.empty:
                raise ValueError("Input DataFrame is empty")
            
            # 检查必要的列是否存在
            required_columns = ['ask_price', 'bid_price', 'ask_size', 'bid_size', 'trade_price', 'trade_size']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")
            
            # 检查数据质量
            if df[required_columns].isnull().any().any():
                print("Warning: Input data contains missing values")
            
            # 添加时间相关的列
            print("\n1. Adding time-related columns...")
            df['date'] = df.index.date
            df['time'] = df.index.time
            df['minute'] = df.index.floor('min')
            df['hour'] = df.index.floor('h')
            
            # 预先创建一个空的DataFrame来存储所有因子
            factors_df = pd.DataFrame(index=df.index)
            
            # 定义需要并行计算的因子计算方法
            factor_methods = [
                ('price_momentum', self._calculate_price_momentum_factors),
                ('spread', self._calculate_spread_factors),
                ('trade', self._calculate_trade_factors),
                ('orderbook_structure', self._calculate_orderbook_structure_factors),
                ('order_dynamic', self._calculate_order_dynamic_factors),
                ('microstructure', self._calculate_microstructure_factors),
                ('cross_interaction', self._calculate_cross_interaction_features),
                ('intraday', self._calculate_intraday_features)
            ]
            
            print("\n2. Calculating factors in parallel...")
            # 使用ThreadPoolExecutor进行并行计算
            with ThreadPoolExecutor(max_workers=4) as executor:
                # 提交所有任务
                future_to_method = {
                    executor.submit(method, df): name 
                    for name, method in factor_methods
                }
                
                # 收集结果
                for future in as_completed(future_to_method):
                    method_name = future_to_method[future]
                    try:
                        result = future.result()
                        if not result.empty:
                            factors_df = pd.concat([factors_df, result], axis=1)
                            print(f"   - Completed {method_name} factors: {len(result.columns)} features")
                        else:
                            print(f"   - No {method_name} factors calculated")
                    except Exception as e:
                        print(f"   - Error calculating {method_name} factors: {str(e)}")
            
            # 标准化处理
            print("\n3. Standardizing features...")
            factors_df = self._standardize_features(factors_df)
            print(f"   - Standardized {len(factors_df.columns)} features")
            
            # 去冗余处理
            print("\n4. Removing highly correlated features...")
            factors_df = self.drop_highly_correlated_features(factors_df)
            print(f"   - Remaining features after correlation filtering: {len(factors_df.columns)}")
            
            # 检查因子计算结果
            if factors_df.empty:
                raise ValueError("No factors were calculated")
                
            # 检查因子质量
            print("\n5. Checking factor quality...")
            constant_factors = factors_df.columns[factors_df.nunique() == 1].tolist()
            if constant_factors:
                print(f"   - Found {len(constant_factors)} constant factors")
                factors_df = factors_df.drop(columns=constant_factors)
            
            # 合并原始数据和因子
            print("\n6. Merging results...")
            result_df = pd.concat([df, factors_df], axis=1)
            print(f"   - Final number of factors: {len(factors_df.columns)}")
            print("=== Factor Calculation Completed ===")
            
            return result_df
            
        except Exception as e:
            print(f"\nError calculating factors: {str(e)}")
            raise
    
    def _calculate_price_momentum_factors(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算价格动量类因子
        
        因子经济含义：
        - 价格动量：反映价格趋势的强度和方向
        - 价格加速度：反映价格趋势的变化率
        - 价格范围：反映价格的波动范围
        - 买卖价差偏度：反映买卖盘的不对称性
        - 价格突破：反映价格突破关键水平的能力
        - 价格波动率：反映价格的波动性
        - 价格弹性：反映价格对成交量的敏感度
        - 局部斜率：反映价格的局部趋势
        - 局部波动率爆发：反映价格的突发性波动
        
        预期行为：
        - 正动量通常预示价格继续上涨
        - 高加速度可能预示趋势反转
        - 大范围通常伴随高波动
        - 偏度变化反映买卖盘力量对比
        - 突破可能预示趋势延续
        - 高波动率通常伴随不确定性
        - 高弹性可能预示流动性变化
        - 高局部斜率预示短期趋势
        - 高波动率爆发预示价格突变
        """
        factors = {}
        
        # 计算mid_price
        df['mid_price'] = (df['ask_price'] + df['bid_price']) / 2
        
        # 基础价格特征
        factors['price__bid_ask_skewness'] = (
            (df['ask_price'] - df['mid_price']) / 
            (df['mid_price'] - df['bid_price'] + 1e-6)
        )
        
        # 价格范围特征
        for window in ['1s', '5s', '30s']:
            factors[f'price__local_range_{window}'] = (
                df['ask_price'].rolling(window).max() - 
                df['bid_price'].rolling(window).min()
            )
        
        # 价格加速度
        factors['price__mid_price_acceleration'] = df['mid_price'].diff().diff()
        factors['price__mid_price_jerk'] = df['mid_price'].diff().diff().diff()  # 三阶导数
        
        # 价格突破特征
        for n in [5, 20, 100]:
            # 计算突破方向（使用历史数据）
            price_high = df['mid_price'].rolling(n).max().shift(1)  # 使用shift(1)避免未来数据
            price_low = df['mid_price'].rolling(n).min().shift(1)
            factors[f'price__breakout_{n}tick'] = (
                (df['mid_price'] > price_high).astype(int) - 
                (df['mid_price'] < price_low).astype(int)
            )
        
        # 价格波动率特征
        for window in ['1s', '5s', '30s']:
            log_returns = np.log(df['mid_price'] / df['mid_price'].shift(1))
            factors[f'price__volatility_{window}'] = log_returns.rolling(window).std()
            factors[f'price__volatility_ratio_{window}'] = (
                log_returns.rolling(window).std() / 
                log_returns.rolling(f'{int(window[:-1])*2}s').std()
            )
            
            # 局部波动率爆发特征
            vol_ma = log_returns.rolling(window).std()
            vol_std = vol_ma.rolling(window).std()
            factors[f'price__volatility_burst_{window}'] = (
                (vol_ma - vol_ma.rolling(window).mean()) / (vol_std + 1e-6)
            )
        
        # 价格弹性特征
        for window in ['1s', '5s', '30s']:
            volume = df['trade_size'].rolling(window).sum()
            price_change = df['mid_price'].diff()
            factors[f'price__elasticity_{window}'] = (
                price_change / (volume + 1e-6)
            )
        
        # 局部斜率特征
        for window in ['1s', '5s']:
            def calculate_slope(x):
                if len(x) < 2:
                    return np.nan
                return np.polyfit(range(len(x)), x, 1)[0]
            
            factors[f'price__local_slope_{window}'] = (
                df['mid_price'].rolling(window).apply(calculate_slope)
            )
        
        # 原有的动量特征
        for n in self.tick_windows:
            factors[f'momentum__mid_price__{n}tick__diff'] = df['mid_price'].diff(n)
            factors[f'momentum__mid_price__{n}tick__pct'] = df['mid_price'].pct_change(n)
            # 添加动量加速度
            factors[f'momentum__mid_price__{n}tick__acceleration'] = (
                df['mid_price'].diff(n).diff(n)
            )
        
        for n in self.time_windows:
            window = f'{n}s'
            factors[f'momentum__mid_price__{window}__diff'] = df['mid_price'].diff().rolling(window).sum()
            factors[f'momentum__mid_price__{window}__pct'] = df['mid_price'].pct_change().rolling(window).sum()
            # 添加动量加速度
            factors[f'momentum__mid_price__{window}__acceleration'] = (
                df['mid_price'].diff().rolling(window).sum().diff()
            )
        
        return pd.DataFrame(factors)
    
    def _calculate_spread_factors(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算价差类因子
        
        因子经济含义：
        - 价差水平：反映市场流动性水平
        - 价差波动：反映市场流动性风险
        - 价差趋势：反映流动性变化趋势
        - 价差弹性：反映价差对成交量的敏感度
        - 价差偏度：反映买卖盘的不对称性
        - 价差压力：反映市场压力水平
        - 价差效率：反映市场效率
        - 价差收敛速度：反映价差的动态变化
        - 价差早期预警：反映价差变化的预警信号
        
        预期行为：
        - 价差扩大通常预示流动性下降
        - 价差波动增加预示市场不确定性
        - 价差趋势变化可能预示市场状态转变
        - 价差弹性反映市场深度
        - 价差偏度反映买卖盘力量对比
        - 价差压力反映市场压力水平
        - 价差效率反映市场有效性
        - 价差收敛速度反映市场调整能力
        - 价差早期预警反映市场状态变化
        """
        factors = {}
        
        # 基础价差特征
        df['spread'] = df['ask_price'] - df['bid_price']
        df['relative_spread'] = df['spread'] / df['mid_price']
        
        # 价差水平特征
        for window in ['1s', '5s', '30s']:
            factors[f'spread__{window}__mean'] = df['spread'].rolling(window).mean()
            factors[f'spread__{window}__median'] = df['spread'].rolling(window).median()
            factors[f'spread__{window}__min'] = df['spread'].rolling(window).min()
            factors[f'spread__{window}__max'] = df['spread'].rolling(window).max()
            
            # 相对价差特征
            factors[f'spread__relative_{window}__mean'] = df['relative_spread'].rolling(window).mean()
            factors[f'spread__relative_{window}__std'] = df['relative_spread'].rolling(window).std()
        
        # 价差波动特征
        for window in ['1s', '5s', '30s']:
            factors[f'spread__volatility_{window}'] = df['spread'].rolling(window).std()
            factors[f'spread__volatility_ratio_{window}'] = (
                df['spread'].rolling(window).std() / 
                df['spread'].rolling(f'{int(window[:-1])*2}s').std()
            )
        
        # 价差趋势特征
        for window in ['1s', '5s', '30s']:
            factors[f'spread__trend_{window}'] = (
                df['spread'].rolling(window).mean().diff()
            )
            factors[f'spread__trend_strength_{window}'] = (
                df['spread'].rolling(window).mean().diff() / 
                (df['spread'].rolling(window).std() + 1e-6)
            )
        
        # 价差弹性特征
        for window in ['1s', '5s', '30s']:
            volume = df['trade_size'].rolling(window).sum()
            spread_change = df['spread'].diff()
            factors[f'spread__elasticity_{window}'] = (
                spread_change / (volume + 1e-6)
            )
        
        # 价差偏度特征
        for window in ['1s', '5s', '30s']:
            factors[f'spread__skewness_{window}'] = (
                (df['ask_price'] - df['mid_price']) / 
                (df['mid_price'] - df['bid_price'] + 1e-6)
            ).rolling(window).mean()
        
        # 价差压力特征
        for window in ['1s', '5s', '30s']:
            factors[f'spread__pressure_{window}'] = (
                (df['ask_price'] - df['mid_price']) / 
                (df['mid_price'] - df['bid_price'] + 1e-6)
            ).rolling(window).mean()
        
        # 价差效率特征
        for window in ['1s', '5s', '30s']:
            mid_price_volatility = df['mid_price'].rolling(window).std()
            spread_volatility = df['spread'].rolling(window).std()
            factors[f'spread__efficiency_{window}'] = (
                mid_price_volatility / (spread_volatility + 1e-6)
            )
        
        # 新增：价差收敛速度特征
        for window in ['1s', '5s']:
            # 使用历史数据计算收敛速度
            spread_ma = df['spread'].rolling(window).mean()
            spread_change = spread_ma.diff()
            factors[f'spread__convergence_speed_{window}'] = (
                spread_change / (spread_ma + 1e-6)
            )
            
            # 价差早期预警特征
            spread_std = df['spread'].rolling(window).std()
            factors[f'spread__early_warning_{window}'] = (
                (spread_change - spread_change.rolling(window).mean()) / 
                (spread_std + 1e-6)
            )
        
        return pd.DataFrame(factors)
    
    def _calculate_trade_factors(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算成交类因子
        
        因子经济含义：
        - 成交方向强度：反映买卖力量的对比
        - 成交不平衡速度：反映成交不平衡的变化率
        - 成交规模：反映市场活跃度
        - 可见流动性比率：反映订单簿深度与成交量的关系
        - 主动成交比例：反映市场参与者的主动性
        - 成交簇特征：反映成交的聚集性
        
        预期行为：
        - 高方向强度通常预示趋势延续
        - 不平衡速度变化可能预示反转
        - 大成交规模通常伴随价格波动
        - 低流动性比率可能预示流动性风险
        - 高主动成交比例预示市场活跃度
        - 成交簇特征反映市场情绪变化
        """
        factors = {}
        
        # 判断成交方向（假设trade_price > mid_price为买，反之为卖）
        mid_price = (df['ask_price'] + df['bid_price']) / 2
        is_buy = (df['trade_price'] > mid_price).astype(int)
        buy_volume = df['trade_size'] * is_buy
        sell_volume = df['trade_size'] * (1 - is_buy)
        
        # 成交方向强度
        total_volume = buy_volume + sell_volume
        factors['trade__direction_strength'] = (
            buy_volume.rolling('1s').sum() / 
            (total_volume.rolling('1s').sum() + 1e-6)
        )
        
        # 成交不平衡速度
        imbalance = buy_volume - sell_volume
        factors['trade__imbalance_velocity'] = imbalance.diff()
        
        # 可见流动性比率
        visible_liquidity = df['bid_size'] + df['ask_size']
        factors['liquidity__visible_liquidity_ratio'] = (
            visible_liquidity / 
            (df['trade_size'].rolling('5s').sum() + 1e-6)
        )
        
        # 新增：细粒度主动成交统计
        for window in ['1s', '5s']:
            # 主动买盘比例
            factors[f'trade__aggressive_buy_ratio_{window}'] = (
                buy_volume.rolling(window).sum() / 
                (total_volume.rolling(window).sum() + 1e-6)
            )
            
            # 主动卖盘比例
            factors[f'trade__aggressive_sell_ratio_{window}'] = (
                sell_volume.rolling(window).sum() / 
                (total_volume.rolling(window).sum() + 1e-6)
            )
            
            # 成交簇特征
            volume_ma = df['trade_size'].rolling(window).mean()
            volume_std = df['trade_size'].rolling(window).std()
            factors[f'trade__volume_cluster_{window}'] = (
                (df['trade_size'] - volume_ma) / (volume_std + 1e-6)
            )
        
        # 原有的成交特征
        for n in self.tick_windows:
            factors[f'trade__size__{n}tick__sum'] = df['trade_size'].rolling(n).sum()
            factors[f'trade__size__{n}tick__mean'] = df['trade_size'].rolling(n).mean()
        
        for n in self.time_windows:
            window = f'{n}s'
            factors[f'trade__size__{window}__sum'] = df['trade_size'].rolling(window).sum()
            factors[f'trade__size__{window}__mean'] = df['trade_size'].rolling(window).mean()
        
        return pd.DataFrame(factors)
    
    def _calculate_orderbook_structure_factors(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算订单簿结构类因子"""
        factors = {}
        
        # 计算spread和imbalance
        df['spread'] = df['ask_price'] - df['bid_price']
        df['orderbook_imbalance'] = (df['bid_size'] - df['ask_size']) / (df['bid_size'] + df['ask_size'] + 1e-6)
        
        # 对spread和imbalance计算rolling统计量 (tick级)
        for n in self.tick_windows:
            factors[f'spread__{n}tick__mean'] = df['spread'].rolling(n).mean()
            factors[f'spread__{n}tick__var'] = df['spread'].rolling(n).var()
            factors[f'orderflow__imbalance__{n}tick__mean'] = df['orderbook_imbalance'].rolling(n).mean()
            factors[f'orderflow__imbalance__{n}tick__var'] = df['orderbook_imbalance'].rolling(n).var()
        
        # 对spread和imbalance计算rolling统计量 (秒级)
        for n in self.time_windows:
            window = f'{n}s'
            factors[f'spread__{window}__mean'] = df['spread'].rolling(window).mean()
            factors[f'spread__{window}__var'] = df['spread'].rolling(window).var()
            factors[f'orderflow__imbalance__{window}__mean'] = df['orderbook_imbalance'].rolling(window).mean()
            factors[f'orderflow__imbalance__{window}__var'] = df['orderbook_imbalance'].rolling(window).var()
        
        # 对spread和imbalance计算rolling统计量 (分钟级)
        for n in self.minute_windows:
            window = f'{n}min'
            factors[f'spread__{window}__mean'] = df['spread'].rolling(window).mean()
            factors[f'spread__{window}__var'] = df['spread'].rolling(window).var()
            factors[f'orderflow__imbalance__{window}__mean'] = df['orderbook_imbalance'].rolling(window).mean()
            factors[f'orderflow__imbalance__{window}__var'] = df['orderbook_imbalance'].rolling(window).var()
        
        # 今日开盘以来的统计量
        df['date_group'] = df.index.date
        for window in self.day_windows:
            factors[f'spread__{window}__mean'] = df.groupby('date_group')['spread'].transform('mean')
            factors[f'spread__{window}__var'] = df.groupby('date_group')['spread'].transform('var')
            factors[f'orderflow__imbalance__{window}__mean'] = df.groupby('date_group')['orderbook_imbalance'].transform('mean')
            factors[f'orderflow__imbalance__{window}__var'] = df.groupby('date_group')['orderbook_imbalance'].transform('var')
        
        return pd.DataFrame(factors)
    
    def _calculate_order_dynamic_factors(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算挂单动态类因子"""
        factors = {}
        
        # ask/bid价格和挂单量变化
        for side in ['ask', 'bid']:
            price_col = f'{side}_price'
            size_col = f'{side}_size'
            
            # tick级变化
            factors[f'order__{side}_price__1tick__diff'] = df[price_col].diff()
            factors[f'order__{side}_size__1tick__diff'] = df[size_col].diff()
            
            # tick级统计量
            for n in self.tick_windows:
                factors[f'order__{side}_price__{n}tick__mean'] = df[price_col].rolling(n).mean()
                factors[f'order__{side}_price__{n}tick__var'] = df[price_col].rolling(n).var()
                factors[f'order__{side}_size__{n}tick__mean'] = df[size_col].rolling(n).mean()
                factors[f'order__{side}_size__{n}tick__var'] = df[size_col].rolling(n).var()
            
            # 秒级统计量
            for n in self.time_windows:
                window = f'{n}s'
                factors[f'order__{side}_price__{window}__mean'] = df[price_col].rolling(window).mean()
                factors[f'order__{side}_price__{window}__var'] = df[price_col].rolling(window).var()
                factors[f'order__{side}_size__{window}__mean'] = df[size_col].rolling(window).mean()
                factors[f'order__{side}_size__{window}__var'] = df[size_col].rolling(window).var()
            
            # 分钟级统计量
            for n in self.minute_windows:
                window = f'{n}min'
                factors[f'order__{side}_price__{window}__mean'] = df[price_col].rolling(window).mean()
                factors[f'order__{side}_price__{window}__var'] = df[price_col].rolling(window).var()
                factors[f'order__{side}_size__{window}__mean'] = df[size_col].rolling(window).mean()
                factors[f'order__{side}_size__{window}__var'] = df[size_col].rolling(window).var()
            
            # 今日开盘以来
            df['date_group'] = df.index.date
            for window in self.day_windows:
                factors[f'order__{side}_price__{window}__mean'] = df.groupby('date_group')[price_col].transform('mean')
                factors[f'order__{side}_price__{window}__var'] = df.groupby('date_group')[price_col].transform('var')
                factors[f'order__{side}_size__{window}__mean'] = df.groupby('date_group')[size_col].transform('mean')
                factors[f'order__{side}_size__{window}__var'] = df.groupby('date_group')[size_col].transform('var')
        
        return pd.DataFrame(factors)
    
    def _calculate_microstructure_factors(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算微观结构特征
        
        新增特征：
        - hidden_order_ratio: 隐藏订单比例
        - cancel_rate: 撤单率
        - best_bid_price_stickiness: 最优买价粘性
        - price_response_to_trade: 成交对价格的影响
        - volume_order_imbalance_ratio: 成交量订单不平衡比率
        - order_dynamics: 订单动态变化率
        - queue_growth: 队列增长趋势
        - order_imbalance_velocity: 订单不平衡变化速度
        
        预期行为：
        - 高隐藏订单比例预示潜在流动性
        - 高撤单率预示市场不确定性
        - 高价格粘性预示市场稳定性
        - 高价格响应预示市场敏感性
        - 高不平衡比率预示市场压力
        - 高订单动态变化预示市场活跃度
        - 高队列增长预示市场深度
        - 高不平衡速度预示市场压力变化
        """
        factors = {}
        
        try:
            # 计算mid_price
            df['mid_price'] = (df['ask_price'] + df['bid_price']) / 2
            
            # 1. hidden_order_ratio
            if 'hidden_size' in df.columns and 'visible_size' in df.columns:
                factors['micro__hidden_order_ratio'] = (
                    df['hidden_size'] / (df['hidden_size'] + df['visible_size'] + 1e-6)
                )
            
            # 2. cancel_rate
            if 'cancel_orders' in df.columns and 'add_orders' in df.columns:
                factors['micro__cancel_rate'] = (
                    df['cancel_orders'] / (df['cancel_orders'] + df['add_orders'] + 1e-6)
                )
                
                # 新增：订单动态变化率
                for window in ['1s', '5s']:
                    factors[f'micro__order_dynamics_{window}'] = (
                        (df['add_orders'] - df['cancel_orders']).rolling(window).mean()
                    )
            
            # 3. best_bid_price_stickiness
            for window in self.time_windows:
                window_str = f'{window}s'
                price_changes = (df['bid_price'].diff() != 0).astype(int)
                factors[f'micro__best_bid_stickiness_{window_str}'] = (
                    1 - price_changes.rolling(window_str).mean()
                )
            
            # 4. price_response_to_trade
            for window in self.time_windows:
                window_str = f'{window}s'
                price_change = df['mid_price'].diff()
                trade_direction = (df['trade_price'] > df['mid_price']).astype(int) * 2 - 1
                factors[f'micro__price_response_{window_str}'] = (
                    price_change * trade_direction
                ).rolling(window_str).mean()
            
            # 5. volume_order_imbalance_ratio
            for window in self.time_windows:
                window_str = f'{window}s'
                volume_imbalance = (
                    df['trade_size'] * (df['trade_price'] > df['mid_price']).astype(int) -
                    df['trade_size'] * (df['trade_price'] < df['mid_price']).astype(int)
                )
                total_volume = df['trade_size'].rolling(window_str).sum()
                factors[f'micro__volume_imbalance_ratio_{window_str}'] = (
                    volume_imbalance.rolling(window_str).sum() / (total_volume + 1e-6)
                )
            
            # 新增：队列增长趋势
            for side in ['bid', 'ask']:
                size_col = f'{side}_size'
                for window in ['1s', '5s']:
                    # 计算连续增长趋势
                    size_diff = df[size_col].diff()
                    factors[f'micro__{side}_queue_growth_{window}'] = (
                        size_diff.rolling(window).sum() / 
                        (df[size_col].rolling(window).mean() + 1e-6)
                    )
            
            # 新增：订单不平衡变化速度
            for window in ['1s', '5s']:
                imbalance = (df['bid_size'] - df['ask_size']) / (df['bid_size'] + df['ask_size'] + 1e-6)
                factors[f'micro__order_imbalance_velocity_{window}'] = (
                    imbalance.diff().rolling(window).mean()
                )
            
        except Exception as e:
            logger.warning(f"Error calculating microstructure features: {str(e)}")
            return pd.DataFrame()
        
        return pd.DataFrame(factors)
        
    def _calculate_cross_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算特征交叉组合，动态选择Top20因子进行两两组合
        
        实现步骤：
        1. 获取所有标准化后的因子
        2. 计算每个因子的rolling均值
        3. 选择Top20均值最大的因子
        4. 两两组合生成交叉特征
        5. 控制特征数量不超过100个
        """
        try:
            logger.info("Starting cross interaction feature calculation...")
            
            # 获取所有标准化后的因子列
            factor_cols = [col for col in df.columns if any([
                col.startswith('price__'),
                col.startswith('spread__'),
                col.startswith('trade__'),
                col.startswith('liquidity__'),
                col.startswith('momentum__'),
                col.startswith('orderflow__'),
                col.startswith('order__'),
                col.startswith('micro__'),
                col.startswith('intraday__'),
                col.startswith('orderbook__')
            ])]
            
            if not factor_cols:
                logger.warning("No base factors found for cross interaction")
                return pd.DataFrame()
            
            logger.info(f"Found {len(factor_cols)} base factors for cross interaction")
            
            # 计算每个因子的rolling均值
            logger.info("Calculating rolling means for base factors...")
            rolling_means = {}
            for col in tqdm(factor_cols, desc="Calculating rolling means"):
                try:
                    rolling_means[col] = df[col].rolling(100, min_periods=50).mean().mean()
                except Exception as e:
                    logger.warning(f"Error calculating rolling mean for {col}: {str(e)}")
                    continue
            
            if not rolling_means:
                logger.warning("No valid rolling means calculated")
                return pd.DataFrame()
            
            # 选择Top20均值最大的因子
            logger.info("Selecting top factors...")
            top_factors = sorted(rolling_means.items(), key=lambda x: x[1], reverse=True)[:20]
            top_factor_cols = [col for col, _ in top_factors]
            logger.info(f"Selected {len(top_factor_cols)} top factors")
            
            # 生成交叉特征
            logger.info("Generating cross features...")
            cross_features = {}
            total_combinations = len(top_factor_cols) * (len(top_factor_cols) - 1) // 2
            
            with tqdm(total=total_combinations, desc="Generating cross features") as pbar:
                for i in range(len(top_factor_cols)):
                    for j in range(i+1, len(top_factor_cols)):
                        col1, col2 = top_factor_cols[i], top_factor_cols[j]
                        try:
                            feature_name = f'cross__{col1}__{col2}'
                            cross_features[feature_name] = df[col1] * df[col2]
                            pbar.update(1)
                            
                            # 控制特征数量不超过100个
                            if len(cross_features) >= 100:
                                break
                        except Exception as e:
                            logger.warning(f"Error generating cross feature {col1} * {col2}: {str(e)}")
                            continue
                    if len(cross_features) >= 100:
                        break
            
            if not cross_features:
                logger.warning("No cross features generated")
                return pd.DataFrame()
            
            logger.info(f"Generated {len(cross_features)} cross features")
            return pd.DataFrame(cross_features)
            
        except Exception as e:
            logger.error(f"Error calculating cross features: {str(e)}")
            return pd.DataFrame()
        
    def _standardize_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        对特征进行标准化处理
        
        Args:
            df (pd.DataFrame): 输入的特征DataFrame
            
        Returns:
            pd.DataFrame: 标准化后的特征DataFrame
        """
        try:
            logger.info("Starting feature standardization...")
            
            # 复制输入数据
            standardized_df = df.copy()
            
            # 获取所有特征列
            feature_cols = [col for col in df.columns if not col.startswith('target_')]
            
            # 分批处理特征
            batch_size = 20
            for i in range(0, len(feature_cols), batch_size):
                batch_cols = feature_cols[i:i+batch_size]
                logger.debug(f"Processing batch {i//batch_size + 1}/{(len(feature_cols)-1)//batch_size + 1}")
                
                for col in batch_cols:
                    try:
                        # 计算滚动统计量（使用100个样本点的窗口）
                        rolling_mean = df[col].rolling(window=100, min_periods=50).mean().shift(1)
                        rolling_std = df[col].rolling(window=100, min_periods=50).std().shift(1)
                        
                        # 处理标准差过小的情况
                        rolling_std = rolling_std.clip(lower=1e-8)
                        
                        # 计算z-score
                        standardized_df[col] = (df[col] - rolling_mean) / rolling_std
                        
                        # 处理无穷大和NaN
                        standardized_df[col] = standardized_df[col].replace([np.inf, -np.inf], np.nan)
                        
                    except Exception as e:
                        logger.warning(f"Error standardizing feature {col}: {str(e)}")
                        standardized_df[col] = df[col]  # 保持原值
                        
                # 清理内存
                gc.collect()
                
            logger.info("Feature standardization completed")
            return standardized_df
            
        except Exception as e:
            logger.error(f"Error in feature standardization: {str(e)}")
            return df  # 发生错误时返回原始数据
    
    def drop_highly_correlated_features(self, df: pd.DataFrame, threshold: float = 0.95) -> pd.DataFrame:
        """
        基于Pearson和Spearman相关系数去除高度相关的特征
        
        Args:
            df (pd.DataFrame): 输入数据框
            threshold (float): 相关系数阈值，默认0.95
            
        Returns:
            pd.DataFrame: 去除冗余特征后的数据框
        """
        try:
            # 分离因子列和目标列
            factor_cols = [col for col in df.columns if not col.startswith('target_')]
            target_cols = [col for col in df.columns if col.startswith('target_')]
            
            if not factor_cols:
                logger.warning("No factor columns found to process")
                return df
            
            # 1. 初步筛选：使用Pearson相关性快速计算
            logger.info("Starting initial Pearson correlation check...")
            try:
                # 计算Pearson相关系数矩阵
                pearson_corr = df[factor_cols].corr(method='pearson')
                # 将相关系数矩阵转换为numpy数组以加速计算
                corr_array = pearson_corr.to_numpy()
                np.fill_diagonal(corr_array, 0)  # 对角线设为0
                
                # 使用集合存储要删除的特征
                to_drop = set()
                
                # 使用numpy的向量化操作加速
                for i in tqdm(range(len(factor_cols)), desc="Checking Pearson correlations"):
                    if factor_cols[i] in to_drop:
                        continue
                        
                    # 找到与当前特征高度相关的其他特征
                    high_corr = np.where(np.abs(corr_array[i]) > threshold)[0]
                    
                    # 保留方差最大的特征
                    if len(high_corr) > 0:
                        variances = df[factor_cols].iloc[:, high_corr].var()
                        max_var_feature = variances.idxmax()
                        to_drop.update([f for f in variances.index if f != max_var_feature])
                
                # 更新因子列
                remaining_factors = [f for f in factor_cols if f not in to_drop]
                logger.info(f"Initial Pearson check removed {len(to_drop)} features")
                
            except Exception as e:
                logger.warning(f"Error in Pearson correlation check: {str(e)}")
                remaining_factors = factor_cols
                
            # 2. 精细复检：对剩余特征使用Spearman相关性
            if len(remaining_factors) > 100:
                logger.info("Too many features remaining for Spearman check, skipping...")
            else:
                logger.info("Starting detailed Spearman correlation check...")
                try:
                    # 计算Spearman相关系数矩阵
                    spearman_corr = df[remaining_factors].corr(method='spearman')
                    corr_array = spearman_corr.to_numpy()
                    np.fill_diagonal(corr_array, 0)
                    
                    # 使用更严格的阈值
                    strict_threshold = min(threshold - 0.02, 0.93)
                    
                    # 使用集合存储要删除的特征
                    to_drop_spearman = set()
                    
                    # 使用numpy的向量化操作加速
                    for i in tqdm(range(len(remaining_factors)), desc="Checking Spearman correlations"):
                        if remaining_factors[i] in to_drop_spearman:
                            continue
                            
                        high_corr = np.where(np.abs(corr_array[i]) > strict_threshold)[0]
                        
                        if len(high_corr) > 0:
                            variances = df[remaining_factors].iloc[:, high_corr].var()
                            max_var_feature = variances.idxmax()
                            to_drop_spearman.update([f for f in variances.index if f != max_var_feature])
                    
                    # 更新因子列
                    remaining_factors = [f for f in remaining_factors if f not in to_drop_spearman]
                    logger.info(f"Spearman check removed {len(to_drop_spearman)} additional features")
                    
                except Exception as e:
                    logger.warning(f"Error in Spearman correlation check: {str(e)}")
            
            # 3. 合并结果
            final_cols = remaining_factors + target_cols
            result_df = df[final_cols]
            
            # 输出最终结果
            total_dropped = len(factor_cols) - len(remaining_factors)
            logger.info(f"Removed {total_dropped} redundant features based on correlation analysis")
            logger.info(f"Final number of features: {len(final_cols)}")
            
            return result_df
            
        except Exception as e:
            logger.error(f"Error in drop_highly_correlated_features: {str(e)}")
            return df
    
    def evaluate_factors(self, df: pd.DataFrame) -> Dict:
        """
        评估因子的统计特征
        
        Args:
            df (pd.DataFrame): 包含因子的DataFrame
            
        Returns:
            Dict: 包含每个因子统计信息的字典
        """
        # 获取所有因子列
        factor_columns = [col for col in df.columns if any([
            col.startswith('momentum__'),
            col.startswith('spread__'),
            col.startswith('orderflow__'),
            col.startswith('trade__'),
            col.startswith('order__'),
            col.startswith('execution__'),
            col.startswith('minute__'),
            col.startswith('micro__'),
            col.startswith('cross__')
        ])]
        
        stats = {}
        for factor in factor_columns:
            if factor in df.columns:
                # 检查是否为分类变量
                if pd.api.types.is_categorical_dtype(df[factor]):
                    stats[factor] = {
                        'unique_values': df[factor].nunique(),
                        'value_counts': df[factor].value_counts().to_dict(),
                        'null_count': df[factor].isnull().sum()
                    }
                else:
                    stats[factor] = {
                        'mean': df[factor].mean(),
                        'std': df[factor].std(),
                        'min': df[factor].min(),
                        'max': df[factor].max(),
                        'null_count': df[factor].isnull().sum()
                    }
        
        return stats

    def _calculate_intraday_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算日内特征
        
        新增特征：
        - cumulative_volume_today: 当天累计成交量
        - cumulative_return_today: 当天累计mid_price变化（相对开盘）
        """
        factors = {}
        
        try:
            # 确保数据按时间排序
            df = df.sort_index()
            
            # 计算mid_price
            df['mid_price'] = (df['ask_price'] + df['bid_price']) / 2
            
            # 1. cumulative_volume_today: 当天累计成交量
            # 按日期分组，计算累计成交量
            df['date'] = df.index.date
            factors['intraday__cumulative_volume'] = (
                df.groupby('date')['trade_size']
                .transform(lambda x: x.cumsum())
            )
            
            # 2. cumulative_return_today: 当天累计mid_price变化
            # 获取每日开盘价
            daily_open = df.groupby('date')['mid_price'].transform('first')
            # 计算相对于开盘价的收益率
            factors['intraday__cumulative_return'] = (
                (df['mid_price'] - daily_open) / daily_open
            )
            
            # 清理临时列
            df.drop(['date'], axis=1, inplace=True)
            
        except Exception as e:
            logger.warning(f"Error calculating intraday features: {str(e)}")
            return pd.DataFrame()
            
        return pd.DataFrame(factors) 