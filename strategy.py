import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import deque
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class MarketState:
    """市场状态数据结构"""
    spread: float
    mid_vol_1s: float
    mid_vol_5s: float
    order_flow_imbalance: float
    timestamp: datetime

@dataclass
class ExecutionRecord:
    """交易执行记录数据结构"""
    execution_time: datetime
    execution_price: float
    execution_vs_twap: float
    execution_vs_best: float
    execution_spread: float
    pre_5s_mid_drift: float
    market_volatility_1s: float
    market_volatility_5s: float
    scoring: float
    market_state: MarketState
    is_forced: bool

class StrategyExecutor:
    def __init__(self, 
                 execution_side: str = 'buy',
                 scoring_quantile: float = 0.7,
                 spread_quantile: float = 0.7,
                 vol_quantile: float = 0.7,
                 lookback_ticks: int = 100,
                 scoring_weights: Tuple[float, float] = (0.5, 0.5)):
        """
        初始化策略执行器
        
        Args:
            execution_side: 执行方向，'buy'或'sell'
            scoring_quantile: scoring历史分布的分位数，用于计算base threshold
            spread_quantile: spread历史分布的分位数上限
            vol_quantile: 波动率历史分布的分位数上限
            lookback_ticks: 回看tick数，用于计算历史分布
            scoring_weights: 两个scoring特征的权重
        """
        if execution_side not in ['buy', 'sell']:
            raise ValueError("execution_side must be either 'buy' or 'sell'")
            
        self.execution_side = execution_side
        self.scoring_quantile = scoring_quantile
        self.spread_quantile = spread_quantile
        self.vol_quantile = vol_quantile
        self.lookback_ticks = lookback_ticks
        self.scoring_weights = scoring_weights
        
        # 用于存储市场状态历史
        self.market_state_history = deque(maxlen=lookback_ticks)
        
        # 当前分钟的状态
        self.current_minute = None
        self.current_minute_start = None
        self.ticks_in_current_minute = 0
        self.total_ticks_in_minute = 0
        self.has_executed_this_minute = False
        
        # 存储每分钟的TWAP
        self.minute_twap = {}
        
        # 市场状态阈值
        self.spread_median = None
        self.volatility_median = None
        self.spread_80p = None
        self.volatility_80p = None
        self.imbalance_threshold = 0.7  # 订单流失衡阈值

    def _calculate_market_state(self, df: pd.DataFrame, current_idx: int) -> MarketState:
        """计算当前市场状态"""
        current_time = df.index[current_idx]
        
        # 预计算滚动标准差
        if not hasattr(self, '_mid_vol_1s'):
            self._mid_vol_1s = df['mid_price'].rolling(window=20, min_periods=1).std()
            self._mid_vol_5s = df['mid_price'].rolling(window=100, min_periods=1).std()
        
        # 使用预计算的值
        mid_vol_1s = self._mid_vol_1s.iloc[current_idx]
        mid_vol_5s = self._mid_vol_5s.iloc[current_idx]
        
        # 计算order flow imbalance
        order_flow_imbalance = (
            df['bid_size'].iloc[current_idx] - df['ask_size'].iloc[current_idx]
        ) / (df['bid_size'].iloc[current_idx] + df['ask_size'].iloc[current_idx] + 1e-6)
        
        return MarketState(
            spread=df['spread'].iloc[current_idx],
            mid_vol_1s=mid_vol_1s,
            mid_vol_5s=mid_vol_5s,
            order_flow_imbalance=order_flow_imbalance,
            timestamp=current_time
        )
    
    def _get_market_state(self, row: pd.Series) -> str:
        """
        根据当前tick的特征判定市场状态
        
        Args:
            row: 当前tick的数据行
            
        Returns:
            str: 市场状态，包括：
                - 'low_vol_low_spread': 低波动低价差
                - 'high_vol_high_spread': 高波动高价差
                - 'unbalanced_flow': 订单流失衡
                - 'normal': 正常状态
        """
        if self.spread_median is None:
            # 初始化阈值
            self.spread_median = 0.01  # 默认值
            self.volatility_median = 0.001  # 默认值
            self.spread_80p = 0.02  # 默认值
            self.volatility_80p = 0.002  # 默认值
        
        spread = row['spread']
        volatility = row['price__volatility_1s']
        imbalance = row['orderflow__imbalance__2s__mean']  # 使用2s的imbalance
        
        if spread < self.spread_median and volatility < self.volatility_median:
            return 'low_vol_low_spread'
        elif spread > self.spread_80p and volatility > self.volatility_80p:
            return 'high_vol_high_spread'
        elif abs(imbalance) > self.imbalance_threshold:
            return 'unbalanced_flow'
        else:
            return 'normal'

    def _get_scoring_columns(self, state: str) -> Tuple[str, Optional[str]]:
        """
        根据市场状态动态选择scoring列组合
        
        Args:
            state: 市场状态
            
        Returns:
            Tuple[str, Optional[str]]: (主评分列, 辅助评分列)
        """
        if self.execution_side == 'buy':
            main_col = 'scoring_buy_twap_diff'
            aux_col = 'scoring_ask_margin'
        else:
            main_col = 'scoring_sell_twap_diff'
            aux_col = 'scoring_bid_margin'
        return main_col, aux_col

    def _calculate_base_threshold(self, df: pd.DataFrame, current_idx: int) -> float:
        """计算base threshold，基于过去N个tick的scoring分布"""
        # 获取回看窗口的起始索引
        lookback_start = max(0, current_idx - self.lookback_ticks)
        
        # 获取scoring列
        main_col, aux_col = self._get_scoring_columns(self._get_market_state(df.iloc[current_idx]))
        if main_col not in df.columns:
            raise ValueError(f"Missing main scoring column: {main_col}")
            
        # 计算过去N个tick的scoring分位数
        lookback_scoring = df[main_col].iloc[lookback_start:current_idx]
        if len(lookback_scoring) > 0:
            base_threshold = lookback_scoring.quantile(self.scoring_quantile)
            
            # 动态环境调整：在波动率异常高时提高阈值
            if len(self.market_state_history) > 0:
                vol_875th = np.quantile([s.mid_vol_1s for s in self.market_state_history], 0.875)
                current_vol = self.market_state_history[-1].mid_vol_1s
                if current_vol > vol_875th:
                    base_threshold *= 1.15
                    
            return base_threshold
        return 0.5  # 默认值
        
    def _get_dynamic_threshold(self, base_threshold: float, relative_time: float, state: str) -> float:
        """
        动态threshold调整
        
        Args:
            base_threshold: 基础阈值
            relative_time: 相对时间（0-1）
            state: 市场状态
            
        Returns:
            float: 调整后的阈值
        """
        time_decay = (1 - relative_time ** 3)

        if state == 'low_vol_low_spread':
            multiplier = 1.0  # 正常标准
        elif state == 'high_vol_high_spread':
            multiplier = 0.8  # 放宽
        elif state == 'unbalanced_flow':
            multiplier = 0.9
        else:
            multiplier = 1.0

        return base_threshold * time_decay * multiplier

    def _calculate_combined_scoring(self, df: pd.DataFrame, i: int, state: str) -> float:
        """
        计算组合scoring值
        
        Args:
            df: 数据DataFrame
            i: 当前索引
            state: 市场状态
            
        Returns:
            float: 组合评分
        """
        main_col, aux_col = self._get_scoring_columns(state)
        
        # 获取主scoring值
        main_scoring = df[main_col].iloc[i]
        
        # 如果辅助scoring列存在，计算加权平均
        if aux_col in df.columns:
            aux_scoring = df[aux_col].iloc[i]
            # 如果spread_drop列存在，也加入组合
            if 'scoring_target_spread_narrowing_5s' in df.columns:
                spread_drop = df['scoring_target_spread_narrowing_5s'].iloc[i]
                # 使用三个特征的加权平均，调整权重
                return (0.5 * main_scoring +    # 主特征权重0.5
                        0.25 * aux_scoring +    # 辅助特征权重0.25
                        0.25 * spread_drop)     # spread_drop权重0.25
            else:
                # 使用两个特征的加权平均
                return (self.scoring_weights[0] * main_scoring + 
                        self.scoring_weights[1] * aux_scoring)
        else:
            return main_scoring

    def _should_execute(self, df: pd.DataFrame, i: int, market_state: MarketState) -> Tuple[bool, bool]:
        """判断是否应该执行交易"""
        current_time = df.index[i]
        # 确保索引是datetime类型
        if not isinstance(current_time, pd.Timestamp):
            current_time = pd.to_datetime(current_time)
        current_minute = current_time.floor('min')
        
        # 如果是新的一分钟，更新状态
        if current_minute != self.current_minute:
            self.current_minute = current_minute
            self.has_executed_this_minute = False
            # 确保索引是datetime类型
            df.index = pd.to_datetime(df.index)
            self.total_ticks_in_minute = len(df[df.index.floor('min') == current_minute])
            self.ticks_in_current_minute = 0
        
        self.ticks_in_current_minute += 1
        
        # 检查是否已经在这一分钟执行过交易
        if self.has_executed_this_minute:
            return False, False
            
        # 计算base threshold
        base_threshold = self._calculate_base_threshold(df, i)
        
        # 获取市场状态
        state = self._get_market_state(df.iloc[i])
        
        # 计算相对时间
        relative_time = self.ticks_in_current_minute / max(self.total_ticks_in_minute, 1)
        
        # 计算动态阈值
        dynamic_threshold = self._get_dynamic_threshold(base_threshold, relative_time, state)
        
        # 计算组合评分
        combined_scoring = self._calculate_combined_scoring(df, i, state)
        
        # 判断是否执行
        should_execute = combined_scoring > dynamic_threshold
        is_forced = False  # 这里可以根据需要添加强制执行的逻辑
        
        return should_execute, is_forced
    
    def execute_strategy(self, df: pd.DataFrame, symbol: str = None) -> List[ExecutionRecord]:
        """
        执行策略
        
        Args:
            df: 包含特征、目标和scoring的DataFrame
            symbol: 股票代码，用于日志输出
            
        Returns:
            List[ExecutionRecord]: 执行记录列表
        """
        execution_records = []
        
        # 检查scoring列
        main_col, aux_col = self._get_scoring_columns(self._get_market_state(df.iloc[0]))
        if main_col not in df.columns:
            if symbol:
                logger.warning(f"Missing main scoring column {main_col} for {symbol}")
            return execution_records
            
        # 预计算每分钟的TWAP（使用每分钟最后一个tick的ask/bid）
        df['minute'] = df.index.floor('min')
        if self.execution_side == 'buy':
            self.minute_twap = df.groupby('minute')['ask_price'].last().to_dict()
        else:
            self.minute_twap = df.groupby('minute')['bid_price'].last().to_dict()
            
        # 计算总分钟数
        total_minutes = len(df['minute'].unique())
        logger.info(f"Total minutes in data: {total_minutes}")
        
        # 遍历每个tick
        for i in range(len(df)):
            current_time = df.index[i]
            current_minute = current_time.floor('min')
            
            # 计算市场状态
            market_state = self._calculate_market_state(df, i)
            self.market_state_history.append(market_state)
            
            # 使用预计算的scoring值
            scoring = df[main_col].iloc[i]
            
            # 判断是否应该执行
            should_execute, is_forced = self._should_execute(df, i, market_state)
            
            if should_execute or is_forced:
                # 标记这一分钟已经执行过交易
                self.has_executed_this_minute = True
                
                # 计算执行价格和相关指标
                execution_price = df['ask_price'].iloc[i] if self.execution_side == 'buy' else df['bid_price'].iloc[i]
                twap_price = self.minute_twap.get(current_minute, execution_price)
                
                # 计算执行前5秒的价格漂移
                pre_5s_start_idx = max(0, i - 100)  # 假设每秒20个tick
                pre_5s_mid_drift = (df['mid_price'].iloc[i] - df['mid_price'].iloc[pre_5s_start_idx]) / df['mid_price'].iloc[pre_5s_start_idx]
                
                # 创建执行记录
                record = ExecutionRecord(
                    execution_time=current_time,
                    execution_price=execution_price,
                    execution_vs_twap=(execution_price - twap_price) / twap_price,
                    execution_vs_best=0.0,  # 暂时设置为0，因为我们没有最优价格的信息
                    execution_spread=df['spread'].iloc[i],
                    pre_5s_mid_drift=pre_5s_mid_drift,
                    market_volatility_1s=market_state.mid_vol_1s,
                    market_volatility_5s=market_state.mid_vol_5s,
                    scoring=scoring,
                    market_state=market_state,
                    is_forced=is_forced
                )
                execution_records.append(record)
                
        # 检查是否每分钟都执行了交易
        executed_minutes = len(set(r.execution_time.floor('min') for r in execution_records))
        execution_ratio = executed_minutes / total_minutes
        logger.info(f"Execution ratio: {execution_ratio:.2%} ({executed_minutes}/{total_minutes} minutes)")
        
        return execution_records

class TransactionRecorder:
    def __init__(self):
        """初始化交易记录器"""
        self.records = []
        
    def add_record(self, record: ExecutionRecord):
        """添加交易记录"""
        self.records.append(record)
        
    def get_records_df(self) -> pd.DataFrame:
        """获取所有记录为DataFrame"""
        if not self.records:
            return pd.DataFrame()
            
        return pd.DataFrame([{
            'execution_time': r.execution_time,
            'execution_price': r.execution_price,
            'execution_vs_twap': r.execution_vs_twap,
            'execution_vs_best': r.execution_vs_best,
            'execution_spread': r.execution_spread,
            'pre_5s_mid_drift': r.pre_5s_mid_drift,
            'market_volatility_1s': r.market_volatility_1s,
            'market_volatility_5s': r.market_volatility_5s,
            'scoring': r.scoring,
            'is_forced': r.is_forced
        } for r in self.records])

class TCAEvaluator:
    def __init__(self, records_df: pd.DataFrame, df: pd.DataFrame, execution_side: str = 'buy'):
        """
        初始化TCA评估器
        
        Args:
            records_df: 包含执行记录的DataFrame
            df: 原始数据DataFrame
            execution_side: 执行方向，'buy'或'sell'
        """
        self.records_df = records_df
        self.df = df
        self.execution_side = execution_side
        
    def calculate_metrics(self) -> Dict[str, float]:
        """计算TCA指标"""
        metrics = {}
        
        # 总执行次数
        metrics['total_executions'] = len(self.records_df)
        
        # 优于TWAP的比例
        metrics['better_than_twap_ratio'] = (self.records_df['execution_vs_twap'] < 0).mean()
        
        # 优于最优价格的比例
        metrics['better_than_best_ratio'] = (self.records_df['execution_vs_best'] < 0).mean()
        
        # 计算累计PNL
        metrics['cumulative_pnl'] = self.records_df['execution_vs_twap'].sum()
        
        # 计算平均执行滑点
        metrics['avg_slippage'] = self.records_df['execution_vs_twap'].mean()
        
        # 计算强制执行的比率
        metrics['forced_execution_ratio'] = self.records_df['is_forced'].mean()
        
        return metrics
        
    def plot_execution_quality(self, symbol: str, execution_side: str):
        """绘制执行质量图表"""
        if len(self.records_df) == 0:
            logger.warning("No execution records to plot")
            return
            
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 设置配色方案
        mid_color = '#34495E'  # 深蓝灰色
        better_color = '#27AE60'  # 柔和的绿色
        worse_color = '#8E44AD'  # 柔和的紫色
        
        # 1. mid-price轨迹 + 执行点标记
        axes[0, 0].plot(self.df.index, self.df['mid_price'], label='Mid Price', 
                       color=mid_color, linewidth=1.5, alpha=0.7)
        
        # 标记执行点
        better_trades = self.records_df[self.records_df['execution_vs_twap'] < 0]
        worse_trades = self.records_df[self.records_df['execution_vs_twap'] >= 0]
        
        axes[0, 0].scatter(better_trades['execution_time'], better_trades['execution_price'], 
                          color=better_color, label='Better than TWAP',
                          alpha=0.6, s=8, marker='o')  # 更小的点，使用圆形标记
        axes[0, 0].scatter(worse_trades['execution_time'], worse_trades['execution_price'], 
                          color=worse_color, label='Worse than TWAP',
                          alpha=0.6, s=8, marker='o')  # 更小的点，使用圆形标记
        
        axes[0, 0].set_title('Price Trajectory and Execution Points', fontsize=12, pad=10)
        axes[0, 0].set_xlabel('Time', fontsize=10)
        axes[0, 0].set_ylabel('Price', fontsize=10)
        axes[0, 0].legend(fontsize=9)
        axes[0, 0].grid(True, alpha=0.2)
        
        # 2. 累计打败TWAP的PNL曲线
        cumulative_pnl = self.records_df['execution_vs_twap'].cumsum()
        axes[0, 1].plot(self.records_df['execution_time'], cumulative_pnl, color=mid_color)
        axes[0, 1].set_title('Cumulative PNL vs TWAP')
        axes[0, 1].set_xlabel('Time')
        axes[0, 1].set_ylabel('Cumulative PNL (bps)')
        
        # 3. 成交滑点 vs market volatility
        axes[1, 0].scatter(self.records_df['market_volatility_1s'], 
                         self.records_df['execution_vs_twap'] * 10000,
                         color=mid_color, alpha=0.6, s=8)
        axes[1, 0].set_title('Execution Slippage vs Market Volatility')
        axes[1, 0].set_xlabel('1s Volatility')
        axes[1, 0].set_ylabel('Slippage (bps)')
        
        # 4. 执行时的scoring分布
        axes[1, 1].hist(self.records_df['scoring'], bins=50, color=mid_color, alpha=0.7)
        axes[1, 1].set_title('Execution Scoring Distribution')
        axes[1, 1].set_xlabel('Scoring')
        axes[1, 1].set_ylabel('Count')
        
        # 设置整体样式
        for ax in axes.flat:
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.grid(True, alpha=0.2)
        
        plt.tight_layout()
        
        # 使用正确的文件名格式保存图表
        output_path = f'results/{symbol}_{execution_side}_execution_quality.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Execution quality plot saved to {output_path}")
        
    def generate_report(self) -> str:
        """生成TCA报告"""
        metrics = self.calculate_metrics()
        
        report = []
        report.append("Transaction Cost Analysis Report")
        report.append("=" * 30)
        report.append(f"Total Executions: {metrics['total_executions']}")
        report.append(f"Better than TWAP Ratio: {metrics['better_than_twap_ratio']:.2%}")
        report.append(f"Better than Best Ratio: {metrics['better_than_best_ratio']:.2%}")
        report.append(f"Cumulative PNL: {metrics['cumulative_pnl']:.2f} bps")
        report.append(f"Average Slippage: {metrics['avg_slippage']:.2f} bps")
        report.append(f"Forced Execution Ratio: {metrics['forced_execution_ratio']:.2%}")
        
        # 添加市场状态统计
        if len(self.records_df) > 0:
            report.append("\nMarket State Statistics")
            report.append("-" * 30)
            report.append(f"Average Spread: {self.records_df['execution_spread'].mean():.6f}")
            report.append(f"Average 1s Volatility: {self.records_df['market_volatility_1s'].mean():.6f}")
            report.append(f"Average 5s Volatility: {self.records_df['market_volatility_5s'].mean():.6f}")
        
        return "\n".join(report) 