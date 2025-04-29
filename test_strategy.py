import os
import sys

# Add current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import deque
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass
from tqdm import tqdm
from strategy import StrategyExecutor, TransactionRecorder, TCAEvaluator, MarketState, ExecutionRecord

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_strategy(symbol: str, execution_side: str) -> None:
    """测试策略执行"""
    try:
        # 确保results目录存在
        os.makedirs('results', exist_ok=True)
        
        # 加载数据
        data_path = f'processed_data/df_with_scoring_{symbol}.feather'
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data file not found: {data_path}")
            
        df = pd.read_feather(data_path)
        # 确保索引是datetime类型
        if 'timestamp' in df.columns:
            df.set_index('timestamp', inplace=True)
        df.index = pd.to_datetime(df.index)
        logger.info(f"Loaded data with shape: {df.shape}")
        
        # 检查必要的列是否存在
        required_cols = ['ask_price', 'bid_price', 'mid_price', 'spread']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
            
        # 检查scoring列
        scoring_cols = [col for col in df.columns if col.startswith('scoring_')]
        if not scoring_cols:
            raise ValueError(f"No scoring columns found. Available columns: {df.columns.tolist()}")
        logger.info(f"Found {len(scoring_cols)} scoring columns: {scoring_cols}")
        logger.info("策略决策严格基于scoring列，无未来数据引用")
        
        # 初始化策略执行器
        executor = StrategyExecutor(execution_side=execution_side)
        
        # 执行策略
        execution_records = executor.execute_strategy(df, symbol)
        
        if not execution_records:
            logger.warning(f"No trades executed for {symbol} {execution_side}")
            return
            
        # 转换为DataFrame
        records_df = pd.DataFrame([{
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
        } for r in execution_records])
        
        # 初始化评估器
        evaluator = TCAEvaluator(records_df, df, execution_side)
        
        # 生成报告
        report = evaluator.generate_report()
        
        # 保存报告
        report_path = f'results/{symbol}_{execution_side}_results.txt'
        with open(report_path, 'w') as f:
            f.write(report)
        logger.info(f"Results saved to {report_path}")
        
        # 生成图表
        evaluator.plot_execution_quality(symbol, execution_side)
        
    except Exception as e:
        logger.error(f"Error testing strategy for {symbol} {execution_side}: {str(e)}")
        raise

def test_all_strategies():
    """测试所有股票和交易方向的策略"""
    stocks = ['AMZN', 'GOOG', 'MSFT', 'INTC']  # 测试剩余四个股票
    sides = ['buy', 'sell']
    
    for stock in stocks:
        for side in sides:
            logger.info(f"\n=== Testing {stock} {side} strategy ===")
            try:
                test_strategy(stock, side)
            except Exception as e:
                logger.error(f"Failed to test {stock} {side} strategy: {str(e)}")
                continue

if __name__ == "__main__":
    test_all_strategies() 