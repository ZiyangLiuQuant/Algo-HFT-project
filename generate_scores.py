import pandas as pd
import numpy as np
import os
from tqdm import tqdm

def add_noise_and_drift(score, noise_scale=0.1, drift_scale=0.05):
    """添加随机噪声和漂移"""
    noise = np.random.normal(0, noise_scale, len(score))
    drift = np.linspace(0, drift_scale, len(score))
    return score * (1 + noise + drift)

def generate_scores_for_stock(symbol, aapl_scores, noise_scale=0.1, drift_scale=0.05):
    """为指定股票生成评分数据"""
    print(f"\nGenerating scores for {symbol}...")
    
    # 读取原始数据
    processed_path = f'processed_data/{symbol}_processed.feather'
    if not os.path.exists(processed_path):
        processed_path = f'processed_data/{symbol}_processed.csv'
    
    df = pd.read_feather(processed_path) if processed_path.endswith('.feather') else pd.read_csv(processed_path)
    
    # 确保时间戳列存在
    if 'timestamp' not in df.columns and 'time' in df.columns:
        df['timestamp'] = pd.to_datetime(df['time'])
    
    # 添加目标列
    target_columns = [
        'target_twap_diff',
        'target_is_better_than_twap',
        'target_buy_better_than_twap',
        'target_sell_better_than_twap',
        'target_best_future_ask',
        'target_best_future_bid',
        'target_ticks_until_best_ask',
        'target_ticks_until_best_bid'
    ]
    
    for col in target_columns:
        if col not in df.columns:
            df[col] = np.nan
    
    # 添加评分列
    score_columns = [
        'scoring_ask_margin',
        'scoring_bid_margin',
        'scoring_buy_twap_diff',
        'scoring_sell_twap_diff',
        'scoring_spread_drop'
    ]
    
    # 基于AAPL的评分生成新评分
    for col in score_columns:
        if col in aapl_scores.columns:
            base_scores = aapl_scores[col].values
            # 确保新生成的评分长度与原始数据匹配
            if len(df) > len(base_scores):
                base_scores = np.tile(base_scores, len(df) // len(base_scores) + 1)[:len(df)]
            else:
                base_scores = base_scores[:len(df)]
            
            # 添加噪声和漂移
            df[col] = add_noise_and_drift(base_scores, noise_scale, drift_scale)
    
    # 添加必要的特征列
    required_features = [
        'price__volatility_1s',
        'price__volatility_5s',
        'price__volatility_30s',
        'spread__volatility_1s',
        'spread__volatility_5s',
        'spread__volatility_30s',
        'orderflow__imbalance__2s__mean',
        'orderflow__imbalance__5s__mean',
        'orderflow__imbalance__30s__mean'
    ]
    
    # 基于AAPL的特征生成新特征
    for col in required_features:
        if col in aapl_scores.columns:
            base_values = aapl_scores[col].values
            if len(df) > len(base_values):
                base_values = np.tile(base_values, len(df) // len(base_values) + 1)[:len(df)]
            else:
                base_values = base_values[:len(df)]
            df[col] = add_noise_and_drift(base_values, noise_scale=0.05, drift_scale=0.02)
        else:
            # 如果AAPL数据中没有这个特征，生成随机值
            df[col] = np.random.normal(0, 0.01, len(df))
    
    # 保存结果
    output_path = f'processed_data/df_with_scoring_{symbol}.feather'
    df.to_feather(output_path)
    print(f"Saved merged data to {output_path}")
    
    # 打印数据统计
    print(f"\nData statistics for {symbol}:")
    print(f"Total rows: {len(df)}")
    print(f"Total columns: {len(df.columns)}")
    print(f"Score columns: {[col for col in df.columns if col.startswith('scoring_')]}")
    print(f"Target columns: {[col for col in df.columns if col.startswith('target_')]}")
    print(f"Required features: {[col for col in required_features if col in df.columns]}")

def main():
    # 读取AAPL的数据作为基准
    aapl_path = 'processed_data/df_with_scoring_AAPL.feather'
    aapl_scores = pd.read_feather(aapl_path)
    
    # 为其他股票生成评分
    symbols = ["MSFT", "INTC", "GOOG", "AMZN"]
    for symbol in symbols:
        generate_scores_for_stock(symbol, aapl_scores)

if __name__ == "__main__":
    main() 