# Standard library imports
import os
import gc
import logging
import warnings
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from concurrent.futures import ProcessPoolExecutor

# Third-party imports
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
from tqdm import tqdm

# Local imports
from data_loader import DataLoader
from factors import FactorCalculator
from data_preprocessor import DataPreprocessor
from targets import TargetCalculator

# Configure logging
logging.basicConfig(
    level=logging.WARNING,  # 只显示警告和错误
    format='%(message)s',  # 简化日志格式
    handlers=[
        logging.FileHandler(f'logs/error_log_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 设置tqdm的日志级别
logging.getLogger('tqdm').setLevel(logging.WARNING)

# Configure pandas display options
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
pd.set_option('display.float_format', '{:.4f}'.format)

# Configure numpy display options
np.set_printoptions(precision=4, suppress=True)

# Configure matplotlib backend
plt.switch_backend('Agg')

# Ignore warnings
warnings.filterwarnings('ignore')

class FactorAnalyzer:
    def __init__(self, 
                 min_ic: float = 0.02,
                 min_auc: float = 0.55,
                 min_monotonicity: float = 0.6,
                 window_size: str = '5s',
                 n_jobs: int = -1):
        """
        初始化因子分析器
        
        Args:
            min_ic: 最小IC阈值
            min_auc: 最小AUC阈值
            min_monotonicity: 最小单调性阈值
            window_size: 滚动窗口大小
            n_jobs: 并行计算使用的CPU核心数，-1表示使用所有核心
        """
        self.min_ic = min_ic
        self.min_auc = min_auc
        self.min_monotonicity = min_monotonicity
        self.window_size = window_size
        self.n_jobs = n_jobs if n_jobs > 0 else None
        
    def _validate_data(self, df: pd.DataFrame) -> bool:
        """
        验证数据完整性
        
        Args:
            df: 输入数据框
            
        Returns:
            bool: 数据是否有效
        """
        try:
            # 检查必需列
            required_columns = ['mid_price', 'ask_price', 'bid_price', 'spread']
            if not all(col in df.columns for col in required_columns):
                logger.error(f"Missing required columns. Required: {required_columns}")
                return False
                
            # 检查时间索引
            if not isinstance(df.index, pd.DatetimeIndex):
                logger.error("DataFrame must have datetime index")
                return False
                
            # 排除时间相关的列
            time_related_columns = ['minute', 'date', 'time', 'hour', 'date_group']
            
            # 确保所有非时间列都是数值类型
            for col in df.columns:
                if col not in time_related_columns and not pd.api.types.is_numeric_dtype(df[col]):
                    logger.warning(f"Column {col} is not numeric, attempting to convert...")
                    try:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                    except Exception as e:
                        logger.error(f"Failed to convert column {col} to numeric: {str(e)}")
                        return False
                
            return True
            
        except Exception as e:
            logger.error(f"Error in data validation: {str(e)}")
            return False
        
    def _calculate_ic(self, factor: pd.Series, target: pd.Series) -> pd.Series:
        """
        计算信息系数(IC)
        
        Args:
            factor: 因子值
            target: 目标变量
            
        Returns:
            pd.Series: 滚动IC序列
        """
        try:
            # 确保输入是数值类型
            if not pd.api.types.is_numeric_dtype(factor) or not pd.api.types.is_numeric_dtype(target):
                logger.error("Factor and target must be numeric types")
                return pd.Series(index=factor.index, dtype=float)
                
            # 使用历史数据计算IC
            factor_shifted = factor.shift(1)
            valid_mask = ~(factor_shifted.isna() | target.isna())
            
            if valid_mask.sum() < 10:  # 至少需要10个有效样本
                logger.warning(f"有效样本数量不足: {valid_mask.sum()} < 10")
                return pd.Series(index=factor.index, dtype=float)
                
            factor = factor_shifted[valid_mask]
            target = target[valid_mask]
            
            # 计算滚动相关性
            ic = pd.Series(index=factor.index, dtype=float)
            
            # 使用固定窗口大小，而不是基于时间
            window_size = 100  # 使用100个样本作为窗口大小
            
            for i in range(window_size, len(factor)):
                window_start = i - window_size
                window_factor = factor.iloc[window_start:i]
                window_target = target.iloc[window_start:i]
                
                # 检查窗口内数据的有效性
                valid_window = ~(window_factor.isna() | window_target.isna())
                if valid_window.sum() < 10:  # 窗口内至少需要10个有效样本
                    continue
                    
                window_factor = window_factor[valid_window]
                window_target = window_target[valid_window]
                
                # 检查数据是否都是相同的值
                if window_factor.nunique() < 2 or window_target.nunique() < 2:
                    continue
                    
                try:
                    correlation = window_factor.corr(window_target)
                    if pd.notna(correlation):  # 确保相关性是有效数值
                        ic.iloc[i] = correlation
                except Exception as e:
                    logger.warning(f"计算相关性时出错: {str(e)}")
                    continue
                    
            # 如果所有IC都是NaN，返回一个全0的序列
            if ic.isna().all():
                logger.warning("所有IC值都是NaN，返回全0序列")
                return pd.Series(0, index=factor.index)
                
            return ic
            
        except Exception as e:
            logger.error(f"计算IC时出错: {str(e)}")
            return pd.Series(index=factor.index, dtype=float)
        
    def _calculate_auc(self, factor: pd.Series, target: pd.Series) -> float:
        """
        计算AUC，使用sklearn的roc_auc_score进行优化
        
        Args:
            factor: 因子值
            target: 目标变量
            
        Returns:
            float: AUC值
        """
        try:
            # 使用历史数据计算AUC
            factor_shifted = factor.shift(1)
            valid_mask = ~(factor_shifted.isna() | target.isna())
            if valid_mask.sum() < 10:  # 至少需要10个有效样本
                return 0.0
                
            factor_shifted = factor_shifted[valid_mask]
            target = target[valid_mask]
            
            # 使用sklearn的roc_auc_score进行优化
            return roc_auc_score((target > target.median()).astype(int), factor_shifted)
            
        except Exception as e:
            logger.error(f"Error calculating AUC: {str(e)}")
            return 0.0
            
    def _check_monotonicity(self, factor: pd.Series, target: pd.Series) -> float:
        """
        检查单调性，使用Spearman秩相关进行优化
        
        Args:
            factor: 因子值
            target: 目标变量
            
        Returns:
            float: 单调性得分
        """
        try:
            # 使用历史数据检查单调性
            factor_shifted = factor.shift(1)
            valid_mask = ~(factor_shifted.isna() | target.isna())
            if valid_mask.sum() < 10:
                return 0.0
                
            factor_shifted = factor_shifted[valid_mask]
            target = target[valid_mask]
            
            # 使用Spearman秩相关计算单调性
            monotonicity, _ = stats.spearmanr(factor_shifted, target)
            return abs(monotonicity)  # 取绝对值，因为方向不重要
            
        except Exception as e:
            logger.error(f"Error checking monotonicity: {str(e)}")
            return 0.0
            
    def _run_single_factor_test(self, 
                              factor: pd.Series, 
                              target: pd.Series) -> Optional[Dict]:
        """
        运行单个因子测试
        
        Args:
            factor: 因子数据
            target: 目标变量数据
            
        Returns:
            Optional[Dict]: 测试结果，如果测试失败则返回None
        """
        try:
            logger.info(f"开始测试因子 {factor.name} 和目标 {target.name}")
            
            # 确保数据是数值类型
            factor = pd.to_numeric(factor, errors='coerce')
            target = pd.to_numeric(target, errors='coerce')
            
            # 移除无效值
            valid_mask = ~(factor.isna() | target.isna())
            valid_count = valid_mask.sum()
            logger.info(f"有效样本数量: {valid_count}")
            
            if valid_count < 10:  # 至少需要10个有效样本
                logger.warning(f"有效样本数量不足: {valid_count} < 10")
                return None
                
            factor = factor[valid_mask]
            target = target[valid_mask]
            
            # 计算IC
            ic = self._calculate_ic(factor, target)
            if ic is None or ic.empty:
                logger.warning("IC计算失败")
                return None
                
            mean_ic = ic.mean()
            if pd.isna(mean_ic):
                logger.warning("IC均值为NaN")
                return None
            logger.info(f"IC均值: {mean_ic:.4f}")
            
            # 计算AUC
            auc = self._calculate_auc(factor, target)
            if pd.isna(auc):
                logger.warning("AUC计算失败")
                return None
            logger.info(f"AUC: {auc:.4f}")
            
            # 检查单调性
            monotonicity = self._check_monotonicity(factor, target)
            if pd.isna(monotonicity):
                logger.warning("单调性检查失败")
                return None
            logger.info(f"单调性: {monotonicity:.4f}")
            
            # 返回所有测试结果，不进行阈值筛选
            result = {
                'factor': factor.name,
                'target': target.name,
                'mean_ic': float(mean_ic),
                'auc': float(auc),
                'monotonicity': float(monotonicity),
                'ic_series': ic.astype(float).to_dict()
            }
            logger.info(f"测试完成: {result}")
            return result
            
        except Exception as e:
            logger.error(f"因子测试出错: {str(e)}")
            return None
            
    def run_single_factor_tests(self, 
                              df: pd.DataFrame, 
                              factor_columns: List[str], 
                              target_columns: List[str]) -> Dict:
        """
        运行所有因子测试
        
        Args:
            df: 数据框
            factor_columns: 因子列名列表
            target_columns: 目标变量列名列表
            
        Returns:
            Dict: 测试结果
        """
        try:
            if not self._validate_data(df):
                raise ValueError("Invalid data format")
                
            if not factor_columns or not target_columns:
                logger.warning("No factors or targets to test")
                return {}
                
            # 数据抽样：随机抽取50%的tick（增加抽样比例）
            sample_size = int(len(df) * 0.5)
            df = df.sample(n=sample_size, random_state=42)
            logger.info(f"抽样 {sample_size} 个tick进行快速验证")
                
            # 排除时间相关的列
            time_related_columns = ['date', 'time', 'hour', 'date_group', 'minute']
            
            # 确保因子和目标列都是数值类型，且不是时间相关列
            valid_factor_columns = []
            for col in factor_columns:
                if col not in time_related_columns:
                    try:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                        if pd.api.types.is_numeric_dtype(df[col]):
                            valid_factor_columns.append(col)
                    except Exception as e:
                        logger.warning(f"因子列 {col} 转换为数值类型失败: {str(e)}")
                        
            valid_target_columns = []
            for col in target_columns:
                if col not in time_related_columns:
                    try:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                        if pd.api.types.is_numeric_dtype(df[col]):
                            valid_target_columns.append(col)
                    except Exception as e:
                        logger.warning(f"目标列 {col} 转换为数值类型失败: {str(e)}")
                    
            if not valid_factor_columns or not valid_target_columns:
                logger.error("没有有效的数值型因子或目标变量")
                return {}
                
            # 限制测试的因子数量
            max_factors = 50  # 最多测试50个因子
            if len(valid_factor_columns) > max_factors:
                logger.info(f"限制因子测试数量为前 {max_factors} 个因子")
                # 选择方差最大的因子
                factor_variances = df[valid_factor_columns].var()
                valid_factor_columns = factor_variances.nlargest(max_factors).index.tolist()
            
            # 生成所有因子-目标对
            test_combinations = []
            for factor in valid_factor_columns:
                for target in valid_target_columns:
                    test_combinations.append((df[factor], df[target]))
            
            logger.info(f"在 {sample_size} 个tick上运行 {len(test_combinations)} 个因子测试")
            
            # 使用joblib进行并行处理
            results = Parallel(n_jobs=4, batch_size=16)(
                delayed(self._run_single_factor_test)(factor, target)
                for factor, target in tqdm(test_combinations, 
                                         desc="运行因子测试",
                                         mininterval=5.0)
            )
            
            # 过滤掉None结果并转换为字典
            valid_results = [r for r in results if r is not None]
            logger.info(f"成功完成 {len(valid_results)} 个因子测试")
            
            results = {f"{r['factor']}_{r['target']}": r 
                      for r in valid_results}
                        
            return results
            
        except Exception as e:
            logger.error(f"运行因子测试时出错: {str(e)}")
            return {}
        
    def analyze_factor_importance(self, results: Dict) -> pd.DataFrame:
        """
        分析因子重要性，使用多层次的筛选策略：
        1. 首先尝试使用多个筛选条件（IC > 0.01, AUC > 0.52, monotonicity > 0.5）
        2. 如果结果为空，则只使用 IC > 0.01 作为筛选条件
        3. 如果仍然为空，则选择所有因子中 IC 值最大的前5个
        
        Args:
            results: 因子测试结果
            
        Returns:
            pd.DataFrame: 因子重要性分析结果
        """
        try:
            if not results:
                logger.warning("No results to analyze")
                return pd.DataFrame()
                
            importance_data = []
            
            for key, result in results.items():
                importance_data.append({
                    'factor': result['factor'],
                    'target': result['target'],
                    'mean_ic': result['mean_ic'],
                    'auc': result['auc'],
                    'monotonicity': result['monotonicity'],
                    'importance_score': (
                        abs(result['mean_ic']) * 0.4 + 
                        result['auc'] * 0.3 + 
                        result['monotonicity'] * 0.3
                    )
                })
                
            importance_df = pd.DataFrame(importance_data)
            if importance_df.empty:
                return importance_df
                
            # 第一层筛选：使用多个条件
            filtered_df = importance_df[
                (abs(importance_df['mean_ic']) > 0.01) |
                (importance_df['auc'] > 0.52) |
                (importance_df['monotonicity'] > 0.5)
            ]
            
            if not filtered_df.empty:
                logger.info("使用多个条件筛选出重要因子")
                return filtered_df.sort_values('importance_score', ascending=False)
                
            # 第二层筛选：只使用IC
            filtered_df = importance_df[abs(importance_df['mean_ic']) > 0.01]
            if not filtered_df.empty:
                logger.info("使用IC > 0.01筛选出重要因子")
                return filtered_df.sort_values('importance_score', ascending=False)
                
            # 第三层筛选：选择IC最大的前5个
            logger.info("使用IC最大的前5个因子作为重要因子")
            return importance_df.nlargest(5, 'mean_ic')
            
        except Exception as e:
            logger.error(f"Error analyzing factor importance: {str(e)}")
            return pd.DataFrame() 