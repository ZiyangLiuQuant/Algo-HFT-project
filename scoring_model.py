import pandas as pd
import numpy as np
import lightgbm as lgb
import joblib
from typing import List, Tuple, Optional
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

class ScoringModel:
    def __init__(self, model_type: str = "lightgbm", top_n_features: int = 50):
        """
        Initialize the scoring model.
        
        Args:
            model_type: The base model type to use, currently only supports 'lightgbm'
            top_n_features: Number of top features to select based on feature importance
        """
        self.model_type = model_type
        self.top_n_features = top_n_features
        self.model = None
        self.selected_features = None
        self.scaler = StandardScaler()
        
        # Initialize model with parameters suitable for high-frequency data
        if model_type == "lightgbm":
            self.model = lgb.LGBMRegressor(
                n_estimators=200,
                learning_rate=0.01,
                num_leaves=31,
                max_depth=6,
                min_child_samples=100,  # Increased to handle noise
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=1.0,  # Increased L1 regularization
                reg_lambda=1.0,  # Increased L2 regularization
                random_state=42,
                force_col_wise=True,
                min_child_weight=1e-3,  # Added to handle imbalanced targets
                min_split_gain=1e-3,    # Added to prevent overfitting
                verbose=-1              # Reduce verbosity
            )
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

    def prepare_data(self, df: pd.DataFrame, target_col: str):
        """
        Prepare data for model training with enhanced class imbalance handling
        
        Args:
            df: Input DataFrame
            target_col: Target column name
        
        Returns:
            tuple: Processed X_train, X_test, y_train, y_test
        """
        # Remove constant and near-constant features
        feature_cols = [col for col in df.columns if col != target_col]
        
        # 只选择数值类型的列
        numeric_cols = df[feature_cols].select_dtypes(include=['int64', 'float64']).columns
        feature_cols = list(numeric_cols)
        
        # 保留score_前缀的特征
        score_features = [col for col in feature_cols if col.startswith('score_')]
        
        # 对非score_前缀的特征进行方差过滤
        non_score_features = [col for col in feature_cols if not col.startswith('score_')]
        variance_threshold = 0.01
        variances = df[non_score_features].var()
        valid_non_score_features = variances[variances > variance_threshold].index.tolist()
        
        # 合并特征
        valid_features = valid_non_score_features + score_features
        
        X = df[valid_features]
        y = df[target_col]
        
        # 处理缺失值
        # 1. 删除目标变量中有缺失值的行
        valid_mask = ~y.isna()
        X = X[valid_mask]
        y = y[valid_mask]
        
        # 2. 对特征进行缺失值填充
        X = X.fillna(X.mean())
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=None  # 移除stratify参数，因为y是连续值
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # 将y转换为二分类问题
        y_train_binary = (y_train > 0).astype(int)
        y_test_binary = (y_test > 0).astype(int)
        
        # Apply SMOTE for minority class oversampling
        smote = SMOTE(random_state=42, sampling_strategy='auto')
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train_binary)
        
        # Calculate class weights
        class_weights = dict(zip(
            np.unique(y_train_binary),
            1 / np.bincount(y_train_binary) * len(y_train_binary) / 2
        ))
        
        self.class_weights = class_weights
        self.feature_names = valid_features
        
        return (
            X_train_resampled, X_test_scaled,
            y_train_resampled, y_test_binary,
            valid_features
        )

    def feature_selection(self, X: pd.DataFrame, y: pd.Series) -> List[str]:
        """
        Select top N features based on LightGBM feature importance with cross-validation.
        
        Args:
            X: Feature matrix
            y: Target vector
            
        Returns:
            List of selected feature names
        """
        # Train a model for feature selection with cross-validation
        model = lgb.LGBMRegressor(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.01,
            min_child_samples=100,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbose=-1
        )
        
        # Get feature importance scores
        importances = []
        n_splits = 5
        for i in range(n_splits):
            # Random subsample
            sample_idx = np.random.choice(len(X), size=len(X)//2, replace=False)
            X_sample = X.iloc[sample_idx]
            y_sample = y.iloc[sample_idx]
            
            model.fit(X_sample, y_sample)
            importances.append(pd.Series(model.feature_importances_, index=X.columns))
        
        # Average importance across folds
        mean_importance = pd.concat(importances, axis=1).mean(axis=1)
        top_features = mean_importance.sort_values(ascending=False).head(self.top_n_features).index.tolist()
        
        self.selected_features = top_features
        return top_features

    def train(self, X_train, y_train, X_val=None, y_val=None):
        """
        Train the LightGBM model.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            
        Returns:
            Trained model
        """
        # Create LightGBM datasets
        train_data = lgb.Dataset(X_train, label=y_train)
        valid_data = lgb.Dataset(X_val, label=y_val) if X_val is not None else None
        
        # Set parameters
        params = {
            'objective': 'binary',
            'metric': 'auc',
            'boosting_type': 'gbdt',
            'num_leaves': 50,
            'learning_rate': 0.003,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 20,
            'verbose': 0,
            'max_depth': 5,
            'is_unbalance': True
        }
        
        # Train model with early stopping callback
        valid_sets = [valid_data] if valid_data is not None else None
        self.model = lgb.train(
            params,
            train_data,
            num_boost_round=3000,
            valid_sets=valid_sets,
            callbacks=[
                lgb.early_stopping(stopping_rounds=100),
                lgb.log_evaluation(period=100)
            ]
        )
        
        return self.model

    def predict(self, X):
        """
        Make predictions
        """
        if self.model is None:
            raise ValueError("Model not trained yet")
        return self.model.predict(X)

    def save_model(self, filepath: str) -> None:
        """
        Save the trained model and selected features to disk.
        
        Args:
            filepath: Path to save the model
        """
        if self.model is None:
            raise ValueError("No model to save")
            
        # 保存模型、特征和scaler
        model_data = {
            'model': self.model,
            'selected_features': self.selected_features,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'class_weights': self.class_weights
        }
        joblib.dump(model_data, filepath)

    def load_model(self, filepath: str) -> None:
        """
        Load a trained model and selected features from disk.
        
        Args:
            filepath: Path to load the model from
        """
        model_data = joblib.load(filepath)
        self.model = model_data['model']
        self.selected_features = model_data['selected_features']
        self.scaler = model_data['scaler']
        self.feature_names = model_data['feature_names']
        self.class_weights = model_data['class_weights']

    def generate_scores(self, df: pd.DataFrame, target_col: str, symbol: str = None) -> pd.DataFrame:
        """
        Generate scores for the given target
        
        Args:
            df: Input DataFrame
            target_col: Name of the target column
            symbol: Stock symbol
            
        Returns:
            DataFrame with added score column
        """
        if self.model is None:
            raise ValueError("Model not trained yet. Please train the model first.")
            
        # 使用保存的特征名称
        if not hasattr(self, 'feature_names') or self.feature_names is None:
            raise ValueError("Feature names not found. Please train the model first.")
            
        # 检查所有需要的特征是否存在
        missing_features = [f for f in self.feature_names if f not in df.columns]
        if missing_features:
            # 如果缺少的特征是score_前缀的特征，我们需要先生成这些特征
            score_features = [f for f in missing_features if f.startswith('score_')]
            other_features = [f for f in missing_features if not f.startswith('score_')]
            
            if other_features:
                raise ValueError(f"Missing non-score features in input data: {other_features}")
                
            # 对于score_前缀的特征，我们需要先生成它们
            for score_feature in score_features:
                # 从特征名中提取目标名
                target_name = score_feature.replace('score_', '')
                # 加载对应的模型
                model_path = f"models/{symbol}_{target_name}_model.joblib" if symbol else f"models/{target_name}_model.joblib"
                temp_model = ScoringModel()
                temp_model.load_model(model_path)
                # 生成分数
                df = temp_model.generate_scores(df, target_name, symbol)
            
        # 准备特征
        X = df[self.feature_names].copy()
        
        # 使用保存的scaler进行特征缩放
        X_scaled = self.scaler.transform(X)
        
        # 生成预测
        scores = self.model.predict(X_scaled)
        
        # 添加分数到DataFrame，保留所有现有的列
        df = df.copy()  # 创建副本以避免修改原始DataFrame
        df[f'score_{target_col}'] = scores
        
        return df 