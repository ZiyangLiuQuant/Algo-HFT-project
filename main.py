import pandas as pd
import numpy as np
import logging
from tqdm import tqdm
from data_loader import DataLoader
from factors import FactorCalculator
from targets import TargetCalculator
from scoring_model import ScoringModel
import os
import warnings
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def is_symbol_processed(symbol: str) -> bool:
    """Check if a symbol has been fully processed"""
    # Check if all target scores exist
    processed_data_path = f"processed_data/{symbol}_processed.feather"
    if not os.path.exists(processed_data_path):
        return False
        
    df = pd.read_feather(processed_data_path)
    target_cols = [col for col in df.columns if col.startswith('target_')]
    
    for target_col in target_cols:
        scores_path = f"results/{symbol}_{target_col}_scores.csv"
        if not os.path.exists(scores_path):
            return False
            
    return True

def process_symbol(symbol: str):
    """Process single symbol data"""
    try:
        # Skip if already processed
        if is_symbol_processed(symbol):
            logger.info(f"{symbol} has already been processed, skipping...")
            return None
            
        # Try to load processed data in different formats
        processed_data_path_csv = f"processed_data/{symbol}_processed.csv"
        processed_data_path_feather = f"processed_data/{symbol}_processed.feather"
        
        if os.path.exists(processed_data_path_csv):
            logger.info(f"Loading processed data for {symbol} from CSV...")
            df = pd.read_csv(processed_data_path_csv)
        elif os.path.exists(processed_data_path_feather):
            logger.info(f"Loading processed data for {symbol} from Feather...")
            df = pd.read_feather(processed_data_path_feather)
        else:
            raise FileNotFoundError(f"No processed data found for {symbol}")
        
        # Get all target columns
        target_cols = [col for col in df.columns if col.startswith('target_')]
        logger.info(f"Found {len(target_cols)} targets to process: {target_cols}")
        
        # Generate scores for each target
        for target_col in tqdm(target_cols, desc=f"Processing targets for {symbol}", leave=True):
            logger.info(f"\nProcessing target: {target_col}")
            
            # Check if scores already exist
            scores_path = f"results/{symbol}_{target_col}_scores.csv"
            if os.path.exists(scores_path):
                logger.info(f"Scores already exist for {target_col}, skipping...")
                continue
            
            # Load or train model
            model_path = f"models/{symbol}_{target_col}_model.joblib"
            model = ScoringModel(model_type="lightgbm", top_n_features=50)
            
            try:
                if os.path.exists(model_path) and os.path.getsize(model_path) > 0:
                    logger.info(f"Loading model for {target_col}...")
                    model.load_model(model_path)
                else:
                    raise FileNotFoundError("Model file not found or empty")
            except (FileNotFoundError, EOFError) as e:
                logger.warning(f"Failed to load model for {target_col}: {str(e)}. Retraining...")
                
                # Prepare data for training
                logger.info("Preparing data for training...")
                X_train, X_test, y_train, y_test, valid_features = model.prepare_data(df, target_col)
                logger.info(f"Data prepared. Training set size: {len(X_train)}, Test set size: {len(X_test)}")
                
                # Train model with validation data
                logger.info("Training model...")
                model.train(X_train, y_train, X_test, y_test)
                
                # Save model
                os.makedirs("models", exist_ok=True)
                model.save_model(model_path)
                logger.info(f"Model saved to {model_path}")
            
            # Generate scores
            logger.info(f"Generating scores for {target_col}...")
            df = model.generate_scores(df, target_col, symbol)
            
            # Save results
            os.makedirs("results", exist_ok=True)
            df.to_csv(scores_path, index=False)
            logger.info(f"Scores saved to {scores_path}")
            
            # Print score statistics
            print(f"\nScore Statistics for {target_col}:")
            print(df[f'score_{target_col}'].describe())
        
        return df
        
    except Exception as e:
        logger.error(f"An error occurred while processing {symbol}: {str(e)}", exc_info=True)
        raise

def main():
    try:
        # Process all symbols
        symbols = ["AMZN", "GOOG", "MSFT", "INTC"]
        for symbol in tqdm(symbols, desc="Processing symbols", leave=True):
            logger.info(f"\nProcessing {symbol}...")
            df = process_symbol(symbol)
            if df is not None:
                logger.info(f"Completed processing {symbol}")
            
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main() 