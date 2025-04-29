import pandas as pd
import numpy as np
import os
from tqdm import tqdm
import sys
import gc
import logging
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('merge_data.log'),
        logging.StreamHandler()
    ]
)

def load_data_file(file_path, chunksize=100000):
    """Load data file efficiently based on file extension"""
    if file_path.endswith('.feather'):
        return pd.read_feather(file_path)
    elif file_path.endswith('.csv'):
        # Read CSV in chunks to handle large files
        chunks = []
        for chunk in pd.read_csv(file_path, chunksize=chunksize):
            chunks.append(chunk)
            gc.collect()  # Force garbage collection after each chunk
        return pd.concat(chunks, ignore_index=True)
    else:
        raise ValueError(f"Unsupported file format: {file_path}")

def process_chunk(chunk, score_df, score_col):
    """Process a single chunk of data"""
    # Convert timestamp to datetime if needed
    if 'timestamp' in chunk.columns:
        chunk['timestamp'] = pd.to_datetime(chunk['timestamp'])
    elif 'time' in chunk.columns:
        chunk['timestamp'] = pd.to_datetime(chunk['time'])
    
    # Merge with score data
    merged = pd.merge(chunk, score_df[['timestamp', score_col]], 
                     on='timestamp', how='left')
    return merged

def merge_stock_data(symbol):
    try:
        logging.info(f"\nProcessing {symbol}...")
        
        # Try to load processed data
        processed_data_path = f'processed_data/{symbol}_processed.feather'
        if not os.path.exists(processed_data_path):
            processed_data_path = f'processed_data/{symbol}_processed.csv'
            if not os.path.exists(processed_data_path):
                raise FileNotFoundError(f"No processed data found for {symbol}")
        
        logging.info(f"Loading processed data from {processed_data_path}")
        
        # Load data in chunks
        chunksize = 100000
        if processed_data_path.endswith('.feather'):
            df = pd.read_feather(processed_data_path)
        else:
            df = pd.read_csv(processed_data_path, chunksize=chunksize)
        
        # Get list of score files
        score_files = [f for f in os.listdir('results') if f.startswith(f'{symbol}_target_') and f.endswith('_scores.csv')]
        logging.info(f"Found {len(score_files)} score files")
        
        # Initialize merged data
        merged_data = []
        
        # Process each score file
        for score_file in tqdm(score_files, desc=f"Merging scores for {symbol}"):
            try:
                logging.info(f"Processing {score_file}")
                score_df = pd.read_csv(f'results/{score_file}')
                
                # Get the target name from the filename
                target_name = score_file.replace(f'{symbol}_target_', '').replace('_scores.csv', '')
                
                # Find the score column
                score_cols = [col for col in score_df.columns if col.startswith('score_')]
                if not score_cols:
                    logging.warning(f"No score columns found in {score_file}")
                    continue
                    
                score_col = score_cols[-1]
                
                # Convert score_df timestamp
                if 'timestamp' in score_df.columns:
                    score_df['timestamp'] = pd.to_datetime(score_df['timestamp'])
                elif 'time' in score_df.columns:
                    score_df['timestamp'] = pd.to_datetime(score_df['time'])
                
                # Process data in chunks if it's a generator
                if isinstance(df, pd.io.parsers.TextFileReader):
                    for chunk in df:
                        processed_chunk = process_chunk(chunk, score_df, score_col)
                        merged_data.append(processed_chunk)
                        gc.collect()
                else:
                    processed_chunk = process_chunk(df, score_df, score_col)
                    merged_data.append(processed_chunk)
                
                logging.info(f"Successfully merged {score_col}")
                
            except Exception as e:
                logging.error(f"Error processing {score_file}: {str(e)}")
                continue
        
        # Combine all processed chunks
        if merged_data:
            final_df = pd.concat(merged_data, ignore_index=True)
        else:
            final_df = df.copy()
        
        # Add missing columns
        scoring_columns = ['scoring_ask_margin', 'scoring_bid_margin', 
                         'scoring_buy_twap_diff', 'scoring_sell_twap_diff',
                         'scoring_spread_drop']
        target_columns = ['target_twap_diff', 'target_is_better_than_twap',
                         'target_buy_better_than_twap', 'target_sell_better_than_twap',
                         'target_best_future_ask', 'target_best_future_bid',
                         'target_ticks_until_best_ask', 'target_ticks_until_best_bid']
        
        for col in scoring_columns + target_columns:
            if col not in final_df.columns:
                final_df[col] = np.nan
        
        # Save merged data
        output_path = f'processed_data/df_with_scoring_{symbol}.feather'
        logging.info(f"Saving merged data to {output_path}")
        final_df.to_feather(output_path)
        
        # Clean up memory
        del final_df
        gc.collect()
        
    except Exception as e:
        logging.error(f"Error processing {symbol}: {str(e)}")

def main():
    try:
        symbols = ["MSFT", "INTC", "GOOG", "AMZN", "AAPL"]
        for symbol in symbols:
            merge_stock_data(symbol)
            gc.collect()  # Force garbage collection after each symbol
    except Exception as e:
        logging.error(f"Error in main: {e}")

if __name__ == "__main__":
    main() 