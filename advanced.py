import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit

# ==============================================================================
#  Configuration
# ==============================================================================
class Config:
    """
    A class to hold all configuration parameters for the script.
    This makes it easy to see and change key settings.
    """
    # --- File Paths ---
    TRAIN_PATH = "/kaggle/input/drw-crypto-market-prediction/train.parquet"
    TEST_PATH = "/kaggle/input/drw-crypto-market-prediction/test.parquet"
    SUBMISSION_PATH = "/kaggle/input/drw-crypto-market-prediction/sample_submission.csv"
    
    # --- Feature & Label Settings ---
    LABEL_COLUMN = "label"
    
    # --- Model Training Settings ---
    # We will use TimeSeriesSplit for robust cross-validation.
    N_SPLITS = 5
    
    # --- LightGBM Model Parameters ---
    # These parameters are chosen to be a robust starting point.
    # They favor regularization to prevent overfitting on the noisy financial data.
    LGB_PARAMS = {
        'objective': 'regression_l1',
        'metric': 'l1',
        'n_estimators': 2000,
        'learning_rate': 0.01,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'lambda_l1': 0.1,
        'lambda_l2': 0.1,
        'num_leaves': 31,
        'verbose': -1,
        'n_jobs': -1,
        'seed': 42,
        'boosting_type': 'gbdt',
    }

# ==============================================================================
#  Feature Engineering
# ==============================================================================
def feature_engineering(df):
    """
    Creates new features based on existing data to improve model performance.
    This function focuses on creating clear, intuitive features related to
    market microstructure.

    Args:
        df (pd.DataFrame): The input dataframe (train or test).

    Returns:
        pd.DataFrame: The dataframe with new features added.
    """
    # --- Basic Liquidity and Order Flow Features ---
    df['spread'] = df['ask_qty'] - df['bid_qty']
    df['total_volume'] = df['buy_qty'] + df['sell_qty']
    df['order_flow_imbalance'] = (df['buy_qty'] - df['sell_qty']) / (df['total_volume'] + 1e-9)
    
    # --- Interaction and Ratio Features ---
    df['log_volume'] = np.log1p(df['volume'])
    df['buy_ratio'] = df['buy_qty'] / (df['total_volume'] + 1e-9)
    df['spread_x_volume'] = df['spread'] * df['total_volume']
    
    # --- **NEW** Rolling Window Features ---
    # These features capture the recent trend and volatility of key metrics.
    # This is a powerful way to give the model a sense of momentum.
    rolling_windows = [5, 10, 20] # 5, 10, and 20-period moving windows
    features_to_roll = ['order_flow_imbalance', 'spread', 'log_volume']
    
    for window in rolling_windows:
        for feature in features_to_roll:
            # Calculate rolling mean (average)
            df[f'{feature}_roll_mean_{window}'] = df[feature].rolling(window=window).mean()
            # Calculate rolling standard deviation (volatility)
            df[f'{feature}_roll_std_{window}'] = df[feature].rolling(window=window).std()
            
    # Clean up any potential infinite or NaN values that may have been created
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(0, inplace=True)
    
    return df

# ==============================================================================
#  Main Execution Block
# ==============================================================================
def main():
    """
    Main function to run the full pipeline:
    1. Load data
    2. Apply feature engineering
    3. Train models using TimeSeriesSplit cross-validation
    4. Generate and average predictions
    5. Save the submission file
    """
    # --- 1. Load Data ---
    print("Loading data...")
    train_df = pd.read_parquet(Config.TRAIN_PATH)
    feature_cols = [col for col in train_df.columns if col != Config.LABEL_COLUMN]
    test_df = pd.read_parquet(Config.TEST_PATH, columns=feature_cols)
    submission_df = pd.read_csv(Config.SUBMISSION_PATH)
    
    # --- 2. Feature Engineering ---
    print("Applying feature engineering...")
    train_df = feature_engineering(train_df)
    test_df = feature_engineering(test_df)
    
    final_feature_cols = [col for col in train_df.columns if col != Config.LABEL_COLUMN]
    
    # --- 3. Model Training with Time-Series Cross-Validation ---
    print(f"Training model with {Config.N_SPLITS}-fold TimeSeriesSplit cross-validation...")
    
    X = train_df[final_feature_cols]
    y = train_df[Config.LABEL_COLUMN]
    
    tscv = TimeSeriesSplit(n_splits=Config.N_SPLITS)
    test_predictions = np.zeros(len(test_df))
    
    for fold, (train_index, valid_index) in enumerate(tscv.split(X)):
        print(f"--- Fold {fold + 1}/{Config.N_SPLITS} ---")
        X_train, X_valid = X.iloc[train_index], X.iloc[valid_index]
        y_train, y_valid = y.iloc[train_index], y.iloc[valid_index]

        model = lgb.LGBMRegressor(**Config.LGB_PARAMS)
        
        model.fit(
            X_train, y_train,
            eval_set=[(X_valid, y_valid)],
            eval_metric='l1',
            callbacks=[lgb.early_stopping(100, verbose=False)]
        )
        
        # Add this fold's predictions to the total. We will average them later.
        test_predictions += model.predict(test_df[final_feature_cols]) / Config.N_SPLITS

    # --- 4. Generate Predictions ---
    print("\nGenerating final predictions on the test set...")
    # The final prediction is the average of predictions from all fold models.
    
    # --- 5. Save Submission ---
    print("Saving submission file...")
    submission_df['prediction'] = test_predictions
    submission_df.to_csv("submission.csv", index=False)
    
    print("\nScript finished successfully!")
    print(f"Submission file created: submission.csv")
    print(submission_df.head())

if __name__ == "__main__":
    main()
