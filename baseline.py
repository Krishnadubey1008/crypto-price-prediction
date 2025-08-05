import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split

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
    # We will use all available features from the dataset.
    LABEL_COLUMN = "label"
    
    # --- Model Training Settings ---
    # We train on the most recent 80% of the data, as market dynamics change over time.
    # This is a simple and effective way to handle the time-series nature of the data.
    TRAIN_PCT = 0.80
    
    # --- LightGBM Model Parameters ---
    # These parameters are chosen to be a robust starting point.
    # They favor regularization to prevent overfitting on the noisy financial data.
    LGB_PARAMS = {
        'objective': 'regression_l1', # MAE is often more robust to outliers than MSE
        'metric': 'l1',
        'n_estimators': 2000,         # High number of estimators, will use early stopping
        'learning_rate': 0.01,        # Low learning rate
        'feature_fraction': 0.8,      # Use a subset of features for each tree
        'bagging_fraction': 0.8,      # Use a subset of data for each tree
        'lambda_l1': 0.1,             # L1 regularization
        'lambda_l2': 0.1,             # L2 regularization
        'num_leaves': 31,             # Controls model complexity
        'verbose': -1,                # Suppress verbose output
        'n_jobs': -1,                 # Use all available CPU cores
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
    
    # Calculate the bid-ask spread proxy (difference between best bid and ask quantities)
    df['spread'] = df['ask_qty'] - df['bid_qty']
    
    # Total volume of buy and sell orders
    df['total_volume'] = df['buy_qty'] + df['sell_qty']
    
    # Order flow imbalance: (buy volume - sell volume) / total volume
    # This indicates the net direction of market pressure.
    df['order_flow_imbalance'] = (df['buy_qty'] - df['sell_qty']) / (df['total_volume'] + 1e-9)
    
    # --- Interaction and Ratio Features ---
    
    # Log of volume to handle skewed distribution
    df['log_volume'] = np.log1p(df['volume'])
    
    # Ratio of buy volume to total volume
    df['buy_ratio'] = df['buy_qty'] / (df['total_volume'] + 1e-9)
    
    # Interaction between spread and total volume
    df['spread_x_volume'] = df['spread'] * df['total_volume']
    
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
    3. Train the model
    4. Generate predictions
    5. Save the submission file
    """
    # --- 1. Load Data ---
    print("Loading data...")
    # Load the training data to infer the full set of feature columns
    train_df = pd.read_parquet(Config.TRAIN_PATH)
    
    # Identify feature columns from the loaded training data, excluding the label
    feature_cols = [col for col in train_df.columns if col != Config.LABEL_COLUMN]
    
    # Load test data using the identified feature columns
    test_df = pd.read_parquet(Config.TEST_PATH, columns=feature_cols)
    submission_df = pd.read_csv(Config.SUBMISSION_PATH)
    
    # --- 2. Feature Engineering ---
    print("Applying feature engineering...")
    train_df = feature_engineering(train_df)
    test_df = feature_engineering(test_df)
    
    # Update the list of features to include the new ones
    final_feature_cols = [col for col in train_df.columns if col != Config.LABEL_COLUMN]
    
    # --- 3. Model Training ---
    print("Training the model...")
    
    # To respect the time-series nature, we train on the most recent data.
    # First, we determine the cutoff point.
    cutoff_index = int(len(train_df) * (1 - Config.TRAIN_PCT))
    train_recent_df = train_df.iloc[cutoff_index:].reset_index(drop=True)
    
    X = train_recent_df[final_feature_cols]
    y = train_recent_df[Config.LABEL_COLUMN]
    
    # We use a validation set to find the optimal number of trees (early stopping)
    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=0.1, random_state=Config.LGB_PARAMS['seed'], shuffle=False # No shuffle for time-series
    )

    model = lgb.LGBMRegressor(**Config.LGB_PARAMS)
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_valid, y_valid)],
        eval_metric='l1',
        callbacks=[lgb.early_stopping(100, verbose=False)] # Stop if validation score doesn't improve for 100 rounds
    )
    
    # --- 4. Generate Predictions ---
    print("Generating predictions on the test set...")
    predictions = model.predict(test_df[final_feature_cols])
    
    # --- 5. Save Submission ---
    print("Saving submission file...")
    submission_df['prediction'] = predictions
    submission_df.to_csv("submission.csv", index=False)
    
    print("\nScript finished successfully!")
    print(f"Submission file created: submission.csv")
    print(submission_df.head())

if __name__ == "__main__":
    main()
