import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.feature_selection import mutual_info_regression
from sklearn.decomposition import PCA

# ==============================================================================
#  Configuration
# ==============================================================================
class Config:
    """
    Holds all configuration parameters for the script.
    """
    # --- File Paths ---
    TRAIN_PATH = "/kaggle/input/drw-crypto-market-prediction/train.parquet"
    TEST_PATH = "/kaggle/input/drw-crypto-market-prediction/test.parquet"
    SUBMISSION_PATH = "/kaggle/input/drw-crypto-market-prediction/sample_submission.csv"
    
    # --- Feature & Label Settings ---
    LABEL_COLUMN = "label"
    
    # --- MI and PCA Settings ---
    N_TOP_MI_FEATURES = 50  # Number of top features to select using Mutual Information
    PCA_CORR_THRESHOLD = 0.8 # Correlation threshold to group features for PCA
    N_PCA_COMPONENTS = 2     # Number of principal components to create for each group
    
    # --- Model Training Settings ---
    N_SPLITS = 5
    
    # --- LightGBM Model Parameters ---
    LGB_PARAMS = {
        'objective': 'regression_l1', 'metric': 'l1', 'n_estimators': 2000,
        'learning_rate': 0.01, 'feature_fraction': 0.7, 'bagging_fraction': 0.8,
        'lambda_l1': 0.2, 'lambda_l2': 0.2, 'num_leaves': 40, 'verbose': -1,
        'n_jobs': -1, 'seed': 42, 'boosting_type': 'gbdt',
    }

# ==============================================================================
#  Feature Engineering & Discovery
# ==============================================================================
def create_pca_features(train_df, test_df, top_mi_features):
    """
    Identifies correlated clusters in the top features and applies PCA.
    """
    print(f"Finding correlated clusters within top {len(top_mi_features)} MI features...")
    
    # Calculate correlation matrix for the high-potential features
    corr_matrix = train_df[top_mi_features].corr().abs()
    
    # Identify clusters of highly correlated features
    clusters = []
    remaining_features = list(top_mi_features)
    while remaining_features:
        feature = remaining_features.pop(0)
        # Find all features highly correlated with the current one
        correlated_group = list(corr_matrix[feature][corr_matrix[feature] > Config.PCA_CORR_THRESHOLD].index)
        
        if len(correlated_group) > 1:
            clusters.append(correlated_group)
            # Remove the clustered features so we don't process them again
            remaining_features = [f for f in remaining_features if f not in correlated_group]

    print(f"Found {len(clusters)} clusters to apply PCA.")

    # Apply PCA to each cluster
    for i, cluster in enumerate(clusters):
        pca = PCA(n_components=Config.N_PCA_COMPONENTS, random_state=Config.LGB_PARAMS['seed'])
        
        # Fit PCA on training data and transform both train and test
        train_pca = pca.fit_transform(train_df[cluster])
        test_pca = pca.transform(test_df[cluster])
        
        # Add new PCA components as features
        for j in range(Config.N_PCA_COMPONENTS):
            train_df[f'pca_cluster_{i}_comp_{j}'] = train_pca[:, j]
            test_df[f'pca_cluster_{i}_comp_{j}'] = test_pca[:, j]
            
    return train_df, test_df

def feature_engineering(df):
    """
    Creates market microstructure and rolling window features.
    """
    # --- Microstructure Features ---
    df['total_depth'] = df['ask_qty'] + df['bid_qty']
    df['total_volume'] = df['buy_qty'] + df['sell_qty']
    df['net_order_flow'] = df['buy_qty'] - df['sell_qty']
    df['order_flow_imbalance'] = df['net_order_flow'] / (df['total_volume'] + 1e-9)
    df['depth_imbalance'] = (df['ask_qty'] - df['bid_qty']) / (df['total_depth'] + 1e-9)
    df['spread'] = df['ask_qty'] - df['bid_qty']
    df['log_volume'] = np.log1p(df['volume'])
    
    # --- Rolling Window Features ---
    rolling_windows = [5, 10, 20]
    features_to_roll = [
        'order_flow_imbalance', 'depth_imbalance', 'spread', 'log_volume'
    ]
    
    for window in rolling_windows:
        for feature in features_to_roll:
            df[f'{feature}_roll_mean_{window}'] = df[feature].rolling(window=window).mean()
            df[f'{feature}_roll_std_{window}'] = df[feature].rolling(window=window).std()
            
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(0, inplace=True)
    
    return df

# ==============================================================================
#  Main Execution Block
# ==============================================================================
def main():
    # --- 1. Load Data ---
    print("Loading data...")
    train_df = pd.read_parquet(Config.TRAIN_PATH)
    test_df = pd.read_parquet(Config.TEST_PATH)
    submission_df = pd.read_csv(Config.SUBMISSION_PATH)
    
    # --- 2. Signal Hunting with Mutual Information ---
    print("Calculating Mutual Information scores to find top X features...")
    x_features = [col for col in train_df.columns if col.startswith('X')]
    
    # Fill NaNs before MI calculation
    mi_scores = mutual_info_regression(train_df[x_features].fillna(0), train_df[Config.LABEL_COLUMN])
    top_mi_features = pd.Series(mi_scores, index=x_features).sort_values(ascending=False).head(Config.N_TOP_MI_FEATURES).index.tolist()
    
    # --- 3. Create PCA Features ---
    train_df, test_df = create_pca_features(train_df, test_df, top_mi_features)

    # --- 4. Apply Standard Feature Engineering ---
    print("Applying standard feature engineering...")
    train_df = feature_engineering(train_df)
    test_df = feature_engineering(test_df)
    
    # --- 5. Model Training ---
    final_feature_cols = [col for col in train_df.columns if col != Config.LABEL_COLUMN]
    # Ensure test set has the same columns as train set, filling missing ones with 0
    test_df = test_df.reindex(columns=final_feature_cols, fill_value=0)

    print(f"Training model with {Config.N_SPLITS}-fold TimeSeriesSplit...")
    X = train_df[final_feature_cols]
    y = train_df[Config.LABEL_COLUMN]
    
    tscv = TimeSeriesSplit(n_splits=Config.N_SPLITS)
    test_predictions = np.zeros(len(test_df), dtype='float32')
    
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
        test_predictions += model.predict(test_df[final_feature_cols]) / Config.N_SPLITS

    # --- 6. Save Submission ---
    print("\nSaving submission file...")
    submission_df['prediction'] = test_predictions
    submission_df.to_csv("submission.csv", index=False)
    
    print("\nScript finished successfully!")
    print(submission_df.head())

if __name__ == "__main__":
    main()
