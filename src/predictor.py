import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import accuracy_score
from sklearn.model_selection import TimeSeriesSplit
import os, glob, joblib, sys

def train_directional_model(df: pd.DataFrame, target_col: str = 'Target_Direction_5d'):
    drop_cols = ['Date', 'Target_Return_5d', 'Target_Direction_5d']
    features = [col for col in df.columns if col not in drop_cols]

    X, y = df[features], df[target_col]

    tscv = TimeSeriesSplit(n_splits = 5)
    print(f"Training Predictor Engine using {len(features)} features...")

    fold = 1
    models, accuracies = [], []

    for train_index, test_index in tscv.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        model = xgb.XGBClassifier(n_estimators = 100, learning_rate = 0.05, max_depth = 4, objective = 'binary:logistic', eval_metric = 'logloss', random_state = 42)

        model.fit(X_train, y_train)
        models.append(model)
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        accuracies.append(acc)
        print(f"Fold {fold} Accuracy: {acc:.4f}")
        fold += 1

    print(f"\nAverage Directional Accuracy: {np.mean(accuracies):.4f}")

    final_model = models[-1]
    importance = pd.DataFrame({'Feature': features, 'Importance': final_model.feature_importances_}).sort_values(by = 'Importance', ascending = False)

    print("\nTop 5 Predictive Features:")
    print(importance.head(5))

    return final_model

def generate_market_predictions(processed_dir: str, models_dir: str):
    os.makedirs(models_dir, exist_ok = True)
    csv_files = glob.glob(os.path.join(processed_dir, "*.csv"))
    print(f"Initializing Predictor Engine for {len(csv_files)} stocks...\n")

    predictions_list = []

    for file_path in csv_files:
        ticker = os.path.basename(file_path).replace(".csv", "")
        print(f"Training and saving model for {ticker}...")
        
        try:
            df = pd.read_csv(file_path)

            import os as sys_os
            old_stdout = sys.stdout
            sys.stdout = open(sys_os.devnull, 'w')
            model = train_directional_model(df)

            sys.stdout = old_stdout 

            model_filename = f"{ticker}_xgb_model.joblib"
            model_save_path = os.path.join(models_dir, model_filename)
            joblib.dump(model, model_save_path)

            latest_data = df.iloc[[-1]]
            drop_cols = ['Date', 'Target_Return_5d', 'Target_Direction_5d']
            features = [col for col in latest_data.columns if col not in drop_cols]

            X_latest = latest_data[features]
            prob_up = model.predict_proba(X_latest)[0][1]

            predictions_list.append({
                'Ticker': ticker,
                'Probability_UP_Next_5_Days (%)': round(prob_up * 100, 2)
            })

        except Exception as e:
            sys.stdout = old_stdout  # type: ignore
            print(f"Failed to train {ticker}: {e}")

    results_df = pd.DataFrame(predictions_list)
    results_df = results_df.sort_values(by = 'Probability_UP_Next_5_Days (%)', ascending = False).reset_index(drop = True)
    
    print("\nBatch Training Complete! Top 5 Buy Candidates:")
    print(results_df.head(5).to_string())
    
    return results_df

if __name__ == "__main__":
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    PROCESSED_DATA_PATH = os.path.abspath(os.path.join(CURRENT_DIR, "../data/processed"))
    MODELS_PATH = os.path.abspath(os.path.join(CURRENT_DIR, "../models1"))
    
    market_forecast = generate_market_predictions(PROCESSED_DATA_PATH, MODELS_PATH)