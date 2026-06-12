import pandas as pd
import joblib
import shap
import os
import warnings

warnings.filterwarnings('ignore')

def explain_latest_prediction(ticker: str, models_dir: str, processed_dir: str):
    model_path = os.path.join(models_dir, f"{ticker}_xgb_model.joblib")
    data_path = os.path.join(processed_dir, f"{ticker}.csv")

    model = joblib.load(model_path)
    df = pd.read_csv(data_path)

    drop_cols = ['Date', 'Target_Return_5d', 'Target_Direction_5d']
    features = [col for col in df.columns if col not in drop_cols]
    latest_data = df.iloc[[-1]][features]
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(latest_data)

    explanation_df = pd.DataFrame({
        'Feature': features,
        'Value': latest_data.iloc[0].values,
        'SHAP_Impact': shap_values[0]
    })

    explanation_df['Abs_Impact'] = explanation_df['SHAP_Impact'].abs()
    explanation_df = explanation_df.sort_values(by='Abs_Impact', ascending=False).drop(columns=['Abs_Impact'])

    return explainer, shap_values, latest_data, explanation_df

if __name__ == "__main__":
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    MODELS_DIR = os.path.abspath(os.path.join(CURRENT_DIR, "../models"))
    PROCESSED_DIR = os.path.abspath(os.path.join(CURRENT_DIR, "../data/processed"))

    ticker_to_explain = "NTPC"

    explainer, shap_vals, latest_data, explanation_df = explain_latest_prediction(ticker_to_explain, MODELS_DIR, PROCESSED_DIR)

    if explanation_df is not None:
        print(f"\n SHAP Explanation for {ticker_to_explain}'s Latest Prediction:")
        print("Positive SHAP means it pushed the probability UP. Negative means DOWN.\n")
        print(explanation_df.head(10).to_string(index=False))