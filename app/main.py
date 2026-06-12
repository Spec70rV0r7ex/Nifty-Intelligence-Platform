import streamlit as st
import pandas as pd
import os
import sys
import matplotlib.pyplot as plt

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.abspath(os.path.join(CURRENT_DIR, "../src"))
sys.path.append(SRC_DIR)

from portfolio import build_price_matrix, generate_portfolio # type: ignore
from explainability import explain_latest_prediction # type: ignore

st.set_page_config(page_title = "Nifty50 Investment Intelligence", page_icon = None, layout = "wide")

PROCESSED_DIR = os.path.abspath(os.path.join(CURRENT_DIR, "../data/processed"))
MODELS_DIR = os.path.abspath(os.path.join(CURRENT_DIR, "../models"))

@st.cache_data
def load_processed_data():
    import glob
    files = glob.glob(os.path.join(PROCESSED_DIR, "*.csv"))
    data_dict = {}
    for f in files:
        ticker = os.path.basename(f).replace(".csv", "")
        data_dict[ticker] = pd.read_csv(f)
    return data_dict

data = load_processed_data()

st.title("Intelligent Investment Platform")
st.markdown("Transforming historical Nifty50 market data into actionable investment intelligence.")

tab_predict, tab_portfolio, tab_explain, tab_risk = st.tabs([
    "Market Forecast",
    "Portfolio Builder",
    "AI Explainability",
    "Risk & Anomalies"
])

with tab_predict:
    st.header("Stock Predictor Engine")
    st.markdown("Run your `src/predictor.py` script to generate the latest predictions, and we will visualize the top candidates here.")
    st.info("Ensure you have run the batch prediction script to populate this list.")

    #this table came from predictor.py file output
    st.table(pd.DataFrame({"Ticker": ["NTPC", "NESTLEIND", "ZEEL", "GAIL", "HDFCBANK"], "Probability UP (%)": [80.49, 77.45, 71.56, 69.37, 69.30]}))

with tab_portfolio:
    st.header("Automated Portfolio Construction")
    
    col1, col2 = st.columns([1, 2])
    with col1:
        risk_profile = st.selectbox("Select Investor Profile:", ["Conservative", "Balanced", "Aggressive"])
        capital = st.number_input("Investment Capital (₹):", min_value = 10000, value = 100000, step = 10000)
        generate_btn = st.button("Generate Portfolio")
        
    with col2:
        if generate_btn:
            with st.spinner("Optimizing using Modern Portfolio Theory..."):
                price_matrix = build_price_matrix(data)
                results = generate_portfolio(price_matrix, profile = risk_profile.lower(), total_capital = capital)
                
                st.success(f"Successfully generated {risk_profile} portfolio!")
                
                st.subheader("Performance Metrics")
                m1, m2, m3 = st.columns(3)
                m1.metric("Expected Annual Return", results["Metrics"]["Expected Annual Return"])
                m2.metric("Annual Volatility", results["Metrics"]["Annual Volatility"])
                m3.metric("Sharpe Ratio", results["Metrics"]["Sharpe Ratio"])
                
                st.subheader("Asset Allocation")
                st.write(results["Weights (%)"])

with tab_explain:
    st.header("Model Transparency (SHAP)")
    st.markdown("Select a stock to see exactly why the ML model made its decision.")
    
    selected_ticker = st.selectbox("Select Stock to Explain:", list(data.keys()))

    if st.button(f"Analyze {selected_ticker}"):
        with st.spinner("Breaking open the black box..."):

            explainer, shap_vals, latest_data, exp_df = explain_latest_prediction(
                selected_ticker, MODELS_DIR, PROCESSED_DIR
            )
            
            if exp_df is not None:
                st.subheader(f"What is driving the prediction for {selected_ticker}?")
                st.write("Green bars push the probability UP (Buy). Red bars push it DOWN (Sell).")

                plot_df = exp_df.sort_values(by='SHAP_Impact', ascending=True)

                colors = ['#ff4b4b' if x < 0 else '#00cc96' for x in plot_df['SHAP_Impact']]

                fig, ax = plt.subplots(figsize=(10, 6))
                ax.barh(plot_df['Feature'], plot_df['SHAP_Impact'], color=colors)

                ax.set_xlabel("SHAP Value (Impact on Probability)")
                ax.set_title(f"Feature Contributions for {selected_ticker}")
                ax.grid(axis='x', linestyle='--', alpha=0.7)

                st.pyplot(fig)

                st.markdown("### Raw Feature Data")
                st.dataframe(exp_df, use_container_width=True)
            else:
                st.error("Could not generate explanation. Ensure the model has been trained!")

with tab_risk:
    st.header("Market Risk & Anomaly Detection")
    st.markdown("Scanning the latest market data for highly unusual trading volume and extreme volatility spikes.")
    
    if st.button("Scan Market for Anomalies"):
        with st.spinner("Analyzing standard deviations across 50 assets..."):
            anomaly_list = []
      
            for ticker, df in data.items():
                if not df.empty:
                    latest_day = df.iloc[-1]

                    if 'Volume_Z_Score' in latest_day and latest_day['Volume_Z_Score'] > 2.5:
                        anomaly_list.append({
                            "Ticker": ticker,
                            "Anomaly Type": "Unusual Volume 📈",
                            "Metric": f"+{latest_day['Volume_Z_Score']:.2f} Z-Score",
                            "Date": latest_day['Date']
                        })

                    if 'Volatility_21d' in latest_day and latest_day['Volatility_21d'] > 0.50:
                        anomaly_list.append({
                            "Ticker": ticker,
                            "Anomaly Type": "Extreme Volatility 🌪️",
                            "Metric": f"{latest_day['Volatility_21d']*100:.2f}% Annualized",
                            "Date": latest_day['Date']
                        })
            
            if anomaly_list:
                st.warning(f"Detected {len(anomaly_list)} market anomalies in the latest data.")
                st.dataframe(pd.DataFrame(anomaly_list), use_container_width=True)
            else:
                st.success("Market conditions are stable. No statistical anomalies detected today.")
