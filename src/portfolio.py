import pandas as pd
from pypfopt.expected_returns import mean_historical_return
from pypfopt.risk_models import sample_cov
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt.discrete_allocation import DiscreteAllocation, get_latest_prices
import warnings

warnings.filterwarnings("ignore")
def build_price_matrix(processed_data_dict: dict) -> pd.DataFrame:
    price_df = pd.DataFrame()
    for ticker, df in processed_data_dict.items():
        df_temp = df.copy()
        df_temp.columns = df_temp.columns.str.strip()
        temp_series = df_temp[['Date', 'Close']].set_index('Date')
        temp_series.columns = [ticker]

        if price_df.empty:
            price_df = temp_series
        else:
            price_df = price_df.join(temp_series, how = 'outer')

    return price_df.ffill().dropna()

def generate_portfolio(price_matrix: pd.DataFrame, profile: str = 'balanced', total_capital: float = 100000):
    print(f"Optimizing for {profile.upper()} profile...")

    mu = mean_historical_return(price_matrix, frequency = 252)
    S = sample_cov(price_matrix, frequency = 252)
    ef = EfficientFrontier(mu, S, weight_bounds = (0.0, 0.15))

    if profile == 'conservative':
        weights = ef.min_volatility()
    elif profile == 'balanced':
        weights = ef.max_sharpe(risk_free_rate = 0.06)
    elif profile == 'aggressive':
        try:
            weights = ef.efficient_risk(target_volatility = 0.30)
        except ValueError:
            weights = ef.max_sharpe(risk_free_rate = 0.06)
    else:
        raise ValueError("Profile must be 'conservative', 'balanced', or 'aggressive'.")

    cleaned_weights = ef.clean_weights()
    expected_annual_return, annual_volatility, sharpe_ratio = ef.portfolio_performance(risk_free_rate = 0.06)
    latest_prices = get_latest_prices(price_matrix)
    da = DiscreteAllocation(cleaned_weights, latest_prices, total_portfolio_value = total_capital) # type: ignore

    try:
        allocation, leftover = da.lp_portfolio()
    except Exception as e:
        allocation = {k: v for k, v in cleaned_weights.items() if v > 0}
        leftover = total_capital

    results = {
        "Profile": profile.capitalize(),
        "Capital Allocated": total_capital - leftover, # type: ignore
        "Cash Remaining": leftover,
        "Allocations (Shares)": allocation,
        "Weights (%)": {k: round(v * 100, 2) for k, v in cleaned_weights.items() if v > 0},
        "Metrics": {
            "Expected Annual Return": f"{expected_annual_return * 100:.2f}%", # type: ignore
            "Annual Volatility": f"{annual_volatility * 100:.2f}%",
            "Sharpe Ratio": round(sharpe_ratio, 2) # type: ignore
        }
    }

    return results