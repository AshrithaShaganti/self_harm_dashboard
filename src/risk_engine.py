import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
import warnings
warnings.filterwarnings("ignore")

def calculate_national_index(df):
    """
    Calculates a daily national risk index from raw posts.
    Formula: (High Risk Count * 1.5 + Medium Risk Count * 1.0) / Total Posts * 100
    Adds some smoothing for graph stability.
    """
    df['date'] = pd.to_datetime(df['date'])
    daily = df.groupby(df['date'].dt.date).apply(
        lambda x: (
            ((x['risk_level'] == 'High Risk').sum() * 1.5 + 
             (x['risk_level'] == 'Medium Risk').sum() * 1.0) / len(x)
        ) * 100 if len(x) > 0 else 0
    ).reset_index(name='risk_index')
    
    daily['date'] = pd.to_datetime(daily['date'])
    
    # Smooth the index with a 7-day rolling average
    daily['smoothed_risk_index'] = daily['risk_index'].rolling(window=7, min_periods=1).mean()
    
    return daily

def forecast_risk(daily_df, steps=30):
    """
    Forecasts the risk index for the next `steps` days using ARIMA (1,1,1).
    """
    # Use the smoothed series to fit the model to avoid extreme noise
    series = daily_df['smoothed_risk_index'].values
    
    try:
        model = ARIMA(series, order=(1,1,1))
        fitted = model.fit()
        forecast = fitted.forecast(steps=steps)
    except Exception as e:
        # Fallback if ARIMA fails: simple moving average forward fill
        print("ARIMA failed, using fallback moving average", e)
        last_val = series[-1]
        forecast = np.array([last_val] * steps)
    
    last_date = daily_df['date'].max()
    future_dates = [last_date + pd.Timedelta(days=i) for i in range(1, steps+1)]
    
    return pd.DataFrame({
        'date': future_dates,
        'predicted_risk_index': forecast
    })

def generate_policy_recommendations(current_risk_index):
    """
    Simulates a government decision intelligence layer returning recommendations
    based on the current national risk thresholds.
    """
    if current_risk_index > 60:
        return [
            "CRITICAL: Deploy emergency mobile mental health response units in high-density urban areas.",
            "Alert national hotline centers for immediate surge capacity.",
            "Initiate direct-outreach ad campaigns on major social networking platforms."
        ]
    elif current_risk_index > 40:
        return [
            "WARNING: Escalate mental health awareness protocols in public schools.",
            "Provide community centers with additional counseling resources.",
            "Monitor regional hotspots for localized intervention."
        ]
    else:
        return [
            "STATUS NORMAL: Continue baseline monitoring of social channels.",
            "Maintain standard funding for community mental health programs.",
            "Review monthly aggregated trends for slow-moving shifts."
        ]
