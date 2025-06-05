import requests
import random

BASE_URL = "http://localhost:8100"

# def test_bert_inference():
#     payload = {
#         "subject": "Компания Kaspi показала рост выручки на 25% в первом квартале"
#     }
#     response = requests.post(f"{BASE_URL}/bert", json=payload)
#     print("=== BERT Inference ===")
#     print("Status Code:", response.status_code)
#     print("Response JSON:", response.json())

def test_lstm_month_inference():
    payload = {
        "ticker": "ATEC",
        "sequence": [
            [1.0883, 0.16428, 0.74744, 287300, 2.5983e9],
            [2.2503, 0.55521, 0.19445, 383120, 4.4810e9],
            [0.72487, 1.1484, 0.1267, 301990, 2.2348e9],
            [2.9705, 0.60306, 0.42648, 325090, 1.7733e9],
            [7.9871, 7.1917, 1.8212, 244130, 3.1200e8],
            [1.7141, 2.3225, 1.9634, 199610, 8.3827e8],
        ],
        "forecast_months": 12
    }
    response = requests.post(f"{BASE_URL}/lstm-month", json=payload)
    print("\n=== LSTM Month Inference ===")
    print("Status Code:", response.status_code)
    print("Response JSON:", response.json())

def test_lstm_day_inference():
    # Generate synthetic valid data for 60 days: [close, mean_negative, mean_positive, mean_neutral]
    days = [
        [
            round(random.uniform(200, 300), 2),  # close price
            round(random.uniform(0, 1), 4),      # mean_negative sentiment
            round(random.uniform(0, 1), 4),      # mean_positive sentiment
            round(random.uniform(0, 1), 4)       # mean_neutral sentiment
        ] for _ in range(60)
    ]
    payload = {
        "ticker": "AIRA",
        "days": days,
        "forecast_days": 5
    }
    response = requests.post(f"{BASE_URL}/lstm-day", json=payload)
    print("\n=== LSTM Day Inference ===")
    print("Status Code:", response.status_code)
    print("Response JSON:", response.json())

if __name__ == "__main__":
    #test_bert_inference()
    test_lstm_month_inference()
    test_lstm_day_inference()
# import joblib

# DAY_SCALER_PATH = r".\lstm_model/scalers_one_day/all_tickers_scalers.pkl"  # Update to your actual path

# def print_daily_lstm_scaler_features():
#     scalers_day = joblib.load(DAY_SCALER_PATH)

#     for ticker, feature_scalers in scalers_day.items():
#         print(f"Ticker: {ticker}")
#         for feature_name, scaler in feature_scalers.items():
#             print(f"  Feature: {feature_name}")
#             # Try to print feature names stored in scaler, if any
#             try:
#                 features_in = getattr(scaler, 'feature_names_in_', None)
#                 if features_in is not None:
#                     print(f"    feature_names_in_: {features_in}")
#                 else:
#                     print("    No feature_names_in_ attribute")
#             except Exception as e:
#                 print(f"    Could not access feature names: {e}")

# if __name__ == "__main__":
#     print_daily_lstm_scaler_features()
