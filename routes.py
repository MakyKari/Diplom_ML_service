from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import torch
import numpy as np
import joblib
import pandas as pd
from sentiment_analyzer import analyze_sentiment
from config import MONTH_SCALER_PATH, DAY_SCALER_PATH

scalers_month = joblib.load(MONTH_SCALER_PATH)
scalers_day = joblib.load(DAY_SCALER_PATH)

class BertRequest(BaseModel):
    subject: str

class LSTMRequest(BaseModel):
    ticker: str
    sequence: List[List[float]]
    forecast_months: int

class LSTMDayRequest(BaseModel):
    ticker: str
    days: List[List[float]]
    forecast_days: int

def setup_routes(app: FastAPI, bert_model, tokenizer, lstm_model, lstm_day_model):
    """
    Setup API routes for the FastAPI application
    
    Args:
        app: FastAPI application instance
        bert_model: BERT model for sentiment analysis
        tokenizer: BERT tokenizer
        lstm_model: LSTM model
    """

    @app.post("/bert")
    def bert_inference(request: BertRequest):
        try:
            sentiment_probs = analyze_sentiment(request.subject, bert_model, tokenizer)
            return {"sentiment_probs": sentiment_probs}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
        
    @app.post("/lstm-month")
    def lstm_inference(request: LSTMRequest):
        try:
            ticker = request.ticker
            n_months = getattr(request, 'forecast_months', 1)

            if ticker in scalers_month:
                feature_scaler = scalers_month[ticker]['features']
                target_scaler = scalers_month[ticker]['target']
                normalized_sequence = feature_scaler.transform(request.sequence)
            else:
                all_features = np.array(request.sequence)
                feature_min = all_features.min(axis=0)
                feature_max = all_features.max(axis=0)
                normalized_sequence = (all_features - feature_min) / (feature_max - feature_min + 1e-6)

            current_sequence = normalized_sequence.copy()
            predictions = []

            for _ in range(n_months):
                seq_tensor = torch.tensor([current_sequence], dtype=torch.float32)
                with torch.no_grad():
                    price_pred, _ = lstm_model(seq_tensor)

                if ticker in scalers_month:
                    predicted_price = target_scaler.inverse_transform(price_pred.numpy())[0][0]
                else:
                    predicted_price = price_pred.item()

                predictions.append(round(predicted_price, 2))

                next_input = np.zeros_like(current_sequence[0])
                next_input[3] = predicted_price
                current_sequence = np.vstack([current_sequence[1:], next_input])

            return {
                "forecast_months": n_months,
                "predicted_prices": predictions
            }

        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))


    @app.post("/lstm-day")
    def lstm_day_inference(request: LSTMDayRequest):
        try:
            ticker = request.ticker
            input_days = np.array(request.days)
            forecast_days = request.forecast_days

            feature_names = ['close', 'mean_negative', 'mean_positive', 'mean_neutral']
            assert input_days.shape[1] == 4, "Each day must have 4 features: close, mean_negative, mean_positive, mean_neutral"

            df_input = pd.DataFrame(input_days, columns=feature_names)

            if ticker in scalers_day:
                df_scaled = df_input.copy()
                for feature in feature_names:
                    scaler = scalers_day[ticker][feature]
                    df_scaled[[feature]] = scaler.transform(df_input[[feature]])
                scaled = df_scaled.to_numpy()
            else:
                feature_min = input_days.min(axis=0)
                feature_max = input_days.max(axis=0)
                scaled = (input_days - feature_min) / (feature_max - feature_min + 1e-6)

            current_sequence = scaled.copy()
            predictions = []

            for _ in range(forecast_days):
                seq_tensor = torch.tensor([current_sequence], dtype=torch.float32)
                with torch.no_grad():
                    next_price_scaled = lstm_day_model(seq_tensor)

                if ticker in scalers_day and 'close' in scalers_day[ticker]:
                    scaler = scalers_day[ticker]['close']
                    next_price = scaler.inverse_transform([[next_price_scaled.item()]])[0][0]
                else:
                    next_price = next_price_scaled.item()

                predictions.append(round(next_price, 2))

                next_day = np.zeros(current_sequence.shape[1])
                next_day[0] = next_price

                current_sequence = np.vstack([current_sequence[1:], next_day])

            return {
                "forecast_days": forecast_days,
                "predicted_prices": predictions
            }

        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))