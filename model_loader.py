import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from lstm_class import TwoHeadedMonthLSTM, DailyLSTMRegressor
from config import *

def load_models():
    """Load and initialize all models"""
    print("[*] Loading models...")

    tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL_PATH)
    bert_model = AutoModelForSequenceClassification.from_pretrained(BERT_MODEL_PATH)
    bert_model.eval()

    lstm_model = TwoHeadedMonthLSTM(INPUT_SIZE_MONTH, HIDDEN_SIZE_MONTH, NUM_LAYERS_MONTH)
    lstm_model.load_state_dict(torch.load(MONTH_LSTM_MODEL_PATH, map_location=torch.device('cpu')))
    lstm_model.eval()

    lstm_day_model = DailyLSTMRegressor(INPUT_SIZE_DAY, HIDDEN_SIZE_DAY, NUM_LAYERS_DAY)
    lstm_day_model.load_state_dict(torch.load(DAY_LSTM_MODEL_PATH, map_location=torch.device('cuda')))
    lstm_day_model.eval()

    print("[*] Models loaded successfully")
    return bert_model, tokenizer, lstm_model, lstm_day_model