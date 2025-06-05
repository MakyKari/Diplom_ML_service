RABBITMQ_HOST = '164.90.167.226'
RABBITMQ_PORT = 5672
RABBITMQ_USER = 'guest'
RABBITMQ_PASS = 'guest'

INPUT_QUEUE = 'kase_news'
OUTPUT_QUEUE = 'kase_news_with_sentiment'

INPUT_SIZE_MONTH = 5
HIDDEN_SIZE_MONTH = 32
NUM_LAYERS_MONTH = 2

INPUT_SIZE_DAY = 4
HIDDEN_SIZE_DAY = 256
NUM_LAYERS_DAY = 4

BERT_MODEL_PATH = r".\bert_classifier"
MONTH_LSTM_MODEL_PATH = r".\lstm_model\best_two_headed_lstm_32.pth"
MONTH_SCALER_PATH = r'.\lstm_model/scalers_by_ticker.pkl' 
DAY_LSTM_MODEL_PATH = r".\lstm_model\best_model_hs256_nl4_day.pth" 
DAY_SCALER_PATH = r".\lstm_model/scalers_one_day/all_tickers_scalers.pkl"
