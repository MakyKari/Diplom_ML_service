from fastapi import FastAPI
import uvicorn
import threading
from routes import setup_routes
from consumer import run_rabbitmq_consumer
from model_loader import load_models

app = FastAPI()

bert_model, tokenizer, lstm_model, lstm_day_model = load_models()

setup_routes(app, bert_model, tokenizer, lstm_model, lstm_day_model)

@app.on_event("startup")
async def startup_event():
    """Start the RabbitMQ consumer in a separate thread when the FastAPI app starts"""
    print("[*] Starting RabbitMQ consumer thread...")
    # Start RabbitMQ consumer in a separate thread
    threading.Thread(target=run_rabbitmq_consumer, 
                     args=(bert_model, tokenizer), 
                     daemon=True).start()
    print("[*] RabbitMQ consumer thread started")

if __name__ == "__main__":
    print("[*] Starting FastAPI application...")
    uvicorn.run(app, host="0.0.0.0", port=8100, log_level="info")