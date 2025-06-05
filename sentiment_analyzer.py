import torch
from torch.nn.functional import softmax

BERT_LABELS = ['neutral', 'positive', 'negative']

def analyze_sentiment(text, model, tokenizer):
    """
    Analyze sentiment of given text using BERT model
    
    Args:
        text (str): Text to analyze
        model: BERT model
        tokenizer: BERT tokenizer
        
    Returns:
        dict: Sentiment probabilities for each label
    """
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = softmax(outputs.logits, dim=1).squeeze()
        prob_values = probs.tolist()
        sentiment_probs = {label: float(prob) for label, prob in zip(BERT_LABELS, prob_values)}
        return sentiment_probs