import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from typing import Dict
import numpy as np

CATEGORIES = ["billing", "clinical_advice", "scheduling", "technical_support"]

# Sample training data (in production, load from dataset)
TRAINING_DATA = [
    ("I need to check my bill", "billing"),
    ("There's a charge I don't recognize", "billing"),
    ("My insurance claim was denied", "billing"),
    ("I have chest pain", "clinical_advice"),
    ("What are the side effects of this medication?", "clinical_advice"),
    ("I need to see a doctor", "clinical_advice"),
    ("Can I reschedule my appointment?", "scheduling"),
    ("I need to cancel my visit", "scheduling"),
    ("When is my next appointment?", "scheduling"),
    ("The app won't load", "technical_support"),
    ("I can't log into my account", "technical_support"),
    ("The website is down", "technical_support"),
]

# Initialize and train model
def train_model():
    """Train TF-IDF + Logistic Regression classifier."""
    X = [text for text, _ in TRAINING_DATA]
    y = [label for _, label in TRAINING_DATA]
    
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=1000, ngram_range=(1, 2))),
        ('clf', LogisticRegression(max_iter=1000, class_weight='balanced'))
    ])
    
    pipeline.fit(X, y)
    return pipeline

# Train model on startup
_model = train_model()

def classify_message(message: str) -> Dict:
    """
    Classify patient message using traditional ML (TF-IDF + Logistic Regression).
    
    Args:
        message: Patient message text
        
    Returns:
        dict with category and confidence
    """
    try:
        # Get prediction probabilities
        probabilities = _model.predict_proba([message])[0]
        predicted_idx = np.argmax(probabilities)
        confidence = probabilities[predicted_idx]
        category = _model.classes_[predicted_idx]
        
        # Get all category scores
        all_scores = dict(zip(_model.classes_, probabilities))
        
        return {
            "category": category,
            "confidence": round(float(confidence), 3),
            "all_scores": all_scores
        }
    except Exception as e:
        return {"error": str(e)}


if __name__ == "__main__":
    # Test with example
    test_message = "I'm trying to reschedule my appointment because the app won't load, and I also need to check a charge that looks wrong on my bill."
    
    result = classify_message(test_message)
    print(f"Message: {test_message}")
    print(f"Result: {result}")

