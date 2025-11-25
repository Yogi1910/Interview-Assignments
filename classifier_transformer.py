from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from typing import Dict
import torch

CATEGORIES = ["billing", "clinical_advice", "scheduling", "technical_support"]

# Use zero-shot classification (no training needed)
# Can also use fine-tuned models if you have training data
MODEL_NAME = "facebook/bart-large-mnli"  # Fast and accurate for zero-shot

def get_classifier():
    """Initialize transformer classifier."""
    return pipeline(
        "zero-shot-classification",
        model=MODEL_NAME,
        device=-1 if not torch.cuda.is_available() else 0  # CPU if no GPU
    )

# Initialize classifier
_classifier = get_classifier()

def classify_message(message: str) -> Dict:
    """
    Classify patient message using transformer-based model.
    
    Args:
        message: Patient message text
        
    Returns:
        dict with category and confidence
    """
    try:
        result = _classifier(message, CATEGORIES)
        
        category = result["labels"][0]  # Top prediction
        confidence = result["scores"][0]
        
        # Get all category scores
        all_scores = dict(zip(result["labels"], result["scores"]))
        
        return {
            "category": category,
            "confidence": round(confidence, 3),
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


