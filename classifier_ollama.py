import os
import requests
from typing import Dict

# Ollama API endpoint (default: http://localhost:11434)
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
MODEL = os.getenv("OLLAMA_MODEL", "llama3")  # or mistral, llama2, etc.

CATEGORIES = ["billing", "clinical_advice", "scheduling", "technical_support"]

def classify_message(message: str) -> Dict:
    """
    Classify patient message using open-source LLM via Ollama.
    
    Args:
        message: Patient message text
        
    Returns:
        dict with category and confidence
    """
    prompt = f"""Classify this patient message into ONE category: billing, clinical_advice, scheduling, or technical_support.

Message: "{message}"

Respond with ONLY the category name. If multiple concerns exist, choose the PRIMARY one."""

    try:
        # Try /api/generate first (more compatible)
        full_prompt = f"You are a medical message classifier. Respond with only the category name.\n\n{prompt}"
        
        response = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json={
                "model": MODEL,
                "prompt": full_prompt,
                "stream": False,
                "options": {
                    "temperature": 0,
                    "num_predict": 20
                }
            },
            timeout=30
        )
        response.raise_for_status()
        
        result = response.json()
        category = result.get("response", "").strip().lower()
        
        # Extract category from response (might have extra text)
        for cat in CATEGORIES:
            if cat in category:
                category = cat
                break
        
        # Validate category
        if category not in CATEGORIES:
            category = "clinical_advice"  # Default to safest option
        
        return {
            "category": category,
            "confidence": "high",
            "raw_response": result.get("response", "")
        }
    except Exception as e:
        return {"error": str(e)}


if __name__ == "__main__":
    # Test with example
    test_message = "I'm trying to reschedule my appointment because the app won't load, and I also need to check a charge that looks wrong on my bill."
    
    result = classify_message(test_message)
    print(f"Message: {test_message}")
    print(f"Result: {result}")

