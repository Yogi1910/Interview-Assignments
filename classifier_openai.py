import os
import json
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

CATEGORIES = ["billing", "clinical_advice", "scheduling", "technical_support", "unknown"]

def load_prompt_template() -> str:
    """Load prompt template from file."""
    prompt_file = os.path.join(os.path.dirname(__file__), "prompt.txt")
    with open(prompt_file, "r", encoding="utf-8") as f:
        return f.read().strip()

def classify_message(message: str) -> dict:
    """
    Classify patient message using OpenAI GPT.
    
    Args:
        message: Patient message text
        
    Returns:
        dict with category and confidence
    """
    prompt_template = load_prompt_template()
    prompt = prompt_template.replace("{message}", message)

    try:
        response = client.chat.completions.create(
            model="gpt-4.1",
            messages=[
                {"role": "system", "content": "You are a medical message classifier. Always respond with valid JSON only."},
                {"role": "user", "content": prompt}
            ],
            temperature=0,
            max_tokens=150,
            response_format={"type": "json_object"}
        )
        
        raw_response = response.choices[0].message.content.strip()
        
        # Parse JSON response
        try:
            result_json = json.loads(raw_response)
            category = result_json.get("category", "").lower()
            reasoning = result_json.get("reasoning", "")
            chat_message = result_json.get("message", "")
        except json.JSONDecodeError:
            # Fallback: try to extract category from text
            category = raw_response.lower()
            reasoning = ""
            chat_message = ""
        
        # Validate category
        if category not in CATEGORIES:
            category = "unknown"
        
        # Generate chat message if not provided
        if not chat_message:
            chat_messages = {
                "billing": "I am redirecting you to billing experts.",
                "clinical_advice": "I am redirecting you to clinical experts.",
                "scheduling": "I am redirecting you to scheduling experts.",
                "technical_support": "I am redirecting you to technical support experts.",
                "unknown": "Can you elaborate more?"
            }
            chat_message = chat_messages.get(category, "Can you elaborate more?")
        
        return {
            "category": category,
            "reasoning": reasoning,
            "message": chat_message,
            "confidence": "high",
            "raw_response": raw_response
        }
    except Exception as e:
        return {"error": str(e)}


if __name__ == "__main__":
    # Test with example
    test_message = "I'm trying to reschedule my appointment because the app won't load, and I also need to check a charge that looks wrong on my bill."
    
    result = classify_message(test_message)
    print(f"Message: {test_message}")
    print(f"Result: {result}")

