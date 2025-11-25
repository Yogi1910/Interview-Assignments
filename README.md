# Patient Message Classification System

A text classification system designed to automatically route patient messages to the appropriate department in healthcare support centers. The system categorizes incoming messages into billing, clinical advice, scheduling, or technical support categories.

## Overview

Patient support centers handle high volumes of free-text messages through various channels including portals, chat systems, and email. Manual triage creates bottlenecks and inconsistencies. This system provides automated classification to enable faster routing, reduce backlog, and improve service operations.

The project implements multiple classification approaches, allowing you to choose the method that best fits your infrastructure, accuracy requirements, and cost constraints.

## Problem Statement

The classifier must handle short, ambiguous, and often incomplete patient messages. Patients use non-standard phrasing, frequently mix multiple concerns in a single message, and provide limited context. The system must maintain high precision for clinical messages to avoid misrouting safety-relevant inquiries. It needs to support near-real-time inference, handle class imbalance, and provide confidence outputs for cases requiring human review.

Example input: "I'm trying to reschedule my appointment because the app won't load, and I also need to check a charge that looks wrong on my bill."

## Classification Categories

- **billing**: Questions about charges, payments, insurance claims, bills, or financial matters
- **clinical_advice**: Medical questions, symptoms, medication inquiries, or health concerns requiring clinical expertise
- **scheduling**: Appointment requests, cancellations, rescheduling, or availability questions
- **technical_support**: Issues with digital platforms, apps, websites, login problems, or technical difficulties
- **unknown**: Messages that are unclear, ambiguous, or don't fit any category above

## Architecture

The system provides four different classification implementations:

### 1. Traditional ML Classifier (`classifier_traditional.py`)

Uses TF-IDF vectorization with Logistic Regression. This is the fastest and most lightweight option, suitable for environments with limited computational resources.

- **Pros**: Fast inference, no external API dependencies, works offline
- **Cons**: Requires training data, may struggle with nuanced language
- **Best for**: Production environments with labeled training data and strict latency requirements

### 2. Transformer-Based Classifier (`classifier_transformer.py`)

Uses Facebook's BART-large-MNLI model for zero-shot classification. No training required, but model weights must be downloaded on first run.

- **Pros**: No training data needed, good accuracy out of the box, handles nuanced language
- **Cons**: Requires significant memory (model is ~1.6GB), slower inference than traditional ML
- **Best for**: Environments where you want good accuracy without training data, and have GPU resources available

### 3. OpenAI Classifier (`classifier_openai.py`)

Uses GPT-4 via OpenAI's API. Provides the highest accuracy and natural language understanding, with detailed reasoning outputs.

- **Pros**: Highest accuracy, provides reasoning explanations, handles complex multi-intent messages well
- **Cons**: Requires API key, incurs per-request costs, depends on external service availability
- **Best for**: High-stakes applications where accuracy is critical and budget allows for API costs

### 4. Ollama Classifier (`classifier_ollama.py`)

Uses open-source LLMs via Ollama, running locally. Provides LLM-level understanding without cloud dependencies.

- **Pros**: No API costs, runs locally, good privacy for sensitive medical data
- **Cons**: Requires local Ollama installation and model download, slower than cloud APIs
- **Best for**: Environments with privacy requirements or where you want to avoid API costs while maintaining LLM capabilities

## Installation

### Prerequisites

- Python 3.8 or higher
- pip or your preferred Python package manager

### Setup Steps

1. Clone the repository:
```bash
git clone <repository-url>
cd Interview-Assignments
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Configure environment variables (for OpenAI and Ollama classifiers):

Create a `.env` file in the project root:
```bash
# For OpenAI classifier
OPENAI_API_KEY=your_openai_api_key_here

# For Ollama classifier (optional, defaults shown)
OLLAMA_URL=http://localhost:11434
OLLAMA_MODEL=llama3
```

### Additional Setup for Specific Classifiers

**For Transformer Classifier:**
The first run will automatically download the BART-large-MNLI model (~1.6GB). Ensure you have sufficient disk space and a stable internet connection.

**For Ollama Classifier:**
1. Install Ollama from https://ollama.ai
2. Pull the desired model:
```bash
ollama pull llama3
```
Or use another model like `mistral` or `llama2`.

## Usage

### Streamlit Web Interface

The easiest way to use the system is through the Streamlit web interface:

```bash
streamlit run streamlit_app.py
```

This launches a web interface where you can:
- Enter patient messages directly
- Select from example messages
- View classification results with category, confidence, and reasoning
- See raw API responses for debugging

The interface currently uses the OpenAI classifier by default. To switch classifiers, modify the import in `streamlit_app.py`:

```python
from classifier_openai import classify_message  # Change this line
# from classifier_traditional import classify_message
# from classifier_transformer import classify_message
# from classifier_ollama import classify_message
```

### Programmatic Usage

Each classifier can be used directly in Python:

```python
from classifier_openai import classify_message

result = classify_message("I need to check my bill from last month")

print(f"Category: {result['category']}")
print(f"Confidence: {result.get('confidence', 'N/A')}")
print(f"Message: {result.get('message', '')}")
```

The response format varies slightly by classifier:

**OpenAI Response:**
```python
{
    "category": "billing",
    "confidence": "high",
    "reasoning": "Patient is asking about their bill...",
    "message": "I am redirecting you to billing experts.",
    "raw_response": "{...}"
}
```

**Traditional/Transformer Response:**
```python
{
    "category": "billing",
    "confidence": 0.95,
    "all_scores": {
        "billing": 0.95,
        "clinical_advice": 0.02,
        "scheduling": 0.02,
        "technical_support": 0.01
    }
}
```

**Ollama Response:**
```python
{
    "category": "billing",
    "confidence": "high",
    "raw_response": "billing"
}
```

All classifiers return an error dictionary if something goes wrong:
```python
{"error": "Error message here"}
```

## Configuration

### Prompt Customization

The OpenAI classifier uses a prompt template stored in `prompt.txt`. You can modify this file to adjust the classification behavior, add examples, or change the response format.

### Model Selection

**Transformer Classifier:** Change the `MODEL_NAME` in `classifier_transformer.py` to use a different zero-shot classification model. Popular alternatives include `facebook/bart-large-mnli`, `typeform/distilbert-base-uncased-mnli`, or fine-tuned models.

**Ollama Classifier:** Set the `OLLAMA_MODEL` environment variable or modify the default in `classifier_ollama.py`. Available models depend on what you've pulled with Ollama.

### Training Data (Traditional Classifier)

The traditional classifier includes minimal sample training data. For production use, replace the `TRAINING_DATA` list in `classifier_traditional.py` with your labeled dataset. The current implementation uses 12 examples across 4 categories - you'll want significantly more data for real-world accuracy.

## Performance Considerations

**Latency:**
- Traditional ML: < 10ms per message
- Transformer: 100-500ms per message (CPU), 20-50ms (GPU)
- OpenAI: 500-2000ms per message (network dependent)
- Ollama: 200-1000ms per message (hardware dependent)

**Resource Requirements:**
- Traditional ML: Minimal (works on any machine)
- Transformer: 2-4GB RAM, optional GPU recommended
- OpenAI: Minimal (API-based)
- Ollama: 4-8GB RAM, model size dependent

**Cost:**
- Traditional ML: Free (compute only)
- Transformer: Free (compute only)
- OpenAI: ~$0.01-0.03 per 1000 messages (GPT-4 pricing)
- Ollama: Free (local compute)

## Error Handling

All classifiers implement try-except blocks and return error dictionaries on failure. Common issues:

- **OpenAI**: Missing API key, rate limits, network issues
- **Transformer**: Out of memory, model download failures
- **Ollama**: Service not running, model not found, network issues
- **Traditional**: Model training failures (shouldn't occur with current setup)

Check the error message in the returned dictionary for specific failure reasons.

## Testing

Each classifier includes a test example in its `__main__` block. Run individual classifiers:

```bash
python classifier_openai.py
python classifier_traditional.py
python classifier_transformer.py
python classifier_ollama.py
```

## Limitations and Future Improvements

**Current Limitations:**
- Traditional classifier uses minimal training data
- No batch processing support
- Limited handling of multi-language messages
- No confidence thresholding for automatic routing
- No logging or monitoring built-in

**Potential Enhancements:**
- Fine-tune transformer models on domain-specific data
- Implement ensemble methods combining multiple classifiers
- Add batch processing for high-volume scenarios
- Integrate with message queue systems (RabbitMQ, Kafka)
- Add comprehensive logging and metrics collection
- Implement confidence-based routing rules
- Support for additional categories or custom categories
- Multi-language support with language detection

## Project Structure

```
Interview-Assignments/
├── classifier_traditional.py    # TF-IDF + Logistic Regression
├── classifier_transformer.py     # BART zero-shot classification
├── classifier_openai.py         # GPT-4 via OpenAI API
├── classifier_ollama.py          # Local LLM via Ollama
├── streamlit_app.py              # Web interface
├── prompt.txt                    # OpenAI prompt template
├── requirements.txt              # Python dependencies
├── expectation.txt              # Project requirements/context
└── README.md                     # This file
```

## Dependencies

Key dependencies include:
- `streamlit`: Web interface framework
- `openai`: OpenAI API client
- `transformers`: Hugging Face transformer models
- `torch`: PyTorch for transformer models
- `scikit-learn`: Traditional ML algorithms
- `requests`: HTTP client for Ollama
- `python-dotenv`: Environment variable management

See `requirements.txt` for complete version specifications.

## License

[Specify your license here]

## Contributing

[Add contribution guidelines if applicable]

## Contact

For questions or issues, please [add contact information or issue tracker link].
