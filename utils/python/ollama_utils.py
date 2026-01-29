"""
Shared Ollama Utility Functions for Python
All team members can import and use these functions

This provides a consistent interface for calling the local LLM.
"""

import requests
import json
from typing import Optional, Dict, Any


# ============================================
# Configuration
# ============================================

OLLAMA_URL = "http://localhost:11434"
DEFAULT_MODEL = "llama3.2"  # Everyone uses this model


# ============================================
# Basic LLM Calling
# ============================================

def call_ollama(
    prompt: str,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.1,
    max_tokens: int = 500,
    **kwargs
) -> Optional[str]:
    """
    Call Ollama LLM with a prompt and get text response.

    Args:
        prompt: The question or instruction for the LLM
        model: Which Ollama model to use (default: llama3.2)
        temperature: How creative (0.0 = focused, 1.0 = creative)
        max_tokens: Maximum length of response
        **kwargs: Additional options passed to Ollama

    Returns:
        str: The LLM's response text, or None if error

    Example:
        >>> response = call_ollama("What is the capital of France?")
        >>> print(response)
        "Paris"
    """
    url = f"{OLLAMA_URL}/api/generate"

    data = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens,
            **kwargs
        }
    }

    try:
        response = requests.post(url, json=data, timeout=120)

        if response.status_code == 200:
            result = response.json()
            return result.get('response', '').strip()
        else:
            print(f"Warning: Ollama returned status code {response.status_code}")
            return None

    except requests.exceptions.ConnectionError:
        print("Error: Can't connect to Ollama. Is it running? Try: ollama serve")
        return None
    except requests.exceptions.Timeout:
        print("Error: Request timed out (LLM took too long)")
        return None
    except Exception as e:
        print(f"Error calling Ollama: {str(e)}")
        return None


# ============================================
# Advanced: Structured Output
# ============================================

def call_ollama_json(
    prompt: str,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.1,
    **kwargs
) -> Optional[Dict[str, Any]]:
    """
    Call Ollama and parse response as JSON.

    Useful for structured extraction tasks where you want the LLM
    to return data in a specific format.

    Args:
        prompt: Should ask LLM to return JSON
        model: Which model to use
        temperature: Lower is better for structured output
        **kwargs: Additional options

    Returns:
        dict: Parsed JSON response, or None if error

    Example:
        >>> prompt = '''
        ... Extract gene name and plot type from this query.
        ... Return as JSON with keys: "gene_name", "plot_type"
        ...
        ... Query: Show me TP53 as a violin plot
        ... JSON:
        ... '''
        >>> result = call_ollama_json(prompt)
        >>> print(result)
        {"gene_name": "TP53", "plot_type": "violin"}
    """
    # Call LLM
    response_text = call_ollama(prompt, model=model, temperature=temperature, **kwargs)

    if response_text is None:
        return None

    # Try to parse as JSON
    try:
        # Sometimes LLM adds markdown code blocks, remove them
        response_text = response_text.strip()
        if response_text.startswith("```json"):
            response_text = response_text[7:]
        if response_text.startswith("```"):
            response_text = response_text[3:]
        if response_text.endswith("```"):
            response_text = response_text[:-3]

        # Parse JSON
        result = json.loads(response_text.strip())
        return result

    except json.JSONDecodeError as e:
        print(f"Warning: Could not parse LLM response as JSON: {e}")
        print(f"Raw response: {response_text}")
        return None


# ============================================
# Status Checking
# ============================================

def check_ollama_status() -> bool:
    """
    Check if Ollama is running and accessible.

    Returns:
        bool: True if Ollama is running, False otherwise

    Example:
        >>> if check_ollama_status():
        ...     print("Ollama is ready!")
        ... else:
        ...     print("Start Ollama with: ollama serve")
    """
    try:
        response = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
        return response.status_code == 200
    except:
        return False


def get_available_models() -> list:
    """
    Get list of models available in Ollama.

    Returns:
        list: Model names, or empty list if error

    Example:
        >>> models = get_available_models()
        >>> print(models)
        ['llama3.2', 'codellama', 'mistral']
    """
    try:
        response = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
        if response.status_code == 200:
            data = response.json()
            return [model['name'] for model in data.get('models', [])]
        return []
    except:
        return []


# ============================================
# Convenience Functions for Common Tasks
# ============================================

def ask_llm_to_classify(
    text: str,
    options: list,
    context: str = ""
) -> Optional[str]:
    """
    Ask LLM to classify text into one of several categories.

    Args:
        text: The text to classify
        options: List of possible categories
        context: Additional context to help LLM

    Returns:
        str: The selected category (one of options), or None

    Example:
        >>> query = "Show me TP53 expression"
        >>> category = ask_llm_to_classify(
        ...     query,
        ...     options=["plot", "stats", "question"],
        ...     context="Classify this user query"
        ... )
        >>> print(category)
        "plot"
    """
    options_str = ", ".join(options)

    prompt = f"""
{context}

Choose ONLY ONE from these options: {options_str}
Return ONLY the option name, nothing else.

Text: {text}

Choice:
"""

    response = call_ollama(prompt, temperature=0.0)

    if response is None:
        return None

    # Clean and validate response
    response = response.strip().lower()

    for option in options:
        if option.lower() in response:
            return option

    return None


def ask_llm_to_extract(
    text: str,
    what_to_extract: str,
    example: str = ""
) -> Optional[str]:
    """
    Ask LLM to extract specific information from text.

    Args:
        text: The text to extract from
        what_to_extract: Description of what to extract
        example: Optional example of desired output

    Returns:
        str: Extracted information, or None

    Example:
        >>> query = "Show me TP53 and BRCA1 expression"
        >>> genes = ask_llm_to_extract(
        ...     query,
        ...     what_to_extract="gene names",
        ...     example="TP53, BRCA1"
        ... )
        >>> print(genes)
        "TP53, BRCA1"
    """
    prompt = f"""
Extract {what_to_extract} from this text.
Return ONLY the extracted information, nothing else.
"""

    if example:
        prompt += f"\nExample output: {example}\n"

    prompt += f"""
Text: {text}

Extracted {what_to_extract}:
"""

    response = call_ollama(prompt, temperature=0.1)
    return response.strip() if response else None


# ============================================
# Learning Objectives Helper
# ============================================

def demonstrate_llm_skills():
    """
    Demonstration of the 4 main LLM skills for the hackathon:
    1. Intent classification
    2. Parameter extraction
    3. Natural language generation
    4. Structured output (JSON)

    Run this to see examples of each skill.
    """
    print("=== LLM Skills Demonstration ===\n")

    # Skill 1: Intent Classification
    print("1. INTENT CLASSIFICATION")
    user_query = "Is TP53 significantly different between groups?"
    intent = ask_llm_to_classify(
        user_query,
        options=["plot", "stats", "question"],
        context="What is the user's intent?"
    )
    print(f"   Query: {user_query}")
    print(f"   Intent: {intent}\n")

    # Skill 2: Parameter Extraction
    print("2. PARAMETER EXTRACTION")
    user_query = "Show TP53 in tumor samples only"
    genes = ask_llm_to_extract(user_query, "gene names")
    filters = ask_llm_to_extract(user_query, "filter conditions")
    print(f"   Query: {user_query}")
    print(f"   Genes: {genes}")
    print(f"   Filters: {filters}\n")

    # Skill 3: Natural Language Generation
    print("3. NATURAL LANGUAGE GENERATION")
    stats_data = "t-test: statistic=-2.45, p-value=0.023"
    prompt = f"Explain these statistical results in one sentence: {stats_data}"
    explanation = call_ollama(prompt, temperature=0.3)
    print(f"   Data: {stats_data}")
    print(f"   Explanation: {explanation}\n")

    # Skill 4: Structured Output (JSON)
    print("4. STRUCTURED OUTPUT (JSON)")
    user_query = "Compare TP53 and BRCA1 with a heatmap"
    prompt = f"""
Extract information from this query and return as JSON with these keys:
- genes: list of gene names
- plot_type: type of plot requested

Query: {user_query}

JSON:
"""
    result = call_ollama_json(prompt)
    print(f"   Query: {user_query}")
    print(f"   JSON: {json.dumps(result, indent=2)}\n")


if __name__ == "__main__":
    # Test basic connection
    print("Testing Ollama connection...")
    if check_ollama_status():
        print("✓ Ollama is running!\n")

        # Show available models
        models = get_available_models()
        print(f"Available models: {', '.join(models)}\n")

        # Run demonstration
        demonstrate_llm_skills()
    else:
        print("✗ Ollama is not running. Start it with: ollama serve")
