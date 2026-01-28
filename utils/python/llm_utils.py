"""
LLM Utility Functions for Python
Functions to interact with local Ollama LLM
"""

import requests
from typing import Optional


# Configuration
OLLAMA_URL = "http://localhost:11434"
DEFAULT_MODEL = "llama3.2"


def ask_ollama(prompt: str, model: str = DEFAULT_MODEL, temperature: float = 0.1) -> Optional[str]:
    """
    Call Ollama LLM with a prompt.

    Parameters
    ----------
    prompt : str
        The question or instruction for the LLM
    model : str
        Which Ollama model to use (default: llama3.2)
    temperature : float
        How creative the response should be (0.0 = focused, 1.0 = creative)

    Returns
    -------
    str or None
        The LLM's response, or None if there was an error
    """
    url = f"{OLLAMA_URL}/api/generate"

    data = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": temperature
        }
    }

    try:
        response = requests.post(url, json=data, timeout=120)

        if response.status_code == 200:
            return response.json()['response'].strip()
        else:
            print(f"Warning: Ollama returned status code {response.status_code}")
            return None

    except requests.exceptions.ConnectionError:
        print("Error: Can't connect to Ollama. Is it running? Try: ollama serve")
        return None
    except requests.exceptions.Timeout:
        print("Error: Request timed out")
        return None
    except Exception as e:
        print(f"Error calling Ollama: {str(e)}")
        return None


def extract_gene_name(user_question: str) -> Optional[str]:
    """
    Extract gene name from natural language query using LLM.

    Parameters
    ----------
    user_question : str
        The user's question in natural language

    Returns
    -------
    str or None
        The extracted gene name (uppercase), or None if not found
    """
    prompt = (
        "Extract ONLY the gene name from this question. "
        "Return ONLY the gene name in uppercase, nothing else. "
        "If no gene name is found, return 'NONE'.\n\n"
        f"Question: {user_question}\n"
        "Gene name:"
    )

    response = ask_ollama(prompt, temperature=0.1)

    if response is None:
        return None

    # Clean up response
    gene_name = response.upper().strip()

    # Remove any non-alphanumeric characters except hyphen
    import re
    gene_name = re.sub(r'[^A-Z0-9-]', '', gene_name)

    # Check if valid
    if gene_name == "NONE" or gene_name == "" or len(gene_name) == 0:
        return None

    return gene_name


def extract_gene_name_with_details(user_question: str) -> dict:
    """
    Extract gene name from natural language query using LLM.
    Returns detailed information about the LLM interaction.

    Parameters
    ----------
    user_question : str
        The user's question in natural language

    Returns
    -------
    dict
        Dictionary with keys:
        - 'gene_name': Extracted gene name (str or None)
        - 'prompt': The prompt sent to LLM
        - 'llm_response': Raw response from LLM
        - 'success': Whether extraction succeeded (bool)
    """
    prompt = (
        "Extract ONLY the gene name from this question. "
        "Return ONLY the gene name in uppercase, nothing else. "
        "If no gene name is found, return 'NONE'.\n\n"
        f"Question: {user_question}\n"
        "Gene name:"
    )

    response = ask_ollama(prompt, temperature=0.1)

    result = {
        'prompt': prompt,
        'llm_response': response if response else "No response from LLM",
        'gene_name': None,
        'success': False
    }

    if response is None:
        return result

    # Clean up response
    gene_name = response.upper().strip()

    # Store original response for display
    result['llm_response'] = response

    # Remove any non-alphanumeric characters except hyphen
    import re
    gene_name = re.sub(r'[^A-Z0-9-]', '', gene_name)

    # Check if valid
    if gene_name == "NONE" or gene_name == "" or len(gene_name) == 0:
        return result

    result['gene_name'] = gene_name
    result['success'] = True
    return result


def check_ollama_status() -> bool:
    """
    Check if Ollama is running and accessible.

    Returns
    -------
    bool
        True if Ollama is running, False otherwise
    """
    try:
        response = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
        return response.status_code == 200
    except:
        return False
