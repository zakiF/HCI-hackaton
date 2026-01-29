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
    "Extract ALL gene names from this query.\n"
    "Return ONLY valid JSON, nothing else. No explanation, no markdown, no code blocks.\n"
    "Format: {\"genes\": [\"GENE1\", \"GENE2\"]}\n\n"
    f"Query: {user_input}\n\n"
    "JSON:")

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
    Extract gene name(s) from natural language query using LLM.
    Returns detailed information about the LLM interaction.

    Parameters
    ----------
    user_question : str
        The user's question in natural language

    Returns
    -------
    dict
        Dictionary with keys:
        - 'gene_name': First extracted gene name (str or None) - for backwards compatibility
        - 'gene_names': List of all extracted gene names (list)
        - 'prompt': The prompt sent to LLM
        - 'llm_response': Raw response from LLM
        - 'success': Whether extraction succeeded (bool)
    """
    prompt = (
        "Extract ALL gene names from this query.\n"
        "Return ONLY valid JSON, nothing else. No explanation, no markdown, no code blocks.\n"
        "Format: {\"genes\": [\"GENE1\", \"GENE2\"]}\n"
        "If only one gene, return: {\"genes\": [\"GENE1\"]}\n"
        "If no genes found, return: {\"genes\": []}\n\n"
        f"Question: {user_question}\n\n"
        "JSON:"
    )

    response = ask_ollama(prompt, temperature=0.1)

    result = {
        'prompt': prompt,
        'llm_response': response if response else "No response from LLM",
        'gene_name': None,
        'gene_names': [],
        'success': False
    }

    if response is None:
        return result

    # Store original response for display
    result['llm_response'] = response

    # Try to parse JSON response
    import json
    import re

    try:
        # Clean the response - extract just the JSON part
        response_clean = response.strip()

        # Remove markdown code blocks if present
        response_clean = re.sub(r'^```json\s*', '', response_clean)
        response_clean = re.sub(r'^```\s*', '', response_clean)
        response_clean = re.sub(r'\s*```$', '', response_clean)

        # Extract content between first { and last }
        if '{' in response_clean and '}' in response_clean:
            start = response_clean.index('{')
            end = response_clean.rindex('}') + 1
            response_clean = response_clean[start:end]

        # Parse JSON
        parsed = json.loads(response_clean)
        genes = parsed.get('genes', [])

        if genes and len(genes) > 0:
            # Clean up gene names
            clean_genes = []
            for gene in genes:
                gene_upper = str(gene).upper().strip()
                # Remove non-alphanumeric except hyphen
                gene_clean = re.sub(r'[^A-Z0-9-]', '', gene_upper)
                if gene_clean and gene_clean != "NONE":
                    clean_genes.append(gene_clean)

            if clean_genes:
                result['gene_names'] = clean_genes
                result['gene_name'] = clean_genes[0]  # First gene for backwards compatibility
                result['success'] = True

    except (json.JSONDecodeError, ValueError, KeyError) as e:
        # JSON parsing failed - try fallback: extract gene-like words
        import re
        # Look for capitalized words that look like gene names (2-10 uppercase letters/numbers)
        potential_genes = re.findall(r'\b[A-Z][A-Z0-9-]{1,9}\b', response.upper())

        if potential_genes:
            # Filter out common words
            exclude = ['NONE', 'NULL', 'UNKNOWN', 'GENE', 'JSON', 'TRUE', 'FALSE']
            genes = [g for g in potential_genes if g not in exclude]

            if genes:
                result['gene_names'] = genes
                result['gene_name'] = genes[0]
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
