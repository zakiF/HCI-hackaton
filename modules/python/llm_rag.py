"""
Gene Information RAG (Retrieval-Augmented Generation) Module
Team Member: David Stone

LLM Learning Goals:
1. Intent classification: Detect information questions vs visualization requests
2. RAG pattern: Retrieval (from knowledge base) + Generation (LLM explanation)
3. Natural language generation: Convert technical info to readable explanations
4. Context injection: Feeding retrieved data into LLM prompts

Examples:
- "What does TP53 do?" → Retrieve TP53 info → LLM explains
- "Tell me about BRCA1" → Retrieve BRCA1 info → LLM explains
- "What is the function of EGFR?" → Retrieve EGFR info → LLM explains
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from utils.python.ollama_utils import call_ollama, ask_llm_to_classify, ask_llm_to_extract
import pandas as pd
import requests

DEBUG = False

# ============================================
# Intent Detection
# ============================================

def is_gene_question(user_input: str) -> bool:
    """
    Detect if user is asking ABOUT a gene (not plotting it).

    Args:
        user_input: User's query

    Returns:
        bool: True if this is an information question

    Example:
        >>> is_gene_question("What does TP53 do?")
        True
        >>> is_gene_question("Show me TP53")
        False

    TODO for David:
    1. Use LLM to classify intent OR use keyword detection
    2. Look for question words: what, tell, explain, function, role, etc.
    3. Distinguish from visualization requests
    """

    # Prefer LLM classification for better intent detection; fall back to keyword heuristics on failure
    category = ask_llm_to_classify(
        user_input,
        options=["info", "plot"],
        context=(
            "Classify the intent of this gene-related query. "
            "Return only 'info' if the user is asking for gene information/description. "
            "Return only 'plot' if the user is asking to visualize/plot expression."
        ),
    )
    if DEBUG:
        print(f"Intent classification result: {category}")

    if category:
        return category.lower() == "info"

    # Keyword fallback (keeps previous behavior if LLM is unavailable)
    question_keywords = [
        'what', 'tell me', 'explain', 'describe',
        'function', 'role', 'purpose', 'about',
        'information', 'details', 'gene info'
    ]

    user_lower = user_input.lower()

    for keyword in question_keywords:
        if keyword in user_lower and not any(plot_word in user_lower for plot_word in ['show', 'plot', 'display', 'graph']):
            return True

    return False


# ============================================
# Knowledge Retrieval
# ============================================

def retrieve_gene_annotation(gene_name: str) -> dict:
    """
    Retrieve gene information from knowledge base.

    For the hackathon, this could be:
    1. A CSV file with gene annotations
    2. An API call to a gene database
    3. A simple dictionary of common genes

    Args:
        gene_name: Gene symbol (e.g., "TP53")

    Returns:
        dict with gene information

    Example:
        >>> info = retrieve_gene_annotation("TP53")
        >>> print(info['description'])
        "Tumor protein p53, tumor suppressor..."

    TODO for David:
    1. Create a simple gene annotation file/dict
    2. Load it when module starts
    3. Return gene info if found, else return basic message
    4. Consider using real APIs like MyGene.info or NCBI
    """

    # Simple hardcoded knowledge base (David can expand this)
    gene_database = {
        "TP53": {
            "full_name": "Tumor Protein P53",
            "description": "TP53 is a tumor suppressor gene that encodes protein p53. It plays a critical role in preventing cancer by regulating cell division, DNA repair, and apoptosis. Mutations in TP53 are found in over 50% of human cancers.",
            "aliases": ["p53", "TRP53"],
            "chromosome": "17p13.1",
            "function": "Tumor suppression, cell cycle regulation, apoptosis"
        },
        "BRCA1": {
            "full_name": "Breast Cancer Type 1 Susceptibility Protein",
            "description": "BRCA1 is a human tumor suppressor gene responsible for repairing DNA. Mutations in BRCA1 significantly increase the risk of breast and ovarian cancers. It plays a role in DNA damage repair, cell cycle checkpoints, and chromatin remodeling.",
            "aliases": ["IRIS", "PSCP", "BRCAI"],
            "chromosome": "17q21.31",
            "function": "DNA repair, tumor suppression"
        },
        "EGFR": {
            "full_name": "Epidermal Growth Factor Receptor",
            "description": "EGFR is a cell surface receptor for epidermal growth factor. It regulates cell growth, survival, and differentiation. Overexpression or mutations in EGFR are associated with several cancers including lung, colorectal, and glioblastoma.",
            "aliases": ["ERBB1", "HER1"],
            "chromosome": "7p11.2",
            "function": "Cell growth, proliferation, differentiation"
        },
        "MYC": {
            "full_name": "MYC Proto-Oncogene",
            "description": "MYC is a transcription factor that regulates expression of many genes involved in cell proliferation, growth, and apoptosis. It is one of the most commonly activated oncogenes in human cancers.",
            "aliases": ["c-Myc", "bHLHe39"],
            "chromosome": "8q24.21",
            "function": "Transcription regulation, cell proliferation"
        }
    }

    # Look up gene
    gene_upper = gene_name.upper()

    if gene_upper in gene_database:
        return {
            'found': True,
            'gene': gene_upper,
            **gene_database[gene_upper]
        }
    else:
        # Not in our database
        return {
            'found': False,
            'gene': gene_upper,
            'description': f"Information about {gene_upper} is not available in our local knowledge base. This gene may be less commonly studied or have specialized functions."
        }


def fetch_from_mygene_api(gene_name: str) -> dict:
    """
    Optional: Fetch gene info from MyGene.info API.

    This is a bonus feature David could add if time permits.

    Args:
        gene_name: Gene symbol

    Returns:
        dict with gene info from API

    Example:
        >>> info = fetch_from_mygene_api("TP53")
    """

    url = "https://mygene.info/v3/query"
    params = {
        "q": gene_name,
        "species": "human",
        "size": 1,
        "fields": "symbol,name,summary,alias,chromosome,map_location,entrezgene,ensembl,uniprot"
    }

    try:
        resp = requests.get(url, params=params, timeout=5)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        return {
            "found": False,
            "gene": gene_name.upper(),
            "description": f"Error fetching info from MyGene.info: {e}"
        }

    hits = data.get("hits", []) if isinstance(data, dict) else []
    if not hits:
        return {
            "found": False,
            "gene": gene_name.upper(),
            "description": f"No information found for {gene_name} via MyGene.info."
        }

    hit = hits[0]
    ensembl_id = None
    ensembl_field = hit.get("ensembl")
    if isinstance(ensembl_field, dict):
        ensembl_id = ensembl_field.get("gene")
    elif isinstance(ensembl_field, list) and ensembl_field:
        ensembl_id = ensembl_field[0].get("gene") if isinstance(ensembl_field[0], dict) else None

    description = hit.get("summary") or hit.get("name") or "No description available."

    return {
        "found": True,
        "gene": (hit.get("symbol") or gene_name).upper(),
        "full_name": hit.get("name"),
        "description": description,
        "aliases": hit.get("alias", []),
        "chromosome": hit.get("chromosome") or hit.get("map_location") or "N/A",
        "entrez_id": hit.get("entrezgene"),
        "ensembl_id": ensembl_id,
        "raw": hit,
    }


# ============================================
# RAG: Generate Natural Language Answer
# ============================================

def generate_gene_explanation(gene_name: str, gene_info: dict, user_question: str) -> str:
    """
    Use LLM to generate natural language explanation from retrieved info.

    This is the "Generation" part of RAG (Retrieval-Augmented Generation).

    Args:
        gene_name: Gene symbol
        gene_info: Retrieved gene information dict
        user_question: Original user question

    Returns:
        str: Natural language explanation

    Example:
        >>> explanation = generate_gene_explanation("TP53", tp53_info, "What does TP53 do?")
        >>> print(explanation)
        "TP53 is a crucial tumor suppressor gene that acts as the 'guardian of the genome'..."

    TODO for David:
    1. Build a prompt that includes the retrieved information
    2. Ask LLM to explain it in accessible language
    3. Make it answer the specific question asked
    4. Use temperature ~0.3-0.5 for more natural explanations
    """
    if DEBUG:
        print(not gene_info.get('found'))
    if not gene_info.get('found'):
        # Gene not in knowledge base
        prompt = f"""
The gene {gene_name} is not in our knowledge base.

Based on your general knowledge, provide a brief explanation of what this gene might do,
or explain that detailed information is not available.

User's question: {user_question}

Answer:
"""
        answer = call_ollama(prompt, temperature=0.5)
        return answer if answer else f"Sorry, I don't have information about {gene_name}."

    # Gene found - inject retrieved info into prompt
    context = f"""
Gene: {gene_info['gene']}
Full Name: {gene_info.get('full_name', 'N/A')}
Location: {gene_info.get('chromosome', 'N/A')}
Function: {gene_info.get('function', 'N/A')}
Description: {gene_info.get('description', '')}
"""

    prompt = f"""
You are a helpful bioinformatics assistant. Answer the user's question about this gene
using the information provided. Explain in clear, accessible language suitable for researchers.

Gene Information:
{context}

User's question: {user_question}

Your answer (2-3 sentences):
"""

    answer = call_ollama(prompt, temperature=0.4)

    return answer if answer else gene_info.get('description', 'No information available.')


# ============================================
# Main Handler
# ============================================

def answer_gene_question(user_input: str) -> str:
    """
    Main entry point for gene information questions.
    Orchestrates: extraction → retrieval → generation

    Args:
        user_input: User's question

    Returns:
        str: Natural language answer

    Example:
        >>> answer = answer_gene_question("What does TP53 do?")
        >>> print(answer)
        "TP53 is a tumor suppressor gene..."
    """

    # Step 1: Extract gene name from question
    gene_name = extract_gene_from_question(user_input)

    if gene_name is None:
        return "I couldn't identify which gene you're asking about. Please mention a specific gene name."

    # Step 2: Retrieve gene information (RAG retrieval step)
    # gene_info = retrieve_gene_annotation(gene_name)
    gene_info = fetch_from_mygene_api(gene_name)
    if DEBUG:
        print(gene_info)

    # Step 3: Generate answer using LLM (RAG generation step)
    answer = generate_gene_explanation(gene_name, gene_info, user_input)

    return answer


def extract_gene_from_question(user_input: str) -> str:
    """
    Extract gene name from a question.

    Args:
        user_input: User's question

    Returns:
        str: Gene name (uppercase) or None

    Example:
        >>> extract_gene_from_question("What does TP53 do?")
        "TP53"
    """

    # Use the utility function from ollama_utils
    gene = ask_llm_to_extract(
        user_input,
        what_to_extract="gene name",
        example="TP53"
    )

    if gene and gene != "NONE":
        return gene.upper().strip()

    return None


# ============================================
# Testing / Development
# ============================================

if __name__ == "__main__":
    """
    Test this module independently.

    Run: python modules/python/llm_rag.py
    """

    print("=== Testing RAG Module ===\n")

    # Test intent detection
    print("1. Intent Detection:")
    test_queries = [
        "What does TP53 do?",
        "Show me TP53",
        "Tell me about BRCA1",
        "Plot EGFR expression"
    ]
    for query in test_queries:
        is_info_q = is_gene_question(query)
        print(f"   '{query}' → Info question: {is_info_q}")
    print()

    # Test retrieval
    print("2. Knowledge Retrieval:")
    gene = "TP53"
    info = retrieve_gene_annotation(gene)
    print(f"   Gene: {gene}")
    print(f"   Found: {info['found']}")
    if info['found']:
        print(f"   Description: {info['description'][:100]}...")
    print()

    # Test full RAG pipeline
    print("3. Full RAG Pipeline:")
    question = "What does TP53 do?"
    answer = answer_gene_question(question)
    print(f"   Question: {question}")
    print(f"   Answer: {answer}")
    print()

    print("\n=== Tips for David ===")
    print("1. Expand the gene_database with more genes")
    print("2. Consider loading gene info from a CSV file")
    print("3. Optional: Integrate with MyGene.info API for real-time data")
    print("4. Experiment with different prompt styles for better explanations")
    print("5. Think about: How to handle multi-gene questions?")
    print("6. Bonus: Add gene-gene interaction information")
