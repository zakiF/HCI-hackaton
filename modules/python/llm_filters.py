"""
Natural Language Data Filtering Module
Team Member: Qing Li

LLM Learning Goals:
1. Parameter extraction: Pull out filter conditions from natural language
2. Structured output: Get LLM to return JSON with filter parameters
3. Intent classification: Detect when user wants filtering

Examples:
- "Show me TP53 in only tumor samples" → filter: condition == "Primary Tumor"
- "Display BRCA1 in normal tissue" → filter: condition == "Normal"
- "Plot EGFR excluding metastatic" → filter: condition != "Metastatic"
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from utils.python.ollama_utils import call_ollama, call_ollama_json


# ============================================
# Main Functions (Qing implements these)
# ============================================

def extract_filters(user_input: str) -> dict:
    """
    Use LLM to extract filtering criteria from user's query.

    Args:
        user_input: User's natural language query

    Returns:
        dict with keys:
            - has_filter: bool (True if filtering detected)
            - condition_filter: str or None (e.g., "Primary Tumor", "Normal")
            - exclude: bool (True if excluding, False if including)
            - description: str (human-readable filter description)

    Example:
        >>> extract_filters("Show TP53 in tumor samples only")
        {
            'has_filter': True,
            'condition_filter': 'Primary Tumor',
            'exclude': False,
            'description': 'only tumor samples'
        }

    TODO for Qing:
    1. Write an LLM prompt that extracts filter conditions
    2. Use call_ollama_json() to get structured response
    3. Map natural language conditions to dataset values:
       - "tumor" → "Primary Tumor"
       - "normal" → "Normal"
       - "metastatic" → "Metastatic"
    4. Handle negations (exclude vs include)
    """

    # STEP 1: Check if query contains filtering intent
    # TODO: Use LLM to detect if user wants filtering

    # For now, simple keyword detection (Qing will replace with LLM)
    filter_keywords = ['only', 'in', 'excluding', 'without', 'just']
    has_filter = any(keyword in user_input.lower() for keyword in filter_keywords)

    if not has_filter:
        return {
            'has_filter': False,
            'condition_filter': None,
            'exclude': False,
            'description': 'no filter'
        }

    # STEP 2: Extract filter parameters using LLM
    # TODO: Qing implements LLM extraction here

    prompt = f"""
Extract filtering information from this query.

The dataset has these conditions: "Normal", "Primary Tumor", "Metastatic"

Return JSON with these keys:
- condition: which condition to filter for (or null if not condition-based)
- exclude: true if excluding, false if including
- description: brief description of the filter

Query: {user_input}

JSON:
"""

    # Call LLM (Qing: test this prompt and refine it!)
    llm_result = call_ollama_json(prompt, temperature=0.1)

    if llm_result is None:
        # Fallback if LLM fails
        return {
            'has_filter': False,
            'condition_filter': None,
            'exclude': False,
            'description': 'LLM extraction failed'
        }

    # STEP 3: Map LLM response to our format
    return {
        'has_filter': True,
        'condition_filter': llm_result.get('condition'),
        'exclude': llm_result.get('exclude', False),
        'description': llm_result.get('description', 'custom filter')
    }


def apply_filters(expr_data: dict, filter_info: dict) -> dict:
    """
    Apply the extracted filters to the expression data.

    Args:
        expr_data: Expression data dict from load_expression_data()
        filter_info: Filter dict from extract_filters()

    Returns:
        dict: Filtered expression data (same structure as input)

    Example:
        >>> filter_info = {'condition_filter': 'Primary Tumor', 'exclude': False}
        >>> filtered = apply_filters(expr_data, filter_info)
        # filtered data now contains only tumor samples

    TODO for Qing:
    1. Access metadata: expr_data['metadata']
    2. Filter samples based on condition
    3. Subset expression matrix to keep only filtered samples
    4. Return new dict with filtered data
    """

    if not filter_info.get('has_filter'):
        return expr_data  # No filtering needed

    # Get metadata
    metadata = expr_data['metadata']
    expression = expr_data['expression']

    # Apply condition filter
    condition_filter = filter_info.get('condition_filter')

    if condition_filter:
        # Filter samples
        if filter_info.get('exclude'):
            # Exclude this condition
            keep_samples = metadata[metadata['condition'] != condition_filter].index
        else:
            # Include only this condition
            keep_samples = metadata[metadata['condition'] == condition_filter].index

        # Subset data
        filtered_metadata = metadata.loc[keep_samples]
        filtered_expression = expression[keep_samples]

        return {
            'expression': filtered_expression,
            'metadata': filtered_metadata
        }

    return expr_data


# ============================================
# Testing / Development
# ============================================

if __name__ == "__main__":
    """
    Test this module independently.

    Run: python modules/python/llm_filters.py
    """

    print("=== Testing Filter Extraction ===\n")

    # Test cases
    test_queries = [
        "Show me TP53",
        "Show me TP53 in tumor samples only",
        "Display BRCA1 in normal tissue",
        "Plot EGFR excluding metastatic",
        "Compare TP53 in normal vs tumor"
    ]

    for query in test_queries:
        print(f"Query: {query}")
        result = extract_filters(query)
        print(f"Result: {result}")
        print()

    print("\n=== Tips for Qing ===")
    print("1. Test different phrasings of filters")
    print("2. Iterate on the LLM prompt to improve accuracy")
    print("3. Handle edge cases (typos, ambiguous conditions)")
    print("4. Consider using temperature=0.0 for more consistent extraction")
