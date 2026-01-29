"""
Statistical Testing Module
Team Member: Tayler Fearn

LLM Learning Goals:
1. Intent classification: Detect statistical questions vs plotting
2. Parameter extraction: Pull gene, test type, groups from natural language
3. Natural language generation: Explain stats results in plain English
4. Structured output: Get LLM to return JSON with test parameters

Examples:
- "Is TP53 significantly different between tumor and normal?"
  → Run t-test between Primary Tumor and Normal
- "Compare BRCA1 across all conditions"
  → Run ANOVA across all 3 groups
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from utils.python.ollama_utils import call_ollama, call_ollama_json
import pandas as pd
from scipy import stats


# ============================================
# Intent Detection
# ============================================

def is_stats_query(user_input: str) -> bool:
    """
    Detect if user is asking a statistical question.

    Args:
        user_input: User's query

    Returns:
        bool: True if this is a stats question

    Example:
        >>> is_stats_query("Is TP53 significant?")
        True
        >>> is_stats_query("Show me TP53")
        False

    TODO for Tayler:
    1. Use LLM to classify intent as "stats" vs "plot" vs "question"
    2. Or use keyword detection for speed
    3. Consider phrases like: "significant", "different", "compare", "test"
    """

    # Simple keyword detection (Tayler can enhance with LLM)
    stats_keywords = [
        'significant', 'significance', 'different', 'difference',
        'compare', 'comparison', 'test', 'p-value', 'pvalue',
        'statistical', 'statistics', 'anova', 't-test', 'ttest'
    ]

    user_lower = user_input.lower()

    return any(keyword in user_lower for keyword in stats_keywords)


# ============================================
# Parameter Extraction
# ============================================

def extract_stats_parameters(user_input: str) -> dict:
    """
    Use LLM to extract statistical test parameters.

    Args:
        user_input: User's query

    Returns:
        dict with keys:
            - gene: str (gene name)
            - test: str ("t-test" or "anova")
            - groups: list of str (conditions to compare)

    Example:
        >>> extract_stats_parameters("Is TP53 different between tumor and normal?")
        {
            'gene': 'TP53',
            'test': 't-test',
            'groups': ['Primary Tumor', 'Normal']
        }

    TODO for Tayler:
    1. Write LLM prompt to extract gene, test type, groups
    2. Map natural language to dataset values:
       - "tumor" → "Primary Tumor"
       - "normal" → "Normal"
       - "metastatic" → "Metastatic"
    3. Decide test type based on number of groups:
       - 2 groups → t-test
       - 3+ groups → ANOVA
    4. Use call_ollama_json() for structured output
    """

    prompt = f"""
Extract statistical test information from this query.

Dataset conditions: "Normal", "Primary Tumor", "Metastatic"

Return JSON with these keys:
- gene: gene name (uppercase)
- groups: list of conditions to compare (use exact condition names from above)
- test: "t-test" (for 2 groups) or "anova" (for 3+ groups)

Query: {user_input}

JSON:
"""

    result = call_ollama_json(prompt, temperature=0.1)

    if result is None:
        # Fallback
        return {
            'gene': None,
            'test': None,
            'groups': []
        }

    # Validate and clean
    gene = result.get('gene', '').upper()
    groups = result.get('groups', [])
    test_type = result.get('test', 't-test')

    # Auto-determine test type based on groups if needed
    if len(groups) == 2:
        test_type = 't-test'
    elif len(groups) >= 3:
        test_type = 'anova'

    return {
        'gene': gene,
        'test': test_type,
        'groups': groups
    }


# ============================================
# Statistical Test Functions (Hardcoded)
# ============================================

def run_ttest(gene_name: str, expr_data: dict, group1: str, group2: str) -> dict:
    """
    Run independent samples t-test.

    Args:
        gene_name: Gene to test
        expr_data: Expression data dict
        group1: First condition (e.g., "Normal")
        group2: Second condition (e.g., "Primary Tumor")

    Returns:
        dict with test results

    Example:
        >>> result = run_ttest("TP53", expr_data, "Normal", "Primary Tumor")
        >>> print(result['pvalue'])
        0.0234
    """

    expression = expr_data['expression']
    metadata = expr_data['metadata']

    # Get samples for each group
    group1_samples = metadata[metadata['condition'] == group1].index
    group2_samples = metadata[metadata['condition'] == group2].index

    # Get expression values
    group1_expr = expression.loc[gene_name, group1_samples]
    group2_expr = expression.loc[gene_name, group2_samples]

    # Run t-test
    statistic, pvalue = stats.ttest_ind(group1_expr, group2_expr)

    return {
        'test': 't-test',
        'gene': gene_name,
        'group1': group1,
        'group2': group2,
        'statistic': float(statistic),
        'pvalue': float(pvalue),
        'significant': pvalue < 0.05,
        'mean_group1': float(group1_expr.mean()),
        'mean_group2': float(group2_expr.mean()),
        'n_group1': len(group1_expr),
        'n_group2': len(group2_expr)
    }


def run_anova(gene_name: str, expr_data: dict, groups: list) -> dict:
    """
    Run one-way ANOVA across multiple groups.

    Args:
        gene_name: Gene to test
        expr_data: Expression data dict
        groups: List of conditions to compare

    Returns:
        dict with test results

    Example:
        >>> result = run_anova("TP53", expr_data, ["Normal", "Primary Tumor", "Metastatic"])
        >>> print(result['pvalue'])
        0.0012
    """

    expression = expr_data['expression']
    metadata = expr_data['metadata']

    # Get expression for each group
    group_data = []
    group_means = {}

    for group in groups:
        samples = metadata[metadata['condition'] == group].index
        group_expr = expression.loc[gene_name, samples]
        group_data.append(group_expr)
        group_means[group] = float(group_expr.mean())

    # Run ANOVA
    statistic, pvalue = stats.f_oneway(*group_data)

    return {
        'test': 'ANOVA',
        'gene': gene_name,
        'groups': groups,
        'statistic': float(statistic),
        'pvalue': float(pvalue),
        'significant': pvalue < 0.05,
        'group_means': group_means
    }


# ============================================
# Natural Language Summary
# ============================================

def generate_stats_summary(stats_results: dict, user_question: str) -> str:
    """
    Use LLM to generate natural language explanation of results.

    Args:
        stats_results: Dict from run_ttest() or run_anova()
        user_question: Original user question

    Returns:
        str: Natural language summary

    Example:
        >>> summary = generate_stats_summary(results, "Is TP53 significant?")
        >>> print(summary)
        "Yes, TP53 shows significantly different expression between Primary Tumor
         and Normal samples (p=0.023), with tumor samples showing higher expression."

    TODO for Tayler:
    1. Write LLM prompt that takes stats results
    2. Ask LLM to explain in 1-2 sentences
    3. Make it answer the user's original question
    4. Use temperature=0.3 for more natural language
    """

    # Build context from results
    gene = stats_results['gene']
    test = stats_results['test']
    pvalue = stats_results['pvalue']
    significant = stats_results['significant']

    if test == 't-test':
        group1 = stats_results['group1']
        group2 = stats_results['group2']
        mean1 = stats_results['mean_group1']
        mean2 = stats_results['mean_group2']

        stats_context = f"""
Gene: {gene}
Test: {test}
Groups compared: {group1} (mean={mean1:.2f}) vs {group2} (mean={mean2:.2f})
P-value: {pvalue:.4f}
Significant (p<0.05): {significant}
"""
    else:  # ANOVA
        groups = stats_results['groups']
        means = stats_results['group_means']

        stats_context = f"""
Gene: {gene}
Test: {test}
Groups compared: {', '.join(groups)}
Group means: {means}
P-value: {pvalue:.4f}
Significant (p<0.05): {significant}
"""

    prompt = f"""
Explain these statistical test results in 1-2 clear sentences.
Answer the user's question directly.

Statistical Results:
{stats_context}

User's question: {user_question}

Explanation:
"""

    summary = call_ollama(prompt, temperature=0.3)

    return summary if summary else f"Statistical test completed. P-value: {pvalue:.4f}"


# ============================================
# Main Handler
# ============================================

def handle_stats_query(user_input: str, expr_data: dict) -> dict:
    """
    Main entry point for statistical queries.
    Orchestrates: extraction → test execution → summary generation

    Args:
        user_input: User's question
        expr_data: Expression data dict

    Returns:
        dict with 'results' and 'summary' keys
    """

    # Step 1: Extract parameters
    params = extract_stats_parameters(user_input)

    if params['gene'] is None or len(params['groups']) < 2:
        return {
            'results': None,
            'summary': "I couldn't extract enough information for a statistical test. Please specify a gene and conditions to compare."
        }

    # Step 2: Run appropriate test
    if params['test'] == 't-test' and len(params['groups']) == 2:
        results = run_ttest(
            params['gene'],
            expr_data,
            params['groups'][0],
            params['groups'][1]
        )
    else:
        results = run_anova(
            params['gene'],
            expr_data,
            params['groups']
        )

    # Step 3: Generate summary
    summary = generate_stats_summary(results, user_input)

    return {
        'results': results,
        'summary': summary
    }


# ============================================
# Testing / Development
# ============================================

if __name__ == "__main__":
    """
    Test this module independently.

    Run: python modules/python/llm_stats.py
    """

    print("=== Testing Stats Module ===\n")

    # Test intent detection
    print("1. Intent Detection:")
    test_queries = [
        "Show me TP53",
        "Is TP53 significant?",
        "Compare BRCA1 between groups"
    ]
    for query in test_queries:
        is_stats = is_stats_query(query)
        print(f"   '{query}' → Stats query: {is_stats}")
    print()

    # Test parameter extraction
    print("2. Parameter Extraction:")
    query = "Is TP53 significantly different between tumor and normal?"
    params = extract_stats_parameters(query)
    print(f"   Query: {query}")
    print(f"   Extracted: {params}")
    print()

    print("\n=== Tips for Tayler ===")
    print("1. Test with different phrasings of statistical questions")
    print("2. Handle ambiguous cases (e.g., 'compare all conditions')")
    print("3. Consider adding correlation analysis as another test type")
    print("4. Think about: Should we support multiple testing correction?")
    print("5. Bonus: Add visualization of test results (e.g., p-value bars)")
