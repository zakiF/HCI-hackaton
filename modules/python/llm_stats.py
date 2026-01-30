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
from unittest import result
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from utils.python.ollama_utils import call_ollama, call_ollama_json
import pandas as pd
from scipy import stats
from utils.python.plot_utils import load_expression_data


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
    #check user input for errors
    if not isinstance(user_input, str) or len(user_input.strip()) == 0:
        return False                                        

    # 1. Clean up user input
    # clean up white space and convert to lower case
    cleaned_input = user_input.strip().lower()

    # 2. Try Heuristic approach with keywords
    heuristic_results = heuristic_stats_detection(cleaned_input)
    if heuristic_results is True:
        return True
    
    # else continue to next step

    # 3. Fall back on LLM if hueristic approach is inconslusive
    llm_stats_detection_result = llm_stats_detection(cleaned_input)
    if llm_stats_detection_result is False or llm_stats_detection_result is True:
        # validate output evenetually 
        return llm_stats_detection_result

    return False  # safe default
    

def heuristic_stats_detection(user_input: str) -> bool:
    """
    Heuristic detection of statistical questions using keyword and phrase matching.

    Args:
        user_input: Cleaned, lowercase user query

    Returns:
        bool: True if this is likely a statistical question

    Examples:
        >>> heuristic_stats_detection("is tp53 significant?")
        True
        >>> heuristic_stats_detection("does tp53 differ between groups?")
        True
        >>> heuristic_stats_detection("plot tp53 expression")
        False
        >>> heuristic_stats_detection("show me tp53")
        False
    """

    text = user_input.lower()

        # ---- strong non-stats overrides ----
    NON_STATS_PATTERNS = [
        "list ",
        "show ",
        "display ",
        "genes expressed",
        "proteins expressed",
    ]

    if any(p in text for p in NON_STATS_PATTERNS):
        return False

    # --- Keyword groups ---
    INFERENCE_KEYWORDS = [
        "significant", "significance",
        "statistical", "statistically",
        "p-value", "p value", "pvalue",
        "confidence interval", "effect size",
        "padj", "fdr",
    ]

    COMPARISON_KEYWORDS = [
        "compare", "comparison",
        "different", "difference", "differs",
        "higher", "lower",
        "increased", "decreased",
        "upregulated", "downregulated",
        "enriched", "depleted",
    ]

    TEST_KEYWORDS = [
        "test", "testing",
        "t-test", "ttest",
        "anova", "manova",
        "wilcoxon", "mann-whitney",
        "chi-square", "fisher",
        "regression", "linear model",
        "correlation", "pearson", "spearman",
    ]

    OMICS_STATS_KEYWORDS = [
        "differential expression",
        "differentially expressed",
        "deg", "degs",
        "fold change", "log2fc",
        "gene set enrichment", "gsea",
        "enrichment analysis",
    ]

    ENRICHMENT_CONTEXT = [
        "enriched after",
        "enriched in",
        "enriched between",
        "enriched compared to",
    ]


    COMPARATIVE_PATTERNS = [
        "higher than", "lower than",
        "more than", "less than",
        "between groups",
        "across conditions",
        "compared to",
        "versus",
    ]

    # using scoring logic to dtermine if its a stats question
    score = 0

    def contains_any(keywords):
        return any(k in text for k in keywords)

    if contains_any(INFERENCE_KEYWORDS):
        score += 2

    if contains_any(TEST_KEYWORDS):
        score += 2

    if contains_any(OMICS_STATS_KEYWORDS):
        score += 2

    if contains_any(COMPARISON_KEYWORDS):
        score += 1

    if contains_any(COMPARATIVE_PATTERNS):
        score += 1

    if contains_any(ENRICHMENT_CONTEXT):
        score += 2

    # Threshold chosen to reduce false positives
    return score >= 2



def llm_stats_detection(user_input: str) -> bool:

    """
    LLM detection of statistical questions using prompt.
    
    Args:
        user_input: User's query
    Returns:
        bool: True if this is a stats question
    Example:
        >>> llm_stats_detection("Is TP53 significant?")
        True
        >>> llm_stats_detection("Show me TP53")
        False  
    
    """

    prompt = f"""
    You are a fallback intent classifier for a bioinformatics query router.

    IMPORTANT CONTEXT:
    - Obvious statistical keywords have ALREADY been checked and were inconclusive.
    - Your task is to resolve ambiguity for subtle biological phrasing.

    Classify the user query into EXACTLY ONE of:
    - stats
    - plot
    - question

    STRICT RULES:

    Return "stats" ONLY if the query implies statistical inference or hypothesis testing,
    even if no explicit statistical terms are used.

    This includes:
    - asking whether something differs between groups or conditions
    - asking if expression is higher/lower in one group
    - asking about enrichment, differential expression, or changes after treatment

    Return "plot" ONLY if the user is asking to visualize or graph data.

    Return "question" if the query is descriptive, a lookup, or asks to list/show values
    WITHOUT implying comparison or testing.

    DECISION GUIDELINES:
    - "Does X differ between groups?" → stats
    - "Are any proteins enriched after treatment?" → stats
    - "Which genes are differentially expressed?" → stats
    - "List genes expressed in this sample" → question
    - "Show TP53 expression" → question
    - "Plot TP53 expression" → plot

    User query:
    "{user_input}"

    Respond with ONLY ONE WORD: stats, plot, or question.
    Do NOT explain your reasoning.
    """


    #call ollama to classify
    result = call_ollama(prompt, temperature=0.0)

    if result is None:
        return False

    #strip and lower to standadize output    
    result = result.strip().lower()

    if result == "stats":
        return True
    elif result in {"plot", "question"}:
        return False
    else:
        return False  # safe default              

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
You are a parameter extraction assistant for a bioinformatics application.

Your task is to extract statistical testing parameters from a user query.
The query refers to gene expression comparisons.

AVAILABLE DATASET CONDITIONS (use EXACT spelling only):
- Normal
- Primary Tumor
- Metastatic

NATURAL LANGUAGE → DATASET VALUE MAPPING:
- "normal" → "Normal"
- "tumor" or "primary tumor" → "Primary Tumor"
- "metastatic" or "metastasis" → "Metastatic"

INSTRUCTIONS:
- Identify the gene being referenced (return uppercase gene symbol).
- Identify which dataset conditions are being compared.
- Map any natural language condition terms to the dataset values using the mapping above.
- Do NOT invent or infer conditions outside the provided list.
- Infer the appropriate statistical test:
  - Use "t-test" if exactly 2 groups are compared.
  - Use "anova" if 3 or more groups are compared.
- Do NOT guess a gene if none is mentioned.

OUTPUT FORMAT:
Return a single valid JSON object with EXACTLY these keys:
- gene: string or None
- groups: array of strings (subset of the dataset conditions)
- test: "t-test", "anova", or None

IMPORTANT RULES:
- If the gene is not specified, return gene: None .
- If fewer than 2 valid groups can be identified, return groups: [].
- If the test cannot be determined, return test: None.
- Use ONLY the dataset condition names exactly as listed.
- Do NOT include any text outside the JSON.
- Do NOT explain your reasoning.

User query:
"{user_input}"

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
    
    gene = (result.get("gene") or "NONE").upper()
    groups = result.get("groups") or []
    test_type = result.get("test")  # keep None if missing

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

    expression = expr_data['expr_matrix']
    metadata = expr_data['metadata']

    # Validate gene exists
    if gene_name not in expression.index:
        raise ValueError(f"Gene '{gene_name}' not found in expression data")

    # Validate groups exist in metadata
    available_conditions = set(metadata['condition'].unique())
    if group1 not in available_conditions:
        raise ValueError(f"Condition '{group1}' not found in metadata")
    if group2 not in available_conditions:
        raise ValueError(f"Condition '{group2}' not found in metadata")

    # Get samples for each group
    group1_samples = metadata[metadata['condition'] == group1].Run
    group2_samples = metadata[metadata['condition'] == group2].Run

    if len(group1_samples) == 0:
        raise ValueError(f"No samples found for condition '{group1}'")
    if len(group2_samples) == 0:
        raise ValueError(f"No samples found for condition '{group2}'")

    # Get expression values
    group1_expr = expression.loc[gene_name, group1_samples]
    group2_expr = expression.loc[gene_name, group2_samples]

    # check the data type expression values
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

    expression = expr_data['expr_matrix']
    metadata = expr_data['metadata']

    # Validate gene exists
    if gene_name not in expression.index:
        raise ValueError(f"Gene '{gene_name}' not found in expression data")

    # Validate groups exist and collect data
    available_conditions = set(metadata['condition'].unique())
    group_data = []
    group_means = {}

    for group in groups:
        if group not in available_conditions:
            raise ValueError(f"Condition '{group}' not found in metadata")

        samples = metadata[metadata['condition'] == group].Run
        if len(samples) == 0:
            raise ValueError(f"No samples found for condition '{group}'")

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
You are a scientific results summarization assistant.

Your task is to explain the statistical test results below in clear, concise language.
You MUST answer the user's original question directly.

STRICT INSTRUCTIONS:
- Write 1–2 sentences only.
- Do NOT speculate or add interpretation beyond the results provided.
- Use the p-value and significance flag exactly as given.
- If the result is significant, clearly state that the gene shows a statistically significant difference.
- If the result is not significant, clearly state that no statistically significant difference was detected.
- Mention the groups compared and the direction of the effect when means are provided.
- Do NOT mention test assumptions, methodology, or confidence intervals unless explicitly provided.

Statistical Results (authoritative):
{stats_context}

User Question:
"{user_question}"

Write a direct answer to the user's question:
"""

    summary = call_ollama(prompt, temperature=0.3)

    return summary if summary else f"Statistical test completed. P-value: {pvalue:.4f}"

def run_stats_tests(user_input: str, expr_data: dict, params: dict) -> dict:
    """
    Main entry point for statistical queries.
    Orchestrates: extraction → test execution → summary generation
    Args:
        user_input: User's question
        expr_data: Expression data dict
        params: Extracted parameters dict       
    Returns:
        dict with 'results' and 'summary' keys
    """

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

    if params['gene'] is None or params['test'] is None or params['gene'] == 'NONE'or len(params['groups']) < 2:
        return {
            'results': None,
            'summary': "I couldn't extract enough information for a statistical test. Please specify a gene, test type, and at least two conditions to compare."
        }
    print(params['gene'].split(','))
    # make this work for multiple genes
    if len(params['gene'].split(',')) > 1:
        gene_list = params['gene'].split(',')
        all_results = {}
        all_summaries = {}
        for gene in gene_list:
            params_single = params.copy()
            params_single['gene'] = gene.strip().upper()
            result = run_stats_tests(user_input, expr_data, params_single)
            all_results[gene.strip().upper()] = result['results']
            all_summaries[gene.strip().upper()] = result['summary']
        return {
            'results': all_results,
            'summary': all_summaries
        }
    
    else:
        # Step 2 & 3: Run test and generate summary
        result = run_stats_tests(user_input, expr_data, params)
        return result



# ============================================
# Testing / Development
# ============================================

# if __name__ == "__main__":

    #load test expression data

    # data = load_expression_data()
    # expr_matrix = data['expr_matrix']

    # test = handle_stats_query(
    #     "Are TP53 or BRAC1 significantly different between tumor and normal?",
    #     data
    # )

    # print(test['results'])
    # print(test['summary'])
    
    # print("Running intent detection tests...\n")

    # test_cases = [
    #     # ---- stats queries ----
    #     ("Is TP53 significant?", True),
    #     ("Compare BRCA1 between groups", True),
    #     ("Does TP53 differ between treatment groups?", True),
    #     ("Which genes are differentially expressed?", True),
    #     ("Are any proteins enriched after treatment?", True),

    #     # ---- non-stats queries ----
    #     ("Show me TP53", False),
    #     ("Plot TP53 expression", False),
    #     ("Visualize RNA-seq results", False),
    #     ("What does TP53 do?", False),
    #     ("List genes expressed in this sample", False),
    # ]

    # failures = 0

    # for query, expected in test_cases:
    #     result = is_stats_query(query)
    #     status = "PASS" if result == expected else "FAIL"

    #     print(f"[{status}] '{query}' → {result}")

    #     if status == "FAIL":
    #         print(f"       expected: {expected}")
    #         failures += 1

    # print("\nSummary:")
    # if failures == 0:
    #     print("✅ All intent detection tests passed!")
    # else:
    #     print(f"❌ {failures} test(s) failed")


    # print("2. Parameter Extraction Tests:\n")

    # test_cases = [
    #     {
    #         "query": "Is TP53 significantly different between tumor and normal?",
    #         "expected": {
    #             "gene": "TP53",
    #             "test": "t-test",
    #             "groups": ["Normal", "Primary Tumor"]
    #         },
    #     },
    #     {
    #         "query": "Compare BRCA1 across all conditions",
    #         "expected": {
    #             "gene": "BRCA1",
    #             "test": "anova",
    #             "groups": ["Normal", "Primary Tumor", "Metastatic"]
    #         },
    #     },
    #     {
    #         "query": "Is MYC different between metastatic and normal samples?",
    #         "expected": {
    #             "gene": "MYC",
    #             "test": "t-test",
    #             "groups": ["Normal", "Metastatic"]
    #         },
    #     },
    #     {
    #         "query": "Are any genes differentially expressed?",
    #         "expected": {
    #             "gene": 'NONE',
    #             "test": None,
    #             "groups": []
    #         },
    #     },
    #     {
    #         "query": "Compare expression between normal and primary tumor",
    #         "expected": {
    #             "gene": 'NONE',
    #             "test": "t-test",
    #             "groups": ["Normal", "Primary Tumor"]
    #         },
    #     },
    # ]

    # failures = 0

    # for case in test_cases:
    #     query = case["query"]
    #     expected = case["expected"]

    #     result = extract_stats_parameters(query)

    #     print(f"Query: {query}")
    #     print(f"Result:   {result}")
    #     print(f"Expected: {expected}")

    #     # --- robust checks ---
    #     if result.get("gene") != expected["gene"]:
    #         print("❌ gene mismatch")
    #         failures += 1

    #     if set(result.get("groups", [])) != set(expected["groups"]):
    #         print("❌ groups mismatch")
    #         failures += 1

    #     if result.get("test") != expected["test"]:
    #         print("❌ test mismatch")
    #         failures += 1

    #     if failures == 0:
    #         print("✅ PASS\n")
    #     else:
    #         print("❌ FAIL\n")

    # print("Summary:")
    # if failures == 0:
    #     print("✅ All parameter extraction tests passed!")
    # else:
    #     print(f"❌ {failures} total failure(s)")


    # print("\n=== Tips for Tayler ===")
    # print("1. Test with different phrasings of statistical questions")
    # print("2. Handle ambiguous cases (e.g., 'compare all conditions')")
    # print("3. Consider adding correlation analysis as another test type")
    # print("4. Think about: Should we support multiple testing correction?")
    # print("5. Bonus: Add visualization of test results (e.g., p-value bars)")

    # # Simple handler tests: exercise `handle_stats_query` for a few queries
    # print("\n3. Handler tests:\n")

    # handler_tests = [
    #     {
    #         'query': "Is TP53 significantly different between tumor and normal?",
    #         'expect_results': True
    #     },
    #     {
    #         'query': "Compare BRCA1 across all conditions",
    #         'expect_results': True
    #     },
    #     {
    #         'query': "Is a different between tumor and normal?",
    #         'expect_results': False
    #     }
    # ]

    # for case in handler_tests:
    #     q = case['query']
    #     out = handle_stats_query(q, data)
    #     has_results = out.get('results') is not None
    #     status = "PASS" if has_results == case['expect_results'] else "FAIL"
    #     print(f"[{status}] {q} -> results present: {has_results}")
    #     print("Summary:")
    #     print(out.get('summary'))

    # # Direct function validation tests
    # print("\n4. Direct validation tests:\n")

    # try:
    #     run_ttest("FAKEGENE", data, "Normal", "Primary Tumor")
    #     print("[FAIL] run_ttest did not raise for missing gene")
    # except ValueError as e:
    #     print("[PASS] run_ttest missing gene:", e)

    # try:
    #     run_anova("TP53", data, ["Normal", "NoSuchCondition"]) 
    #     print("[FAIL] run_anova did not raise for missing condition")
    # except ValueError as e:
    #     print("[PASS] run_anova missing condition:", e)