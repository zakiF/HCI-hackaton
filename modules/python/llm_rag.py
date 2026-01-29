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
from datetime import datetime
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from utils.python.ollama_utils import call_ollama, ask_llm_to_classify, ask_llm_to_extract
from utils.python.llm_utils import extract_gene_name_with_details
import pandas as pd
import requests
import re

DEBUG = False
TEST_DATA_PATH = os.path.join(os.path.dirname(__file__), "tests", "gene_query_tests.csv")
TEST_OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "tests", "results")

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
    """

    # Prefer LLM classification for better intent detection; fall back to keyword heuristics on failure
    category = ask_llm_to_classify(
        user_input,
        options=["info", "plot"],
        context=(
            "Classify the intent of this gene-related query. "
            "Return only 'info' if the user is asking for gene information/description. "
            "Return only 'plot' if the user is asking to visualize/plot expression. "
            "Return only 'stats' if the user is asking for statistical analysis."
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

def fetch_from_mygene_api(gene_name: str) -> dict:
    """
    Fetch gene info from MyGene.info API.

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

def generate_gene_explanations(gene_names: list[str], gene_infos: list[dict], user_question: str) -> str:
    """
    Use LLM to generate natural language explanations for one or more genes in a single call.

    Args:
        gene_names: List of gene symbols (uppercase)
        gene_infos: List of gene info dicts from retrieval, same order as gene_names
        user_question: Original user question

    Returns:
        str: Joined paragraphs, one per gene
    """

    if not gene_names or not gene_infos or len(gene_names) != len(gene_infos):
        return "I couldn't assemble gene information to answer that question."

    blocks = []
    for name, info in zip(gene_names, gene_infos):
        if info.get("found"):
            blocks.append(
                f"Gene: {name}\n"
                f"Full Name: {info.get('full_name', 'N/A')}\n"
                f"Location: {info.get('chromosome', 'N/A')}\n"
                f"Function: {info.get('function', 'N/A')}\n"
                f"Description: {info.get('description', '')}\n"
            )
        else:
            blocks.append(
                f"Gene: {name}\n"
                f"Description: Not found in knowledge base; provide a concise general explanation if possible.\n"
            )
    if DEBUG:
        print("---")
        print("Context blocks for LLM:")
        for block in blocks:
            print(block)
            print("---")
    context = "\n---\n".join(blocks)

    prompt = f"""
You are a helpful bioinformatics assistant. Answer the user's question about this gene
using the information provided. Explain in clear, accessible language suitable for researchers.

Context:
{context}

User's question: {user_question}

Output format (one paragraph per gene):
GENE_1: <2-3 sentence explanation>
...
GENE_N: <2-3 sentence explanation>
"""

    answer = call_ollama(prompt, temperature=0.4)

    # Fallback: if the LLM returns nothing, stitch descriptions to avoid empty output
    if not answer:
        stitched = []
        for name, info in zip(gene_names, gene_infos):
            desc = info.get('description') or 'No information available.'
            stitched.append(f"{name}: {desc}")
        return "\n\n".join(stitched)

    return answer.strip()


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

    genes = extract_genes_from_question(user_input)

    if not genes:
        return "I couldn't identify which gene you're asking about. Please mention one or more specific gene names."

    gene_infos = []
    for gene_name in genes:
        gene_info = fetch_from_mygene_api(gene_name)
        if DEBUG:
            print(gene_info)
        gene_infos.append(gene_info)

    return generate_gene_explanations(genes, gene_infos, user_input)


def extract_genes_from_question(user_input: str) -> list[str]:
    """
    Extract one or more gene names from a question.

    Returns a list of uppercase gene symbols (may be empty).
    """

    details = extract_gene_name_with_details(user_input)
    genes = []
    if details and details.get("success"):
        genes = [g.upper().strip() for g in details.get("gene_names", []) if g]

    # Fallback to single-gene extraction if multi-gene parsing failed
    if not genes:
        gene = ask_llm_to_extract(
            user_input,
            what_to_extract="gene name",
            example="TP53"
        )
        if gene and gene != "NONE":
            genes = [gene.upper().strip()]

    # Deduplicate while preserving order
    seen = set()
    deduped = []
    for g in genes:
        if g and g not in seen:
            deduped.append(g)
            seen.add(g)

    return deduped


def extract_gene_from_question(user_input: str) -> str:
    """Backward-compatible helper returning only the first gene (if any)."""

    genes = extract_genes_from_question(user_input)
    return genes[0] if genes else None


# ============================================
# Test Runner
# ============================================

def judge_answer_with_llm(question: str, answer: str) -> tuple[str, str]:
    """
    Use the LLM as a lightweight judge to grade an answer.

    Returns:
        verdict: correct | partial | incorrect | unknown
        reason: short rationale from the judge
    """

    prompt = f"""
You are scoring an answer to a gene information question.
Provide:
verdict: correct | partial | incorrect
reason: one short sentence explaining why.

Question: {question}
Answer: {answer}
"""

    resp = call_ollama(prompt, temperature=0)
    if not resp:
        return "unknown", "No judge response"

    verdict = "unknown"
    reason = ""
    for line in resp.splitlines():
        lower = line.lower()
        if lower.startswith("verdict:"):
            verdict = line.split(":", 1)[1].strip().lower()
        if lower.startswith("reason:"):
            reason = line.split(":", 1)[1].strip()

    if not reason:
        reason = resp.strip()

    return verdict, reason


def run_batch_tests(
    test_path: str,
    max_rows: int | None = None,
    output_path: str | None = None,
    run_judge: bool = True,
    manual_review_count: int = 3,
) -> None:
    """
    Load test cases from CSV and evaluate intent detection, gene extraction, and (optionally) answer quality.

    Args:
        test_path: Path to CSV with columns user_input, expected_intent, expected_gene.
        max_rows: Optional cap on number of rows to evaluate.
        output_path: Optional CSV path to write detailed results (predictions, answers, judge verdict).
        run_judge: If True, call LLM judge for info-type questions.
        manual_review_count: Number of info answers to print for manual inspection.
    """

    if not os.path.exists(test_path):
        print(f"No test data found at {test_path}")
        return

    df = pd.read_csv(test_path)
    if max_rows:
        df = df.head(max_rows)

    total = len(df)
    if total == 0:
        print("Test file is empty.")
        return

    intent_correct = 0
    gene_correct = 0
    mismatches = []
    results = []
    manual_review = []

    for idx, row in df.iterrows():
        user_input = row["user_input"]
        expected_intent = str(row["expected_intent"]).strip().lower()
        expected_gene_cell = str(row["expected_gene"]).strip()
        expected_genes = [g.strip().upper() for g in re.split(r"[;,]", expected_gene_cell) if g.strip()] if expected_gene_cell else []

        predicted_intent = "info" if is_gene_question(user_input) else "plot"
        predicted_genes = extract_genes_from_question(user_input)
        predicted_gene = predicted_genes[0] if predicted_genes else ""

        answer = ""
        judge_verdict = "skipped"
        judge_reason = ""

        if predicted_intent == "info":
            answer = answer_gene_question(user_input)
            if run_judge:
                judge_verdict, judge_reason = judge_answer_with_llm(user_input, answer)
            if len(manual_review) < manual_review_count:
                manual_review.append(
                    {
                        "index": idx,
                        "user_input": user_input,
                        "answer": answer,
                        "judge_verdict": judge_verdict,
                        "judge_reason": judge_reason,
                    }
                )

        intent_ok = predicted_intent == expected_intent
        gene_ok = all(g in predicted_genes for g in expected_genes) if expected_genes else False

        intent_correct += int(intent_ok)
        gene_correct += int(gene_ok)

        if not (intent_ok and gene_ok):
            mismatches.append(
                {
                    "index": idx,
                    "user_input": user_input,
                    "expected_intent": expected_intent,
                    "predicted_intent": predicted_intent,
                    "expected_gene": expected_gene_cell,
                    "expected_genes": expected_genes,
                    "predicted_gene": predicted_gene,
                    "predicted_genes": predicted_genes,
                    "answer": answer,
                    "judge_verdict": judge_verdict,
                    "judge_reason": judge_reason,
                }
            )

        results.append(
            {
                "user_input": user_input,
                "expected_intent": expected_intent,
                "predicted_intent": predicted_intent,
                "expected_gene": expected_gene_cell,
                "expected_genes": expected_genes,
                "predicted_gene": predicted_gene,
                "predicted_genes": predicted_genes,
                "answer": answer,
                "judge_verdict": judge_verdict,
                "judge_reason": judge_reason,
            }
        )

    print(f"Ran {total} test cases (max_rows={max_rows}).")
    print(f"Intent accuracy: {intent_correct}/{total}")
    print(f"Gene extraction accuracy: {gene_correct}/{total}")

    if mismatches:
        print("First few mismatches (up to 5):")
        for miss in mismatches[:5]:
            print(
                f"- #{miss['index']}: '{miss['user_input']}' "
                f"expected intent={miss['expected_intent']} predicted intent={miss['predicted_intent']} "
                f"expected gene={miss['expected_gene']} predicted gene={miss['predicted_gene']}"
            )
    else:
        print("All test cases matched expectations.")

    if manual_review:
        print("\nManual review samples (info answers):")
        for item in manual_review:
            print(
                f"- #{item['index']} | {item['user_input']}\n"
                f"  Answer: {item['answer']}\n"
                f"  Judge: {item['judge_verdict']} ({item['judge_reason']})"
            )

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        pd.DataFrame(results).to_csv(output_path, index=False)
        print(f"Detailed results written to {output_path}")


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
    info = fetch_from_mygene_api(gene)
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

    # Batch tests from CSV (adjust max_rows to cap runtime)
    print("4. Batch test suite from CSV:")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(TEST_OUTPUT_DIR, f"gene_query_results_{timestamp}.csv")
    run_batch_tests(
        TEST_DATA_PATH,
        max_rows=None,
        output_path=output_path,
        run_judge=True,
        manual_review_count=5,
    )

    
