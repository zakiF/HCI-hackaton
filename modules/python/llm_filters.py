"""
Natural Language Data Filtering Module
Team Member: Qing Li

LLM Learning Goals:
1. Parameter extraction: Pull out filter conditions from natural language
2. Structured output: Get LLM to return JSON with filter parameters
3. Intent classification: Detect when user wants filtering

Dataset Reality (IMPORTANT):
- Metadata columns: Run, group
- group values look like: g2.tumor, g1.normal, g3.mets
- Expression matrix is genes x samples (sample IDs are columns)

Examples we handle:
- "Show me TP53 in only tumor samples" → group_suffix == "tumor"
- "Display BRCA1 in normal tissue" → group_suffix == "normal"
- "Plot EGFR excluding metastatic" → group_suffix != "mets"
"""

import sys
import os
import re
from typing import Optional, Dict, Any

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from utils.python.ollama_utils import call_ollama_json


# ============================================
# Helpers / Constants
# ============================================

# What we actually filter on in this dataset
_ALLOWED_GROUP_SUFFIXES = ["normal", "tumor", "mets", "non_normal"]


# Natural language -> group suffix
_SYNONYMS_TO_SUFFIX = {
    # ---- primary tumor ONLY (explicit) ----
    "primary tumor": "tumor",
    "primary tumour": "tumor",

    # ---- normal ----
    "normal": "normal",
    "healthy": "normal",
    "control": "normal",
    "non-tumor": "normal",
    "non tumor": "normal",

    # ---- metastatic ----
    "metastatic": "mets",
    "metastasis": "mets",
    "mets": "mets",

    # ---- non-normal (tumor + mets) ----
    "non-normal": "non_normal",
    "non normal": "non_normal",
    "all tumor": "non_normal",
    "all tumors": "non_normal",
    "all tumours": "non_normal",
    "tumor including mets": "non_normal",
    "tumour including mets": "non_normal",
    "tumor + mets": "non_normal",
    "tumor and mets": "non_normal",

    # KEY RULE: plain tumor ALWAYS means non_normal
    "tumor": "non_normal",
    "tumour": "non_normal",
}



_EXCLUDE_WORDS = ["exclude", "excluding", "without", "remove", "not", "omit", "except"]
_INCLUDE_HINTS = ["only", "just", "in ", "from ", "within", "restricted to"]
_MULTI_GROUP_CUES = [" vs ", " versus ", "compare", "between", " and ", " plus ", "along with", ","]
def _is_multi_group_request(text: str) -> bool:
    tl = text.lower()
    return any(cue in tl for cue in _MULTI_GROUP_CUES)


def _normalize_suffix(raw: Optional[str]) -> Optional[str]:
    """Normalize LLM output or raw user condition text -> one of tumor/normal/mets or None."""
    if raw is None:
        return None

    raw_l = str(raw).strip().lower()

    # exact
    if raw_l in _ALLOWED_GROUP_SUFFIXES:
        return raw_l

    # synonym match
    for k, v in _SYNONYMS_TO_SUFFIX.items():
        if k in raw_l:
            return v

    return None


def _mentioned_suffixes(text: str) -> list[str]:
    """Detect which known conditions appear in text, mapped to suffixes."""
    t = text.lower()
    found = set()
    for k, v in _SYNONYMS_TO_SUFFIX.items():
        if re.search(rf"\b{re.escape(k)}\b", t):
            found.add(v)
    return sorted(found)


def _has_filter_intent(text: str) -> bool:
    """Rule-based intent classification (fast + reliable)."""
    tl = text.lower()
    return (
        any(w in tl for w in _EXCLUDE_WORDS)
        or any(w in tl for w in _INCLUDE_HINTS)
        or len(_mentioned_suffixes(tl)) > 0
    )


# ============================================
# Main Functions
# ============================================

def extract_filters(user_input: str) -> Dict[str, Any]:
    """
    Use LLM to extract considered filtering criteria from user's query.

    Returns dict with keys:
      - has_filter: bool
      - group_suffix: str|None (tumor|normal|mets)
      - exclude: bool
      - description: str

    Notes:
      - If multiple conditions are mentioned (e.g., "normal vs tumor"), we return
        group_suffix=None to avoid applying an incorrect single filter.
    """
    text = user_input.strip()
    text_l = text.lower()

    has_filter = _has_filter_intent(text)
    if not has_filter:
        return {
            "has_filter": False,
            "group_suffix": None,
            "exclude": False,
            "description": "no filter",
        }

       
    mentioned = _mentioned_suffixes(text)

    # Multi-group: filter to UNION of mentioned groups (single filtered dataset)
    if _is_multi_group_request(text) and len(mentioned) >= 2:
        # normalize: if both non_normal and mets present, union is non_normal (since mets ⊂ non_normal)
        uniq = sorted(set(mentioned))
        if "non_normal" in uniq and "mets" in uniq:
            uniq = [x for x in uniq if x != "mets"]

        return {
            "has_filter": True,
            "group_suffix": None,
            "exclude": False,
            "include_suffixes": uniq,
            "description": f"compare {uniq[0]} vs {uniq[1]}" if len(uniq) == 2 else f"compare groups: {', '.join(uniq)}",
        }

    # LLM prompt: return suffix values that match this dataset
    prompt = f"""
You extract filtering parameters from user queries about RNA-seq plots.

You are a JSON generator. You MUST output EXACTLY one JSON object and NOTHING else.
No prose. No markdown. No code fences. No Python. No backticks.

Schema (must match exactly):
{{"group_suffix":"normal|tumor|mets|non_normal|null","exclude":true|false,"include_suffixes": ["normal","tumor","mets","non_normal"]|null,"description":"string"}}


The metadata has a column 'group' with values like:
- g2.tumor
- g1.normal
- g3.mets


Rules:
- Map user language:
  - "normal" / "non-tumor" -> "normal"
  - "primary tumor" -> "tumor"
  - "metastatic" / "mets" -> "mets"
  - "tumor" / "all tumor" / "non-normal" / "tumor including mets" -> "non_normal"   (tumor + mets)

- exclusion words (exclude/excluding/without/not/except) => exclude=true
- "only"/"just" => exclude=false
- if no group mentioned => group_suffix=null

If you output anything other than the JSON object, the program will fail.

Examples:

Query: "Plot EGFR excluding metastatic"
JSON: {{ "group_suffix": "mets", "exclude": true, "include_suffixes": null, "description": "exclude metastatic samples" }}

Query: "Show TP53 without mets"
JSON: {{ "group_suffix": "mets", "exclude": true, "include_suffixes": null, "description": "exclude mets samples" }}

Query: "Only metastatic"
JSON: {{ "group_suffix": "mets", "exclude": false, "include_suffixes": null, "description": "only metastatic samples" }}

Query: "Display BRCA1 in normal tissue"
JSON: {{ "group_suffix": "normal", "exclude": false, "include_suffixes": null, "description": "only normal samples" }}

Query: "Show TP53 in tumor samples"
JSON: {{ "group_suffix": "non_normal", "exclude": false, "include_suffixes": null, "description": "all tumor (primary + mets)" }}

Query: "Show TP53 in primary tumor only"
JSON: {{ "group_suffix": "tumor", "exclude": false, "include_suffixes": null, "description": "primary tumor only" }}

Query: "Show TP53 in non-tumor samples"
JSON: {{ "group_suffix": "normal", "exclude": false, "include_suffixes": null, "description": "normal only" }}

Query: "Compare TP53 in normal vs mets"
JSON: {{ "group_suffix": null, "exclude": false, "include_suffixes": ["normal", "mets"], "description": "compare normal vs mets" }}

Query: "Plot TP53 in normal and tumor"
JSON: {{ "group_suffix": null, "exclude": false, "include_suffixes": ["normal", "non_normal"], "description": "compare normal vs non-normal" }}

Query: "Compare TP53 in primary tumor vs mets"
JSON: {{ "group_suffix": null, "exclude": false, "include_suffixes": ["tumor", "mets"], "description": "compare primary tumor vs mets" }}


Now process this query:
Query: {user_input}

JSON:
""".strip()



    llm_result = call_ollama_json(prompt, temperature=0.0)

    # Defaults / fallback
    exclude = any(w in text_l for w in _EXCLUDE_WORDS)
    group_suffix = mentioned[0] if len(mentioned) == 1 else None
    description = "custom filter"

    if isinstance(llm_result, dict):
        exclude = bool(llm_result.get("exclude", exclude))
        group_suffix = _normalize_suffix(llm_result.get("group_suffix")) or group_suffix
        description = str(llm_result.get("description", description))

    # If LLM failed to identify but we have a mention, keep fallback
    if group_suffix is None and len(mentioned) == 1:
        group_suffix = mentioned[0]

    if group_suffix is None:
        # Filter intent but no specific group
        return {
            "has_filter": True,
            "group_suffix": None,
            "exclude": exclude,
            "description": "filter intent detected but no group identified",
        }

    return {
        "has_filter": True,
        "group_suffix": group_suffix,
        "exclude": exclude,
        "description": description,
    }


def apply_filters(expr_data: dict, filter_info: dict) -> dict:
    if not filter_info.get("has_filter"):
        return expr_data

    metadata = expr_data["metadata"]
    expression = expr_data["expression"]

    if "group" not in metadata.columns:
        return expr_data

    g = metadata["group"].astype(str)

    def in_suffix(suf: str):
        if suf == "non_normal":
            return ~g.str.endswith("normal")  # tumor + mets
        return g.str.endswith(suf)  # normal / tumor / mets

    # Multi-group union filter
    include_suffixes = filter_info.get("include_suffixes")
    if isinstance(include_suffixes, list) and len(include_suffixes) > 0:
        uniq = sorted(set(include_suffixes))
        # simplify redundant union: non_normal already includes mets
        if "non_normal" in uniq and "mets" in uniq:
            uniq = [x for x in uniq if x != "mets"]

        mask = in_suffix(uniq[0])
        for suf in uniq[1:]:
            mask = mask | in_suffix(suf)

        keep_samples = metadata.index[mask]
        return {
            "metadata": metadata.loc[keep_samples],
            "expression": expression.loc[:, keep_samples],
        }

    # Single-group include/exclude
    group_suffix = filter_info.get("group_suffix")
    exclude = filter_info.get("exclude", False)
    if group_suffix is None:
        return expr_data

    mask = in_suffix(group_suffix)
    keep_samples = metadata.index[~mask] if exclude else metadata.index[mask]

    return {
        "metadata": metadata.loc[keep_samples],
        "expression": expression.loc[:, keep_samples],
    }





# ============================================
# Testing / Development
# ============================================

if __name__ == "__main__":
    print("=== Testing Filter Extraction ===\n")

    test_queries = [
        "Show me TP53",
        "Show me TP53 in tumor samples only",
        "Display BRCA1 in normal tissue",
        "Plot EGFR excluding metastatic",
        "Compare TP53 in normal vs tumor",
        "Show TP53 without mets",
        "Only metastatic please",
    ]

    for query in test_queries:
        print(f"Query: {query}")
        result = extract_filters(query)
        print(f"Result: {result}\n")

    print("=== Tips ===")
    print("1) Use temperature=0.0 for stable extraction")
    print("2) Keep rule-based fallback for reliability")
    print("3) Confirm group suffixes match metadata: tumor/normal/mets")
