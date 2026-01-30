"""
Conversational Context Management Module
Team Member: Miao Ai

LLM Learning Goals:
1. Intent classification: Detect follow-up vs new query
2. Parameter extraction: Resolve references like "it", "that gene"
3. Context management: Track conversation state

Examples:
- User: "Show me TP53" → Bot: [shows plot]
- User: "Now show it as a violin plot" → LLM resolves "it" = TP53
- User: "Compare it to BRCA1" → LLM knows "it" = TP53
"""

import os
import re
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from utils.python.ollama_utils import call_ollama


# ============================================
# Conversation Manager Class
# ============================================

class ConversationManager:
    """
    Manages conversation state and resolves context.

    This is similar to session management in backend systems - maintaining
    state across multiple user interactions.
    """

    def __init__(self):
        """Initialize conversation manager."""
        self.history = []  # List of (user_input, gene_name, filter_info) tuples
        self.last_gene = None
        self.last_filter_info = None

    def add_turn(self, user_input: str, gene_name: str, filter_info: dict = None):
        """
        Record a conversation turn.

        Args:
            user_input: What the user asked
            gene_name: Which gene was plotted
            filter_info: Filter information (optional), e.g., {'has_filter': True, 'group_suffix': 'tumor', ...}

        Example:
            >>> conv_mgr.add_turn("Show me TP53", "TP53")
            >>> conv_mgr.add_turn("Plot TP53 in tumor and normal", "TP53", filter_info={'has_filter': True, ...})
        """
        self.history.append((user_input, gene_name, filter_info))
        self.last_gene = gene_name
        self.last_filter_info = filter_info

    def resolve_context(self, user_input: str, current_gene: str = None) -> str:
        """
        Resolve contextual references in user input.

        Uses LLM to understand if user is referring to previous context.

        Args:
            user_input: Current user query
            current_gene: Currently displayed gene (if any)

        Returns:
            str: Resolved query with context filled in

        Example:
            >>> # After showing TP53:
            >>> resolved = conv_mgr.resolve_context("Show it as violin plot")
            >>> print(resolved)
            "Show TP53 as violin plot"

        TODO for Miao:
        1. Detect if query has contextual references ("it", "that", etc.)
        2. Use LLM to resolve references using conversation history
        3. Return expanded/resolved query
        """

        # STEP 1: Check if this is a follow-up query
        is_followup = self._is_followup_query(user_input)

        if not is_followup:
            return user_input  # No context resolution needed

        # STEP 2: Build context from history
        context = self._build_context(current_gene=current_gene)

        # STEP 3: Use LLM to resolve references
        resolved_query = self._resolve_with_llm(user_input, context)

        return resolved_query

    def _is_followup_query(self, user_input: str) -> bool:
        """
        Detect if query is a follow-up (has contextual references).

        TODO for Miao:
        - Use LLM or keyword detection
        - Look for: "it", "that", "also", "now", "change to", etc.

        Example:
            >>> self._is_followup_query("Show it as violin")
            True
            >>> self._is_followup_query("Show me BRCA1")
            False
        """

        user_lower = user_input.lower()

        # LLM-based classification (preferred)
        llm_prompt = f"""
You are classifying whether a user query is a follow-up that depends on prior context.

Examples of contextual keywords: it, that, this, those, them, one, same, also, previous, earlier, now, change to.

Return ONLY "YES" or "NO".

Query: "{user_input}"

Answer:
"""

        llm_result = call_ollama(llm_prompt, temperature=0.1)
        if llm_result:
            llm_result = llm_result.strip().upper()
            if llm_result == "YES":
                return True
            # If LLM says "NO", still check fallback keywords as LLM might be wrong

        # Fallback: Keyword-based detection (always check this for robustness)
        # Explicit pronoun or reference words as whole words
        if re.search(r"\b(it|that|this|those|them|one|same|previous|earlier)\b", user_lower):
            return True

        # Follow-up intent phrases
        if re.search(r"\b(now|also|instead|change|switch|another|different)\b", user_lower):
            return True

        # Plot-type change without naming a gene - but ONLY if we have conversation history
        # If no history, it's a new query, not a followup
        if len(self.history) > 0 and detect_plot_type_change(user_input) is not None:
            # Check if the query explicitly mentions a gene name
            # If it does, it's not a follow-up, it's a new plot request
            import re as regex
            # Look for patterns like "Plot GENE as", "Show GENE as"
            if regex.search(r'\b(plot|show|display|visualize)\s+[A-Z][A-Z0-9-]{2,}\s+(as|in|with)', user_input, regex.IGNORECASE):
                # This is a new plot request with explicit gene and plot type
                return False
            return True

        return False

    def _build_context(self, current_gene: str = None) -> str:
        """
        Build context string from conversation history.

        TODO for Miao:
        - Format recent history for LLM prompt
        - Include last 2-3 turns
        - Maybe include current gene/plot type

        Example:
            "Previous conversation:
             - User asked about TP53
             - Currently showing: TP53 boxplot"
        """

        if len(self.history) == 0:
            return "No previous conversation."

        # Get last few turns
        recent = self.history[-3:]  # Last 3 turns

        context_lines = ["Previous conversation:"]
        for item in recent:
            # Handle both old format (user_q, gene) and new format (user_q, gene, filter_info)
            if len(item) == 3:
                user_q, gene, filter_info = item
            else:
                user_q, gene = item
                filter_info = None

            context_lines.append(f"- User asked: {user_q}")
            context_lines.append(f"  Gene shown: {gene}")

            # Add filter information if present
            if filter_info and filter_info.get('has_filter'):
                if 'include_suffixes' in filter_info and filter_info['include_suffixes']:
                    # Multi-group filter (e.g., "tumor and normal")
                    conditions = filter_info['include_suffixes']
                    context_lines.append(f"  Conditions: {', '.join(conditions)}")
                elif filter_info.get('group_suffix'):
                    # Single-group filter
                    suffix = filter_info['group_suffix']
                    exclude = filter_info.get('exclude', False)
                    if exclude:
                        context_lines.append(f"  Excluding: {suffix}")
                    else:
                        context_lines.append(f"  Only showing: {suffix}")

        active_gene = current_gene or self.last_gene
        if active_gene:
            context_lines.append(f"\nCurrently showing: {active_gene}")

            # Add current filter info if available
            if self.last_filter_info and self.last_filter_info.get('has_filter'):
                if 'include_suffixes' in self.last_filter_info and self.last_filter_info['include_suffixes']:
                    conditions = self.last_filter_info['include_suffixes']
                    # Map internal suffixes to readable condition names
                    condition_map = {
                        'normal': 'Normal',
                        'tumor': 'Primary Tumor',
                        'mets': 'Metastatic',
                        'non_normal': 'tumor (Primary + Metastatic)'
                    }
                    readable_conditions = [condition_map.get(c, c) for c in conditions]
                    context_lines.append(f"Comparing conditions: {' vs '.join(readable_conditions)}")

        return "\n".join(context_lines)

    def _resolve_with_llm(self, user_input: str, context: str) -> str:
        """
        Use LLM to resolve contextual references.

        TODO for Miao:
        1. Write prompt that gives LLM the context
        2. Ask LLM to rewrite query with references resolved
        3. Return the resolved query

        Example:
            Input: "Show it as violin"
            Context: "Currently showing: TP53"
            Output: "Show TP53 as violin"
        """

        prompt = f"""
You are resolving contextual references in a conversation about gene expression analysis.

{context}

User just said: "{user_input}"

TASK:
Rewrite the query to be standalone and explicit:
- Replace "it"/"that"/"this" with the actual gene name
- If asking about statistical significance/tests, include both the gene AND the conditions being compared
- If conditions are mentioned in context, include them in the rewritten query

EXAMPLES:
- "Is it significant?" + Context: "TP53, comparing tumor vs normal" → "Is TP53 expression significantly different between tumor and normal?"
- "Show it as violin" + Context: "TP53" → "Show TP53 as violin plot"
- "What about normal samples?" + Context: "TP53 in tumor" → "Show TP53 in normal samples"

Return ONLY the rewritten query, nothing else.

Rewritten query:
"""

        resolved = call_ollama(prompt, temperature=0.1)

        if resolved:
            return resolved.strip()

        # Fallback: return original if LLM fails
        return user_input

    def reset(self):
        """Clear conversation history."""
        self.history = []
        self.last_gene = None
        self.last_filter_info = None


# ============================================
# Standalone Helper Functions
# ============================================

def detect_plot_type_change(user_input: str) -> str:
    """
    Detect if user wants to change plot type.

    Args:
        user_input: User's query

    Returns:
        str: Plot type ("violin", "boxplot", "heatmap") or None

    Example:
        >>> detect_plot_type_change("Show it as violin plot")
        "violin"

    TODO for Miao (optional enhancement):
    Use LLM to extract requested plot type.
    """

    user_lower = user_input.lower()

    plot_types = {
        'violin': ['violin'],
        'boxplot': ['boxplot', 'box plot', 'box-plot'],
        'heatmap': ['heatmap', 'heat map'],
        'barplot': ['barplot', 'bar plot', 'bar chart']
    }

    for plot_type, keywords in plot_types.items():
        if any(kw in user_lower for kw in keywords):
            return plot_type

    return None


# ============================================
# Testing / Development
# ============================================

if __name__ == "__main__":
    """
    Test this module independently.

    Run: python modules/python/conversation.py
    """

    print("=== Testing Conversation Manager ===\n")

    # Create conversation manager
    conv_mgr = ConversationManager()

    # Simulate conversation
    print("Turn 1:")
    print("User: Show me TP53")
    conv_mgr.add_turn("Show me TP53", "TP53")
    print()

    print("Turn 2 (follow-up):")
    user_input = "Now show it as a violin plot"
    print(f"User: {user_input}")
    resolved = conv_mgr.resolve_context(user_input)
    print(f"Resolved: {resolved}")
    print()

    print("Turn 3 (new query):")
    user_input = "Show me BRCA1"
    print(f"User: {user_input}")
    resolved = conv_mgr.resolve_context(user_input)
    print(f"Resolved: {resolved}")
    conv_mgr.add_turn("Show me BRCA1", "BRCA1")
    print()

    print("Turn 4 (follow-up referencing new gene):")
    user_input = "Change that to boxplot"
    print(f"User: {user_input}")
    resolved = conv_mgr.resolve_context(user_input)
    print(f"Resolved: {resolved}")
    print()

    print("\n=== Tips for Miao ===")
    print("1. Test with different types of references (it, that, also, etc.)")
    print("2. Handle ambiguous cases (when is 'it' referring to?)")
    print("3. Consider adding plot type to context (not just gene)")
    print("4. Think about: How long should history be kept?")
    print("5. Bonus: Add conversation summarization if history gets long")
