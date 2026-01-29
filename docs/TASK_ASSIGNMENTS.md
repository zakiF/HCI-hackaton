# Detailed Task Assignments

## Overview

Each person/team has ONE main task: Implement an LLM-powered feature.

**Work independently** on your feature branch, then integrate at Day 2 11am.

---

## 📋 Task 1: Multi-Gene Visualization (R)

**Team Members**: Zaki Wilmot + Udhayakumar Gopal

**Branch**: `feature/multi-gene`

**Files to Edit**:
- `modules/R/multi_gene_viz.R` (your main work)
- Minimal edits to `app/python/chatbot.py` (during integration only)

### **What You're Building**

An LLM-powered system that:
1. Extracts multiple gene names from natural language
2. Detects which plot type the user wants
3. Routes to appropriate plotting function

### **Example Queries Your Feature Handles**

```
Input: "Compare TP53, BRCA1, and EGFR"
LLM extracts: ["TP53", "BRCA1", "EGFR"]
LLM decides: Use heatmap (3 genes)
Output: Heatmap plot

Input: "Show me TP53 as a violin plot"
LLM extracts: ["TP53"]
LLM decides: Use violin plot
Output: Violin plot
```

### **Step-by-Step Implementation**

#### **Step 1: Test Ollama from R** (30 min)

```r
# In RStudio:
source("utils/R/ollama_utils.R")

# Test basic call
response <- call_ollama("What is 2+2?")
print(response)

# Test extraction
genes <- ask_llm_to_extract("Show me TP53 and BRCA1", "gene names")
print(genes)
```

**Goal**: Make sure you can call Ollama successfully

#### **Step 2: Implement `extract_gene_names()`** (1-2 hours)

Edit `modules/R/multi_gene_viz.R`:

```r
extract_gene_names <- function(user_input) {
  # TODO: Write your LLM prompt here

  prompt <- paste0(
    "Extract ALL gene names from this query.\n",
    "Return as JSON array with key 'genes'.\n\n",
    "Query: ", user_input, "\n\n",
    "JSON:"
  )

  result <- call_ollama_json(prompt, temperature = 0.1)

  # ... process result

  return(genes)
}
```

**Test it**:
```r
genes <- extract_gene_names("Compare TP53, BRCA1, and EGFR")
print(genes)  # Should print: c("TP53", "BRCA1", "EGFR")
```

**Iterate**: Try different prompts if extraction fails.

#### **Step 3: Implement `detect_plot_type()`** (1 hour)

```r
detect_plot_type <- function(user_input, num_genes) {
  # TODO: Write LLM prompt to detect plot type

  prompt <- paste0(
    "What type of plot does the user want?\n",
    "Options: boxplot, violin, heatmap, barplot\n\n",
    "Query: ", user_input, "\n\n",
    "Plot type:"
  )

  # ... call LLM and return plot type
}
```

#### **Step 4: Create Plotting Functions** (2-3 hours)

Implement these three functions:
- `plot_gene_boxplot()` ✅ (example provided)
- `plot_gene_violin()` ✅ (example provided)
- `plot_genes_heatmap()` ✅ (example provided)

**You can use the provided examples or customize them!**

#### **Step 5: Wire It All Together** (30 min)

The `plot_genes_smart()` function calls your LLM functions and routes to plotting.

Test it:
```r
plot <- plot_genes_smart("Compare TP53 and BRCA1", expr_data, metadata)
print(plot)
```

### **LLM Skills You'll Practice**

- ✅ Parameter extraction (multiple gene names)
- ✅ Structured output (JSON list of genes)
- ✅ Intent classification (which plot type)

### **Testing Strategy**

Test with these queries:
- "Show me TP53"
- "Compare TP53 and BRCA1"
- "Plot TP53, BRCA1, and EGFR"
- "Show me MYC as violin plot"

### **Integration Into Chatbot**

During integration (Day 2 11am), your code will be called like this:

```python
# In chatbot.py
if ENABLE_MULTI_GENE and len(genes) > 1:
    # Call R function via rpy2
    plot = call_r_multi_gene_function(genes, expr_data)
```

**Note**: You don't need to worry about Python integration - just make sure your R functions work!

---

## 📋 Task 2: Natural Language Filtering

**Team Member**: Qing Li

**Branch**: `feature/filters`

**Files to Edit**:
- `modules/python/llm_filters.py` (your main work)

### **What You're Building**

An LLM system that understands filtering requests in natural language.

### **Example Queries**

```
Input: "Show TP53 in only tumor samples"
LLM extracts: {"condition": "Primary Tumor", "exclude": false}
Code applies filter
Output: Filtered data

Input: "Display BRCA1 excluding metastatic"
LLM extracts: {"condition": "Metastatic", "exclude": true}
Output: Data without metastatic samples
```

### **Step-by-Step Implementation**

#### **Step 1: Test Ollama** (15 min)

```python
from utils.python.ollama_utils import call_ollama, call_ollama_json

response = call_ollama("What is 2+2?")
print(response)
```

#### **Step 2: Implement `extract_filters()`** (2-3 hours)

Focus on writing a good LLM prompt:

```python
def extract_filters(user_input: str) -> dict:
    prompt = f"""
Extract filtering information from this query.

Dataset conditions: "Normal", "Primary Tumor", "Metastatic"

Return JSON with:
- condition: which condition (or null)
- exclude: true/false
- description: brief description

Query: {user_input}

JSON:
"""

    result = call_ollama_json(prompt, temperature=0.1)
    # ... process and return
```

**Test cases**:
- "Show TP53 in tumor samples only"
- "Display BRCA1 in normal tissue"
- "Plot EGFR excluding metastatic"
- "Compare MYC in normal and tumor" (no filter - should detect!)

#### **Step 3: Implement `apply_filters()`** (1 hour)

This is mostly pandas code:

```python
def apply_filters(expr_data: dict, filter_info: dict) -> dict:
    metadata = expr_data['metadata']

    if filter_info['exclude']:
        keep_samples = metadata[metadata['condition'] != filter_info['condition']].index
    else:
        keep_samples = metadata[metadata['condition'] == filter_info['condition']].index

    # ... subset data and return
```

#### **Step 4: Test Standalone** (30 min)

```bash
python modules/python/llm_filters.py
```

Should run test cases and show results.

### **LLM Skills You'll Practice**

- ✅ Parameter extraction (filter conditions)
- ✅ Structured output (JSON)
- ✅ Intent classification (detect when filtering is needed)

### **Integration**

Your functions will be called in `chatbot.py` like this:

```python
if ENABLE_FILTERS:
    filter_info = extract_filters(user_input)
    if filter_info['has_filter']:
        filtered_data = apply_filters(expr_data, filter_info)
```

---

## 📋 Task 3: Conversational Context

**Team Member**: Miao Ai

**Branch**: `feature/conversation`

**Files to Edit**:
- `modules/python/conversation.py` (your main work)

### **What You're Building**

A conversation manager that tracks context and resolves references like "it", "that gene".

### **Example Flow**

```
Turn 1:
User: "Show me TP53"
System stores: last_gene = "TP53"

Turn 2:
User: "Now show it as violin plot"
LLM resolves: "it" → "TP53"
Returns: "Show TP53 as violin plot"
```

### **Step-by-Step Implementation**

#### **Step 1: Understand the Class Structure** (15 min)

The `ConversationManager` class maintains state:

```python
class ConversationManager:
    def __init__(self):
        self.history = []  # List of (user_input, gene_name) tuples
        self.last_gene = None
```

#### **Step 2: Implement `_is_followup_query()`** (30 min)

Detect if query has contextual references:

```python
def _is_followup_query(self, user_input: str) -> bool:
    # Option 1: Simple keywords
    followup_keywords = ['it', 'that', 'also', 'now']
    return any(kw in user_input.lower() for kw in followup_keywords)

    # Option 2: Use LLM (more sophisticated)
    # ...
```

#### **Step 3: Implement `_resolve_with_llm()`** (2-3 hours)

This is the core LLM work:

```python
def _resolve_with_llm(self, user_input: str, context: str) -> str:
    prompt = f"""
You are resolving contextual references in a conversation.

{context}

User just said: "{user_input}"

Rewrite by replacing "it"/"that" with the actual gene name.

Rewritten query:
"""

    resolved = call_ollama(prompt, temperature=0.1)
    return resolved
```

#### **Step 4: Test** (1 hour)

```python
conv_mgr = ConversationManager()
conv_mgr.add_turn("Show me TP53", "TP53")

resolved = conv_mgr.resolve_context("Now show it as violin")
print(resolved)  # Should be: "Show TP53 as violin"
```

### **LLM Skills You'll Practice**

- ✅ Intent classification (follow-up vs new query)
- ✅ Parameter extraction (resolve references)
- ✅ Context management (use conversation history)

### **Integration**

Called early in chatbot pipeline:

```python
if ENABLE_CONVERSATION:
    resolved_input = conversation_mgr.resolve_context(
        user_input,
        current_gene
    )
```

---

## 📋 Task 4: Statistical Testing

**Team Member**: Tayler Fearn

**Branch**: `feature/stats`

**Files to Edit**:
- `modules/python/llm_stats.py` (your main work)

### **What You're Building**

LLM-powered statistical analysis with natural language summaries.

### **Example**

```
Input: "Is TP53 significantly different between tumor and normal?"

LLM extracts:
  gene: "TP53"
  test: "t-test"
  groups: ["Primary Tumor", "Normal"]

Hardcoded function runs: scipy.stats.ttest_ind()

Results: {statistic: -2.45, pvalue: 0.023}

LLM generates summary:
  "Yes, TP53 shows significantly different expression between
   Primary Tumor and Normal samples (p=0.023), with tumor
   samples showing higher expression."
```

### **Step-by-Step Implementation**

#### **Step 1: Implement `is_stats_query()`** (30 min)

```python
def is_stats_query(user_input: str) -> bool:
    # Simple keyword detection or LLM classification
    stats_keywords = ['significant', 'different', 'compare', 'test']
    return any(kw in user_input.lower() for kw in stats_keywords)
```

#### **Step 2: Implement `extract_stats_parameters()`** (2-3 hours)

This is your main LLM work:

```python
def extract_stats_parameters(user_input: str) -> dict:
    prompt = f"""
Extract statistical test info from this query.

Conditions: "Normal", "Primary Tumor", "Metastatic"

Return JSON with:
- gene: gene name
- groups: list of conditions to compare
- test: "t-test" (2 groups) or "anova" (3+ groups)

Query: {user_input}

JSON:
"""

    result = call_ollama_json(prompt, temperature=0.1)
    # ... validate and return
```

#### **Step 3: Test Hardcoded Stats Functions** (30 min)

The stats functions are already provided:
- `run_ttest()` ✅
- `run_anova()` ✅

Test them with real data to make sure they work.

#### **Step 4: Implement `generate_stats_summary()`** (1-2 hours)

Use LLM to explain results:

```python
def generate_stats_summary(stats_results: dict, user_question: str) -> str:
    stats_context = f"""
Gene: {stats_results['gene']}
Test: {stats_results['test']}
P-value: {stats_results['pvalue']}
Significant: {stats_results['significant']}
"""

    prompt = f"""
Explain these stats results in 1-2 sentences.

{stats_context}

User asked: {user_question}

Explanation:
"""

    return call_ollama(prompt, temperature=0.3)
```

### **LLM Skills You'll Practice**

- ✅ Intent classification (stats vs plot)
- ✅ Parameter extraction (gene, test, groups)
- ✅ Natural language generation (explain results)
- ✅ Structured output (JSON)

### **Integration**

```python
if ENABLE_STATS and is_stats_query(user_input):
    result = handle_stats_query(user_input, expr_data)
    st.write(result['summary'])
```

---

## 📋 Task 5: Gene Information RAG

**Team Member**: David Stone

**Branch**: `feature/rag`

**Files to Edit**:
- `modules/python/llm_rag.py` (your main work)

### **What You're Building**

RAG (Retrieval-Augmented Generation) for gene information:
1. Retrieve gene info from knowledge base
2. Use LLM to generate readable explanation

### **Example**

```
Input: "What does TP53 do?"

Step 1 - Retrieval:
  Retrieved from database: "TP53 is a tumor suppressor gene..."

Step 2 - Generation:
  LLM takes retrieved info and generates:
  "TP53, known as the 'guardian of the genome', is a crucial
   tumor suppressor gene that regulates cell division and
   prevents cancer. Mutations in TP53 are found in over 50%
   of human cancers."
```

### **Step-by-Step Implementation**

#### **Step 1: Expand Gene Database** (1-2 hours)

Edit `retrieve_gene_annotation()`:

```python
gene_database = {
    "TP53": {
        "full_name": "Tumor Protein P53",
        "description": "...",
        "function": "..."
    },
    # Add more genes here!
    "BRCA1": {...},
    "EGFR": {...},
    # ...
}
```

**Bonus**: Load from CSV file instead of hardcoding.

#### **Step 2: Implement `is_gene_question()`** (30 min)

```python
def is_gene_question(user_input: str) -> bool:
    question_keywords = ['what', 'tell me', 'explain', 'function']
    user_lower = user_input.lower()

    for kw in question_keywords:
        if kw in user_lower:
            # Make sure it's not a plot request
            if 'show' not in user_lower and 'plot' not in user_lower:
                return True

    return False
```

#### **Step 3: Implement `generate_gene_explanation()`** (2-3 hours)

This is the RAG generation part:

```python
def generate_gene_explanation(gene_name, gene_info, user_question):
    context = f"""
Gene: {gene_info['gene']}
Full Name: {gene_info['full_name']}
Function: {gene_info['function']}
Description: {gene_info['description']}
"""

    prompt = f"""
You are a bioinformatics assistant.

Gene Information:
{context}

User's question: {user_question}

Your answer (2-3 sentences):
"""

    return call_ollama(prompt, temperature=0.4)
```

#### **Step 4: Test** (30 min)

```bash
python modules/python/llm_rag.py
```

Try different questions:
- "What does TP53 do?"
- "Tell me about BRCA1"
- "Explain the function of EGFR"

### **LLM Skills You'll Practice**

- ✅ Intent classification (question vs plot)
- ✅ Parameter extraction (gene name from question)
- ✅ Natural language generation (explain gene info)
- ✅ RAG pattern (retrieval + generation)

### **Integration**

```python
if ENABLE_RAG and is_gene_question(user_input):
    answer = answer_gene_question(user_input)
    st.write(answer)
    # Stop here - don't plot
```

---

## 🎯 General Tips for Everyone

### **Prompt Engineering Tips**

1. **Be specific**: "Return ONLY the gene name" works better than "What's the gene?"
2. **Give examples**: "Return as JSON. Example: {\"gene\": \"TP53\"}"
3. **Use structure**: Break complex prompts into sections
4. **Set temperature low** (0.0-0.2) for extraction, higher (0.3-0.5) for generation

### **Testing Strategy**

1. **Test LLM calls first**: Make sure Ollama works
2. **Test your module standalone**: Run the module file directly
3. **Try edge cases**: Typos, ambiguous queries, missing genes
4. **Iterate on prompts**: If LLM fails, rewrite the prompt

### **Time Management**

- **Hour 0-2**: Get basic LLM calling working, test with simple prompts
- **Hour 2-5**: Implement main feature, iterate on prompts
- **Hour 5-7**: Test thoroughly, handle edge cases
- **Hour 7**: Integration (one person drives, others support)

### **When You're Stuck**

1. Check the example code at bottom of your module file
2. Run `demonstrate_llm_skills()` in `ollama_utils.py/R`
3. Ask David or Tayler for help
4. Test with simpler prompts first, then add complexity

---

## 📦 Deliverables Per Person

By end of hackathon:

- ✅ Working module file with implemented functions
- ✅ Test cases that demonstrate your feature
- ✅ Can explain: What LLM skills did you practice?
- ✅ Can demo: Show your feature working (even if not integrated)

**Success = Learning happened**, not perfect code!
