# ChatSeq Hackathon - LLM Learning Project

## 🎯 Main Goal

**Learn to use local LLMs hands-on** by building an interactive chatbot for gene expression visualization.

Each team member will implement **one LLM-powered feature**, gaining practical experience with different LLM techniques.

---

## 🧠 Four Core LLM Skills (Everyone Learns These)

By the end of the hackathon, each person will have hands-on experience with these fundamental LLM skills:

### 1. **Intent Classification**
**What**: Having the LLM categorize user input into predefined categories

**Examples**:
- Is this a plotting request, statistics question, or information query?
- "Show me TP53" → `plot`
- "Is TP53 significant?" → `stats`
- "What does TP53 do?" → `question`

**Who practices this**: Everyone! Each module needs to detect its own intent.

**Code example**:
```python
from utils.python.ollama_utils import ask_llm_to_classify

intent = ask_llm_to_classify(
    user_input,
    options=["plot", "stats", "question"],
    context="What is the user's intent?"
)
```

---

### 2. **Parameter Extraction**
**What**: Having the LLM pull out specific information from natural language

**Examples**:
- Extract gene names: "Show TP53 and BRCA1" → `["TP53", "BRCA1"]`
- Extract conditions: "in tumor samples only" → `{"condition": "Primary Tumor"}`
- Extract test parameters: "compare tumor and normal" → `{"groups": ["Primary Tumor", "Normal"]}`

**Who practices this**: Everyone extracts different parameters for their feature.

**Code example**:
```python
from utils.python.ollama_utils import ask_llm_to_extract

genes = ask_llm_to_extract(
    user_input,
    what_to_extract="gene names",
    example="TP53, BRCA1"
)
```

---

### 3. **Natural Language Generation**
**What**: Having the LLM convert structured data into readable explanations

**Examples**:
- Stats results → "TP53 shows significantly higher expression in tumor samples (p=0.023)"
- Gene info → "TP53 is a tumor suppressor gene that..."
- Error → "Sorry, I couldn't find that gene. Did you mean TP63?"

**Who practices this**: Tayler (stats summaries) and David (gene explanations).

**Code example**:
```python
from utils.python.ollama_utils import call_ollama

prompt = f"Explain these stats results in plain English: {stats_data}"
explanation = call_ollama(prompt, temperature=0.3)
```

---

### 4. **Structured Output (JSON)**
**What**: Getting the LLM to return data in a specific format (usually JSON)

**Examples**:
```json
{
  "genes": ["TP53", "BRCA1"],
  "plot_type": "heatmap",
  "filter": {"condition": "Primary Tumor"}
}
```

**Who practices this**: Qing (filters), Tayler (stats params), Miao (context resolution).

**Code example**:
```python
from utils.python.ollama_utils import call_ollama_json

prompt = """
Extract gene and plot type. Return as JSON.
Query: Show TP53 as violin plot
JSON:
"""

result = call_ollama_json(prompt)
# Returns: {"gene": "TP53", "plot_type": "violin"}
```

---

## 👥 Team Assignments

### **Zaki Wilmot + Udhayakumar Gopal** (R Team)
**Feature**: Multi-gene visualization with smart plot selection

**LLM Skills**:
- ✅ Parameter extraction: Extract multiple gene names
- ✅ Intent classification: Detect plot type preference
- ✅ Structured output: Return list of genes as JSON

**Module**: `modules/R/multi_gene_viz.R`

**Example queries they handle**:
- "Compare TP53, BRCA1, and EGFR" → Extract 3 genes → Heatmap
- "Show me TP53 as violin plot" → Extract gene + plot type

---

### **Qing Li** (Python)
**Feature**: Natural language data filtering

**LLM Skills**:
- ✅ Parameter extraction: Extract filter conditions
- ✅ Structured output: Return filter params as JSON
- ✅ Intent classification: Detect when filtering is requested

**Module**: `modules/python/llm_filters.py`

**Example queries they handle**:
- "Show TP53 in only tumor samples" → Filter to Primary Tumor
- "Display BRCA1 in normal tissue" → Filter to Normal
- "Plot EGFR excluding metastatic" → Exclude Metastatic

---

### **Miao Ai** (Python)
**Feature**: Conversational context/follow-ups

**LLM Skills**:
- ✅ Intent classification: Detect follow-up vs new query
- ✅ Parameter extraction: Resolve references ("it", "that gene")
- ✅ Context management: Track conversation state

**Module**: `modules/python/conversation.py`

**Example queries they handle**:
- User: "Show me TP53" → Bot: [shows plot]
- User: "Now show it as violin" → Resolves "it" = TP53
- User: "Compare it to BRCA1" → Knows "it" = TP53

---

### **Tayler Fearn** (Python)
**Feature**: Statistical testing with LLM

**LLM Skills**:
- ✅ Intent classification: Detect stats vs plot questions
- ✅ Parameter extraction: Extract gene, test type, groups
- ✅ Natural language generation: Explain results in plain English
- ✅ Structured output: Get test params as JSON

**Module**: `modules/python/llm_stats.py`

**Example queries they handle**:
- "Is TP53 significantly different between tumor and normal?" → T-test
- "Compare BRCA1 across all conditions" → ANOVA
- Returns: Natural language summary of p-value and significance

---

### **David Stone** (Python)
**Feature**: Gene information RAG (Retrieval-Augmented Generation)

**LLM Skills**:
- ✅ Intent classification: Detect info questions vs plot requests
- ✅ Parameter extraction: Extract gene name from question
- ✅ Natural language generation: Convert technical info to readable explanations
- ✅ RAG pattern: Retrieval (from database) + Generation (LLM explanation)

**Module**: `modules/python/llm_rag.py`

**Example queries they handle**:
- "What does TP53 do?" → Retrieve TP53 info → LLM explains
- "Tell me about BRCA1" → Fetch annotation → Generate explanation

---

## 🏗️ Project Architecture

```
chatseq/
├── app/
│   └── python/
│       ├── chatbot_base.py      # BACKUP VERSION (if integration fails)
│       └── chatbot.py            # MAIN VERSION (with feature flags)
│
├── modules/
│   ├── R/
│   │   └── multi_gene_viz.R      # Zaki + Udhaya
│   └── python/
│       ├── llm_filters.py        # Qing
│       ├── conversation.py       # Miao
│       ├── llm_stats.py          # Tayler
│       └── llm_rag.py            # David
│
├── utils/
│   ├── python/
│   │   ├── ollama_utils.py       # SHARED: Everyone uses these functions
│   │   ├── llm_utils.py          # Existing gene extraction
│   │   └── plot_utils.py         # Existing plotting
│   └── R/
│       └── ollama_utils.R        # SHARED: R team uses this
│
├── data/                         # Expression data
│
└── docs/                         # Documentation
    ├── HACKATHON_OVERVIEW.md     # This file
    ├── TASK_ASSIGNMENTS.md       # Detailed tasks per person
    ├── HACKATHON_SCHEDULE.md     # 7-hour timeline
    ├── GIT_WORKFLOW.md           # Branch strategy
    └── OLLAMA_SETUP.md           # Installation guide
```

---

## 🎓 Learning Philosophy

### **What You WILL Learn**

✅ How to call a local LLM from your code (Python/R)
✅ Prompt engineering basics
✅ Getting structured output from LLMs
✅ Handling LLM errors and variability
✅ When to use LLMs vs hardcoded logic

### **What LLMs Do in This Project**

- **Understand**: Parse natural language to extract meaning
- **Decide**: Choose which function to call, what plot type to use
- **Explain**: Generate human-readable summaries

### **What LLMs Do NOT Do**

- ❌ Calculate statistics (scipy/R does that - more reliable)
- ❌ Generate plotting code (we have hardcoded plot functions)
- ❌ Load data or manipulate dataframes (pandas/tidyverse does that)

**Philosophy**: **LLM as smart interface, hardcoded functions as reliable engine**

---

## 🚀 Success Metrics

By the end of the hackathon, each person should be able to answer:

1. ✅ "How do I call a local LLM from my code?"
2. ✅ "How do I write a prompt to extract specific information?"
3. ✅ "What are LLM limitations and how do I handle them?"
4. ✅ "Where could I use LLMs in my actual research work?"

**Plus**: You have a working demo showing 5 different LLM capabilities!

---

## 📊 Demo Structure (3pm Day 2)

### **5-Minute Demo Plan**

1. **Intro** (30 sec): Goal was to learn LLMs hands-on

2. **Live Demo** (3 min):
   - Basic: "Show me TP53" → Works!
   - Multi-gene: "Compare TP53, BRCA1, EGFR" → Heatmap
   - Filtering: "Show TP53 in tumor samples only" → Filtered plot
   - Follow-up: "Now as violin plot" → Conversation
   - Stats: "Is TP53 different between groups?" → P-value + explanation
   - Info: "What does TP53 do?" → RAG explanation

3. **Behind the Scenes** (1 min):
   - Show one person's LLM prompt/response
   - "This is how the LLM parsed the query"

4. **Lessons Learned** (30 sec):
   - What we learned about LLMs
   - What would we do differently
   - How we'd use this in real work

---

## 🎯 Key Takeaways

### **For Your Research**

After this hackathon, you'll be able to:

- Add LLM-powered interfaces to your pipelines
- Use LLMs for data exploration and querying
- Understand when LLMs add value vs when they're overkill
- Build your own custom LLM applications with local models

### **Technical Skills Gained**

- Prompt engineering
- Structured output parsing
- Error handling with LLMs
- Context management
- RAG patterns
- Intent classification

---

## 📚 Resources

- **Ollama Documentation**: https://ollama.com/
- **LLM Prompting Guide**: See `utils/python/ollama_utils.py` → `demonstrate_llm_skills()`
- **Module Examples**: Each module file has working examples
- **Shared Utils**: `utils/python/ollama_utils.py` and `utils/R/ollama_utils.R`

---

## 🙋 Getting Help

During the hackathon:

1. **Check module example code**: Each module has working examples at the bottom
2. **Test your module standalone**: Run `python modules/python/your_module.py`
3. **Ask David or Tayler**: They have LLM experience and can help debug
4. **Slack #llm-help channel**: Post questions there

---

## 🎉 Have Fun!

Remember: The goal is **learning**, not perfection. Even if features don't fully integrate, you'll still learn valuable LLM skills by building your module!
