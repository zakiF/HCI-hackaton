# ✅ Hackathon Setup Complete!

## What Has Been Done

All files and structure for the 7-hour LLM learning hackathon are ready.

---

## 📦 Created Files

### **Main Chatbot Files**
- ✅ `app/python/chatbot.py` - Main chatbot with feature flags
- ✅ `app/python/chatbot_base.py` - Backup version

### **Module Skeletons** (One per person/team)
- ✅ `modules/R/multi_gene_viz.R` - Zaki + Udhaya
- ✅ `modules/python/llm_filters.py` - Qing
- ✅ `modules/python/conversation.py` - Miao
- ✅ `modules/python/llm_stats.py` - Tayler
- ✅ `modules/python/llm_rag.py` - David

### **Shared Utilities**
- ✅ `utils/python/ollama_utils.py` - Python LLM helpers
- ✅ `utils/R/ollama_utils.R` - R LLM helpers

### **Documentation**
- ✅ `HACKATHON_README.md` - Quick start guide
- ✅ `docs/HACKATHON_OVERVIEW.md` - Goals and learning objectives
- ✅ `docs/TASK_ASSIGNMENTS.md` - Detailed tasks per person
- ✅ `docs/HACKATHON_SCHEDULE.md` - 7-hour timeline
- ✅ `docs/GIT_WORKFLOW.md` - Branch strategy and Git guide
- ✅ `docs/OLLAMA_SETUP.md` - Installation instructions

### **Git Branches**
- ✅ `feature/multi-gene` - Zaki + Udhaya
- ✅ `feature/filters` - Qing
- ✅ `feature/conversation` - Miao
- ✅ `feature/stats` - Tayler
- ✅ `feature/rag` - David

---

## 🎯 What Each File Does

### **Chatbot Files**

**`chatbot.py`** (Main version):
- Feature flags for each person's work
- Safe imports with try-except
- Placeholders showing where each person's code plugs in
- Routes: RAG → Stats → Plotting (with filters and conversation)

**`chatbot_base.py`** (Backup):
- Simple working chatbot with no experimental features
- Run this if integration fails completely
- Guaranteed to work for demo

### **Module Files**

Each module file contains:
- Function stubs with TODOs for the person
- Working example code they can use/modify
- Test cases at the bottom
- Comments explaining LLM learning goals
- Can be run standalone for testing

**Example structure**:
```python
# Main functions (person implements these)
def extract_something(user_input):
    # TODO: Write LLM prompt here
    pass

# Hardcoded helper functions (provided)
def apply_something(data, params):
    # Implementation provided
    pass

# Testing section
if __name__ == "__main__":
    # Test cases to run standalone
    pass
```

### **Utility Files**

**`utils/python/ollama_utils.py`**:
Provides shared functions everyone can use:
- `call_ollama()` - Basic LLM calling
- `call_ollama_json()` - Get JSON responses
- `ask_llm_to_classify()` - Classification helper
- `ask_llm_to_extract()` - Extraction helper
- `demonstrate_llm_skills()` - Shows examples of the 4 skills

**`utils/R/ollama_utils.R`**:
R version of the same, for Zaki + Udhaya

---

## 📚 Documentation Summary

### **HACKATHON_README.md** (Start here!)
- Quick overview and checklist
- Pre-hackathon setup steps
- Team assignments table
- Quick tips for LLM calling

### **HACKATHON_OVERVIEW.md** (Main goals)
- The 4 core LLM skills everyone learns
- Detailed explanation of each person's task
- Learning philosophy
- Demo structure

### **TASK_ASSIGNMENTS.md** (Detailed instructions)
- Step-by-step implementation guide per person
- Code examples
- Testing strategies
- Time management tips
- "Tips for [Person]" sections

### **HACKATHON_SCHEDULE.md** (Timeline)
- Hour-by-hour schedule for 7 hours
- Milestones for each session
- Integration plan (Day 2 11am)
- Success criteria and checklists

### **GIT_WORKFLOW.md** (Collaboration)
- Branch strategy explained
- How to commit and push
- Integration process
- Troubleshooting Git issues
- Cheat sheet of common commands

### **OLLAMA_SETUP.md** (Installation)
- How to install Ollama
- How to download llama3.2
- Testing instructions
- Troubleshooting common issues
- Day-of startup checklist

---

## 🎓 The Learning Structure

### **4 Core LLM Skills** (Everyone Learns):

1. **Intent Classification**
   - Categorizing user input
   - Example: plot vs stats vs question

2. **Parameter Extraction**
   - Pulling specific info from natural language
   - Example: gene names, filter conditions, test parameters

3. **Natural Language Generation**
   - Converting data to readable explanations
   - Example: Explaining stats results, gene functions

4. **Structured Output**
   - Getting LLM to return JSON/structured data
   - Example: `{"gene": "TP53", "plot_type": "violin"}`

### **How Each Person Practices These**:

**Zaki + Udhaya** (Multi-gene viz):
- ✅ Extraction (multiple gene names)
- ✅ Classification (plot type selection)
- ✅ Structured output (JSON list of genes)

**Qing** (Filters):
- ✅ Extraction (filter conditions)
- ✅ Structured output (filter params as JSON)
- ✅ Classification (detect filtering intent)

**Miao** (Conversation):
- ✅ Classification (follow-up vs new query)
- ✅ Extraction (resolve "it", "that" references)
- ✅ Context management

**Tayler** (Stats):
- ✅ Classification (stats vs plot)
- ✅ Extraction (gene, test, groups)
- ✅ Generation (explain results)
- ✅ Structured output (test params as JSON)

**David** (RAG):
- ✅ Classification (question vs plot)
- ✅ Extraction (gene from question)
- ✅ Generation (explain gene function)
- ✅ RAG pattern (retrieval + generation)

---

## 🚀 How It All Works Together

### **User Query Flow**:

```
User input: "Show TP53 in tumor samples only"
    ↓
[Route 1 Check] David's RAG: Is this a gene question? → No
    ↓
[Route 2 Check] Tayler's Stats: Is this a stats query? → No
    ↓
[Route 3: Plotting]
    ├─ Miao's Conversation: Resolve context → "Show TP53 in tumor samples only" (no change)
    ├─ Base: Extract gene → "TP53"
    ├─ Qing's Filters: Extract filter → {condition: "Primary Tumor", exclude: false}
    ├─ Apply filter → subset data
    └─ Zaki+Udhaya: Create plot → boxplot of TP53 in tumor samples
```

### **Feature Flags Control Integration**:

```python
# In chatbot.py
ENABLE_RAG = True            # David's feature
ENABLE_STATS = True          # Tayler's feature
ENABLE_FILTERS = True        # Qing's feature
ENABLE_CONVERSATION = True   # Miao's feature
ENABLE_MULTI_GENE = True     # Zaki+Udhaya's feature

# If integration breaks:
ENABLE_RAG = False  # Disable broken feature
# Chatbot still works with other features!
```

---

## 📅 Timeline Quick Reference

### **Pre-Hackathon** (Before Day 1):
- [ ] Everyone installs Ollama
- [ ] Everyone downloads llama3.2
- [ ] Everyone tests their setup
- [ ] Everyone reads HACKATHON_README.md

### **Day 1 Morning** (10am-12pm):
- 10:00-10:30: LLM workshop (all together)
- 10:30-10:45: Task assignment review
- 10:45-12:00: Start individual work

### **Day 1 Afternoon** (3pm-5pm):
- Continue feature development
- Test with different inputs
- Iterate on prompts

### **Day 2 Morning** (9am-12pm):
- 9:00-11:00: Finish features
- 11:00-12:00: **Integration** (critical!)

### **Day 2 Afternoon** (2pm-3pm):
- Demo prep and rehearsal

### **Day 2 3pm**:
- **Presentation!**

---

## 🎯 Success Metrics

### **Technical**:
- ✅ Base chatbot works
- ✅ Each person's module works standalone
- ✅ 3-5 features integrated (not all required)

### **Learning**:
Each person can answer:
- ✅ How do I call a local LLM?
- ✅ How do I write prompts for extraction/classification?
- ✅ What are LLM limitations?
- ✅ Where would I use LLMs in my research?

### **Demo**:
- ✅ 10-15 minute presentation
- ✅ Live demo of features
- ✅ Explanation of LLM concepts learned

---

## 🛠️ Testing Before Hackathon

### **You should test**:

1. **Ollama works**:
   ```bash
   ollama serve
   # In another terminal:
   ollama run llama3.2 "What is 2+2?"
   ```

2. **Base chatbot works**:
   ```bash
   streamlit run app/python/chatbot_base.py
   ```

3. **Python utils work**:
   ```python
   from utils.python.ollama_utils import call_ollama
   response = call_ollama("Test")
   print(response)
   ```

4. **R utils work** (for Zaki+Udhaya):
   ```r
   source("utils/R/ollama_utils.R")
   response <- call_ollama("Test")
   cat(response)
   ```

---

## 📦 What to Send to Team

### **Email to Team**:

**Subject**: Hackathon Setup - Action Required Before [Day 1 Date]

**Body**:
```
Hi team,

The hackathon structure is ready! Please complete these steps BEFORE we start:

1. **Install Ollama**: https://ollama.com/
   - Download and install for your OS
   - Run: ollama pull llama3.2

2. **Clone the repo**:
   git clone [REPO URL]
   cd HCI-hackaton

3. **Checkout your branch**:
   - Zaki + Udhaya: git checkout feature/multi-gene
   - Qing: git checkout feature/filters
   - Miao: git checkout feature/conversation
   - Tayler: git checkout feature/stats
   - David: git checkout feature/rag

4. **Install packages**:
   - Python: pip install streamlit pandas matplotlib requests scipy
   - R: install.packages(c("shiny", "httr", "jsonlite", "tidyverse", "ggplot2"))

5. **Read the docs** (in this order):
   - HACKATHON_README.md (quick overview)
   - docs/HACKATHON_OVERVIEW.md (goals and learning objectives)
   - docs/TASK_ASSIGNMENTS.md (your specific task)
   - docs/OLLAMA_SETUP.md (setup help if needed)

6. **Test everything works**:
   - ollama serve (keep running)
   - streamlit run app/python/chatbot_base.py

If you have ANY issues with setup, post in Slack NOW so we can help!

See you [Day 1 Date] at 10am!

Zaki
```

---

## 🎉 You're Done!

Everything is ready for a successful LLM learning hackathon.

### **Key Files for You to Review**:
1. `HACKATHON_README.md` - Send this to your team
2. `docs/HACKATHON_OVERVIEW.md` - Make sure goals match your vision
3. `docs/TASK_ASSIGNMENTS.md` - Review each person's assignment

### **Next Steps**:
1. Review all documentation
2. Test the base chatbot works
3. Send setup email to team
4. Confirm everyone can access the repo
5. Day of: Make sure everyone's Ollama is running!

---

**Ready to run a great hackathon!** 🚀🧬🤖
