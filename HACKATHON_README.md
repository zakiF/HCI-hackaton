# ChatSeq Hackathon - Quick Start

## 🚀 You're All Set!

Everything is ready for the 7-hour LLM learning hackathon.

---

## 📚 Read These First (In Order)

1. **[HACKATHON_OVERVIEW.md](docs/HACKATHON_OVERVIEW.md)** ← Start here!
   - Main goals and learning objectives
   - Team assignments
   - The 4 core LLM skills everyone will learn

2. **[TASK_ASSIGNMENTS.md](docs/TASK_ASSIGNMENTS.md)**
   - Detailed step-by-step instructions for your task
   - Code examples and testing strategies

3. **[HACKATHON_SCHEDULE.md](docs/HACKATHON_SCHEDULE.md)**
   - 7-hour timeline (Day 1-2)
   - What to do when
   - Milestones and checkpoints

4. **[OLLAMA_SETUP.md](docs/OLLAMA_SETUP.md)** ⚠️ DO BEFORE DAY 1!
   - Install Ollama
   - Download llama3.2 model
   - Test your setup

5. **[GIT_WORKFLOW.md](docs/GIT_WORKFLOW.md)**
   - Branch strategy
   - How to commit and push
   - Integration plan

---

## 🎯 Before Hackathon Starts

### **Everyone Must Do** (30 minutes)

1. **Install Ollama** and download model:
   ```bash
   # Install from: https://ollama.com/
   ollama pull llama3.2
   ollama serve  # Keep running!
   ```

2. **Clone repository**:
   ```bash
   git clone https://github.com/YOUR-REPO/HCI-hackaton.git
   cd HCI-hackaton
   ```

3. **Checkout your branch**:
   - **Zaki + Udhaya**: `git checkout feature/multi-gene`
   - **Qing**: `git checkout feature/filters`
   - **Miao**: `git checkout feature/conversation`
   - **Tayler**: `git checkout feature/stats`
   - **David**: `git checkout feature/rag`

4. **Install packages**:
   ```bash
   # Python
   pip install streamlit pandas matplotlib requests scipy

   # R (in R console)
   install.packages(c("shiny", "httr", "jsonlite", "tidyverse", "ggplot2"))
   ```

5. **Test everything works**:
   ```bash
   # Test Ollama
   ollama run llama3.2 "What is 2+2?"

   # Test chatbot
   streamlit run app/python/chatbot_base.py
   ```

---

## 👥 Team Assignments Quick Reference

| Person | Branch | Module File | LLM Skills |
|--------|--------|-------------|------------|
| **Zaki + Udhaya** | `feature/multi-gene` | `modules/R/multi_gene_viz.R` | Extraction, Classification |
| **Qing** | `feature/filters` | `modules/python/llm_filters.py` | Extraction, JSON |
| **Miao** | `feature/conversation` | `modules/python/conversation.py` | Context, Extraction |
| **Tayler** | `feature/stats` | `modules/python/llm_stats.py` | Classification, Generation |
| **David** | `feature/rag` | `modules/python/llm_rag.py` | RAG, Generation |

---

## 📂 Project Structure

```
chatseq/
├── HACKATHON_README.md        ← You are here
├── docs/
│   ├── HACKATHON_OVERVIEW.md   ← Read first!
│   ├── TASK_ASSIGNMENTS.md     ← Your detailed tasks
│   ├── HACKATHON_SCHEDULE.md   ← 7-hour timeline
│   ├── OLLAMA_SETUP.md         ← Setup guide
│   └── GIT_WORKFLOW.md         ← Git instructions
│
├── app/python/
│   ├── chatbot.py              ← Main chatbot (with features)
│   └── chatbot_base.py         ← Backup (if integration fails)
│
├── modules/
│   ├── R/
│   │   └── multi_gene_viz.R    ← Zaki + Udhaya work here
│   └── python/
│       ├── llm_filters.py      ← Qing works here
│       ├── conversation.py     ← Miao works here
│       ├── llm_stats.py        ← Tayler works here
│       └── llm_rag.py          ← David works here
│
├── utils/
│   ├── python/
│   │   ├── ollama_utils.py     ← SHARED: Everyone uses this!
│   │   ├── llm_utils.py        ← Existing gene extraction
│   │   └── plot_utils.py       ← Existing plotting
│   └── R/
│       └── ollama_utils.R      ← SHARED: R team uses this!
│
└── data/                       ← Expression data
```

---

## 🧪 Testing Your Module

Each module file can run standalone:

```bash
# Python modules
python modules/python/llm_filters.py
python modules/python/llm_stats.py
python modules/python/llm_rag.py
python modules/python/conversation.py

# R module (in R console)
source("modules/R/multi_gene_viz.R")
```

---

## 💡 Quick Tips

### **LLM Calling**

```python
# Python
from utils.python.ollama_utils import call_ollama

response = call_ollama("Extract gene name from: Show me TP53")
print(response)  # "TP53"
```

```r
# R
source("utils/R/ollama_utils.R")

response <- call_ollama("Extract gene name from: Show me TP53")
cat(response)  # "TP53"
```

### **Getting Structured Output**

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

### **Helpful Utility Functions**

```python
from utils.python.ollama_utils import (
    call_ollama,           # Basic LLM call
    call_ollama_json,      # Get JSON response
    ask_llm_to_classify,   # Classify into categories
    ask_llm_to_extract     # Extract specific info
)
```

---

## 🎓 The 4 LLM Skills You'll Learn

1. **Intent Classification**: "Is this a plot, stats, or question?"
2. **Parameter Extraction**: Pull out genes, filters, groups from text
3. **Natural Language Generation**: Convert data to explanations
4. **Structured Output**: Get LLM to return JSON

**Everyone practices these** through their specific feature!

---

## 📅 Hackathon Timeline (Super Quick)

### **Day 1: 10am-12pm**
- LLM workshop (30 min)
- Start your feature (~50% done)

### **Day 1: 3pm-5pm**
- Continue feature development
- Test with different inputs

### **Day 2: 9am-12pm**
- Finish features (9am-11am)
- **Integration** (11am-12pm) ← Critical!

### **Day 2: 2pm-3pm**
- Demo prep and rehearsal

### **Day 2: 3pm**
- **Presentation!** 🎉

---

## 🆘 Getting Help

### **Before Hackathon**
- Slack: Post setup questions
- Test everything works **before Day 1**!

### **During Hackathon**
- **David + Tayler**: LLM experts, can help debug
- **Slack #llm-help**: Post questions
- **Module examples**: Check bottom of each module file

---

## ✅ Pre-Hackathon Checklist

**Complete by Day 1 10am**:

- [ ] Ollama installed (`ollama --version` works)
- [ ] llama3.2 downloaded (`ollama list` shows it)
- [ ] Ollama running (`ollama serve` in one terminal)
- [ ] Repository cloned
- [ ] Your branch checked out (`git checkout feature/YOUR-FEATURE`)
- [ ] Python/R packages installed
- [ ] Test chatbot works (`streamlit run app/python/chatbot_base.py`)
- [ ] Read HACKATHON_OVERVIEW.md
- [ ] Read your section in TASK_ASSIGNMENTS.md

---

## 🎯 Success Criteria

**By end of hackathon, you should be able to**:

1. ✅ Call a local LLM from your code
2. ✅ Write prompts to extract information
3. ✅ Handle LLM errors and variability
4. ✅ Explain when/where to use LLMs

**Plus**: Have a working module that demonstrates LLM capabilities!

---

## 🎉 Final Notes

- **Focus on learning**, not perfection
- **Test early, test often**
- **Ask for help** when stuck
- **Have fun!** This is hands-on learning

---

## 📞 Contact

**Questions before hackathon?**
- Post in Slack
- Email Zaki

**Ready to learn LLMs?** See you at 10am! 🚀

---

## 🔗 Quick Links

- **Ollama**: https://ollama.com/
- **Ollama Docs**: https://ollama.com/docs
- **Project Repo**: https://github.com/YOUR-REPO/HCI-hackaton

---

**Let's build something cool and learn LLMs together!** 🧬🤖
