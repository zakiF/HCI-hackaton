# 🚀 ChatSeq Quick Start Guide

Get the basic chatbot running in **5 minutes**!

---

## Prerequisites

### 1. Install Ollama

Visit https://ollama.com/ and install for your OS.

### 2. Download and Start LLM

```bash
# Pull the model (one time, ~2-4GB download)
ollama pull llama3.2

# Start Ollama server (keep this running!)
ollama serve
```

---

## Option 1: Python Chatbot (Streamlit)

### Install Dependencies

```bash
pip install streamlit pandas matplotlib requests
# python3 -m pip install streamlit pandas matplotlib requests
```

### Run the Chatbot

```bash
cd /path/to/chatseq
streamlit run app/python/chatbot_basic.py
# python3 -m streamlit run app/python/chatbot_basic.py
```

Opens at `http://localhost:8501`

---

## Option 2: R Chatbot (Shiny)

### Install Dependencies

```r
install.packages(c("shiny", "httr", "jsonlite", "tidyverse", "ggplot2"))
```

### Run the Chatbot

```r
# In R or RStudio
shiny::runApp("app/R/chatbot_basic.R")
```

Opens at `http://127.0.0.1:XXXX`

---

## How to Use

### Example Queries

```
Show me expression of TP53
Plot BRCA1 expression
Display gene A1CF across conditions
```

The chatbot will:
1. Extract the gene name from your question
2. Plot a boxplot showing expression across:
   - Normal
   - Primary Tumor
   - Metastatic

---

## Troubleshooting

### "Can't connect to Ollama"

**Solution:** Make sure Ollama is running
```bash
ollama serve
```

### "Gene not found"

**Solution:** Gene names are case-sensitive. Try:
- TP53 (uppercase)
- Check available genes in `data/Normalized_expression.csv`

### Python: "Module not found"

```bash
pip install streamlit pandas matplotlib requests
```

### R: "Package not found"

```r
install.packages(c("shiny", "httr", "jsonlite", "tidyverse"))
```

---

## What's Next?

This is a **minimal skeleton**. Hackathon participants can extend it by:

- Adding more plot types (heatmaps, violin plots)
- Supporting multiple genes at once
- Adding statistical tests
- Implementing the analysis workflows (PCA, DEG)

See the main `README.md` for project goals and structure!

---

## File Structure

```
chatseq/
├── QUICKSTART.md           ← You are here
├── README.md               ← Project overview
├── data/                   ← Expression data
├── app/
│   ├── R/chatbot_basic.R        ← R Shiny app
│   └── python/chatbot_basic.py  ← Python Streamlit app
└── utils/
    ├── R/                  ← R helper functions
    └── python/             ← Python helper functions
```

Ready to hack! 🎉
