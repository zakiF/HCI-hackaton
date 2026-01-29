# ChatSeq  
**Interactive Visualization of Gene and Protein Expression Data Using a Local LLM Chatbot**

## Overview
**ChatSeq** is a hackathon project that explores how a local large language model (LLM) can be used as a natural-language interface for exploring gene and protein expression data.

The goal is to enable users to interact with expression datasets using plain English and receive meaningful visualizations and analyses in return.

### Example queries
- “Show me the expression of TP53 across conditions”
- “Plot the top 20 most variable genes”
- “Run PCA on this dataset”

The system uses a local LLM to interpret user intent and route requests to predefined bioinformatics workflows written in **R** and/or **Python**.

---

## Example Dataset
ChatSeq uses an example RNA-seq dataset from a published colorectal cancer study:

> **Source:**  
> Molecular Oncology (2014)  
> https://febs.onlinelibrary.wiley.com/doi/10.1016/j.molonc.2014.06.016

### Dataset description
The dataset contains colorectal cancer RNA-seq samples from **three biological conditions**:
- **Normal**
- **Primary tumor**
- **Metastatic**

### Data files
All data files are located in the `data/` directory:
- **Normalized expression (.csv)** – used for **Aim 1 (Visualization)**
- **Raw counts (.csv)** – used for **Aim 2 (Analysis)**
- **Metadata (.csv)** – sample annotations (Normal / Tumor / Metastatic)

---

## Project Goals
This project is designed for a **1.5-day hackathon**.

We define **two main goals**, each powered by a chatbot interface using a local LLM.

---

## Goal 1: Visualization (Core Goal)
**Primary deliverable for the hackathon**

Build a chatbot that allows users to visualize expression data without requiring prior RNA-seq expertise.

### Supported visualization tasks
- **Gene-level expression plots**
  - Boxplots 
  - Bar plots 
- **Heatmaps**
  - Top expressed genes
  - Top variable genes
- **Interactive plots** 

This goal ensures that everyone on the team can participate and contribute.

---

## Goal 2: Analysis (Stretch Goal)
**Implemented if time allows**

Extend the chatbot to trigger standard expression analysis workflows.

### Planned analysis features
- Data loading and normalization (from raw counts)
- Principal Component Analysis (PCA)
- Differential expression analysis (DEGs)
- Pathway enrichment analysis
- Clustering (genes or samples)

All analyses are executed using established bioinformatics tools, with the LLM acting as a **controller**, not a replacement for statistical methods.

---

## 🚀 Quick Start

### Prerequisites

1. **Install Ollama:** https://ollama.com/
2. **Pull a model:** `ollama pull llama3.2`
3. **Start Ollama:** `ollama serve` (keep running)

4. **(Optional) R bridge:** To enable calling R plotting from Python install `rpy2` (`pip install rpy2`) and ensure the following R packages are available: `ggplot2`, `tidyr`, `dplyr`, `jsonlite`.

Alternatively, the app can call R via `Rscript` (no `rpy2` required). Ensure `Rscript` is on your PATH and install R packages with `source("environment/requirements.R")`.

### Run the Chatbot

**Python (Streamlit):**
```bash
pip install -r environment/requirements.txt
streamlit run app/python/chatbot_basic.py
```

**R (Shiny):**
```r
source("environment/requirements.R")
shiny::runApp("app/R/chatbot_basic.R")
```

📖 **See [QUICKSTART.md](QUICKSTART.md) for detailed instructions!**

---

## 📂 Project Structure

```
chatseq/
├── README.md               ← Project overview (you are here)
├── QUICKSTART.md           ← How to run the chatbots
├── data/                   ← RNA-seq data files
│   ├── Normalized_expression.csv
│   ├── Raw_counts.csv
│   └── Metadata.csv
├── app/
│   ├── R/
│   │   └── chatbot_basic.R        ← R Shiny chatbot
│   └── python/
│       └── chatbot_basic.py       ← Python Streamlit chatbot
├── utils/
│   ├── R/
│   │   ├── llm_utils.R            ← R functions for Ollama
│   │   └── plot_utils.R           ← R plotting functions
│   └── python/
│       ├── llm_utils.py           ← Python functions for Ollama
│       └── plot_utils.py          ← Python plotting functions
├── visualization/          ← Future: Extended viz features
├── analysis/               ← Future: Analysis workflows
├── llm/                    ← Future: Advanced prompts
└── environment/
    ├── requirements.txt    ← Python packages
    └── requirements.R      ← R packages
```

---

## 💡 Current Features (v0.1 - Basic Skeleton)

The current implementation provides a **minimal working chatbot** with:

✅ Natural language gene queries (e.g., "Show me TP53 expression")
✅ LLM-powered gene name extraction
✅ Boxplot visualization across 3 conditions
✅ Both R (Shiny) and Python (Streamlit) versions
✅ Clean, extensible codebase for hackathon

**What it does:**
- User asks: *"Show me expression of TP53"*
- LLM extracts: `TP53`
- App plots: Boxplot across Normal, Tumor, Metastatic

---

## 🛠️ How to Extend (For Hackathon Participants)

### Add More Visualizations

**Example: Add violin plots**

1. Add function to `utils/R/plot_utils.R` or `utils/python/plot_utils.py`
2. Update chatbot to detect "violin plot" in user query
3. Call your new function

### Add Multiple Gene Comparison

1. Modify LLM prompt to extract multiple genes
2. Create heatmap or multi-panel plot
3. Update UI to show results

### Add Statistical Tests

1. Add analysis functions (t-test, ANOVA)
2. Display p-values alongside plots
3. Add interpretation text

---

## 🤝 Contributing

This is a hackathon project! Feel free to:
- Add new visualization types
- Implement analysis workflows (PCA, DEG)
- Improve LLM prompts
- Enhance the UI
- Add documentation

**Workflow:**
1. Fork the repo
2. Create a feature branch
3. Make your changes
4. Test both R and Python versions (if applicable)
5. Submit a pull request

---

## 📚 Resources

### LLM Integration
- Ollama: https://ollama.com/
- Ollama API docs: https://github.com/ollama/ollama/blob/main/docs/api.md

### R Packages
- Shiny: https://shiny.rstudio.com/
- ggplot2: https://ggplot2.tidyverse.org/

### Python Packages
- Streamlit: https://docs.streamlit.io/
- Matplotlib: https://matplotlib.org/

### Bioinformatics
- DESeq2 (R): https://bioconductor.org/packages/DESeq2/
- Scanpy (Python): https://scanpy.readthedocs.io/

### BulkRNA-seq LLM
- Made by Shivaprasad Patil: https://www.linkedin.com/feed/update/urn:li:activity:7418265230620725248/
- Github: https://lnkd.in/eP5_UGPG

---

## 📄 License

This project is for educational purposes (hackathon). Feel free to use and modify!

---

## 🙋 Questions?

See [QUICKSTART.md](QUICKSTART.md) for troubleshooting or open an issue!


