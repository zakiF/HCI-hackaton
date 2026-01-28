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


