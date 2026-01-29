##### Multi-Gene Visualization Module #####
# Team Members: Zaki Wilmot + Udhayakumar Gopal
#
# LLM Learning Goals:
# 1. Parameter extraction: Extract multiple gene names from natural language
# 2. Intent classification: Detect when user wants multi-gene comparison
# 3. Structured output: Get LLM to return list of genes as JSON
# 4. Decision making: LLM chooses appropriate plot type
#
# Examples:
# - "Compare TP53, BRCA1, and EGFR" → Extract 3 genes → Use heatmap
# - "Show me TP53 as violin plot" → Extract 1 gene + plot type
# - "Plot expression of MYC and KRAS" → Extract 2 genes

library(ggplot2)
library(tidyverse)
library(httr)
library(jsonlite)

# Source utilities
source_path <- file.path(dirname(sys.frame(1)$ofile), "..", "..", "utils", "R", "ollama_utils.R")
source(source_path)


# ============================================
# Gene Name Extraction
# ============================================

#' Extract Gene Names from User Query
#'
#' Use LLM to extract one or more gene names from natural language.
#'
#' @param user_input Character string, user's query
#' @return Character vector of gene names (uppercase)
#'
#' @examples
#' genes <- extract_gene_names("Compare TP53, BRCA1, and EGFR")
#' print(genes)  # c("TP53", "BRCA1", "EGFR")
#'
#' @details
#' TODO for Zaki + Udhaya:
#' 1. Write LLM prompt to extract ALL gene names from query
#' 2. Handle various formats: "TP53 and BRCA1", "TP53, BRCA1, EGFR", etc.
#' 3. Use call_ollama_json() to get structured list
#' 4. Clean and validate gene names
extract_gene_names <- function(user_input) {

  # Prompt for LLM
  prompt <- paste0(
    "Extract ALL gene names from this query.\n",
    "Return ONLY valid JSON, nothing else. No explanation, no markdown, no code blocks.\n",
    "Format: {\"genes\": [\"GENE1\", \"GENE2\"]}\n\n",
    "Query: ", user_input, "\n\n",
    "JSON:"
  )

  # Call LLM
  result <- call_ollama_json(prompt, temperature = 0.1)

  if (is.null(result)) {
    warning("LLM failed to extract genes")
    return(character(0))
  }

  # Extract genes from JSON
  genes <- result$genes

  if (is.null(genes) || length(genes) == 0) {
    return(character(0))
  }

  # Clean up gene names
  genes <- toupper(trimws(genes))

  # Remove empty strings
  genes <- genes[genes != ""]

  return(genes)
}


# ============================================
# Plot Type Selection
# ============================================

#' Detect Requested Plot Type
#'
#' Use LLM to determine which type of plot the user wants.
#'
#' @param user_input Character string, user's query
#' @param num_genes Integer, how many genes (helps LLM decide)
#' @return Character string: "boxplot", "violin", "heatmap", or "barplot"
#'
#' @examples
#' plot_type <- detect_plot_type("Show me TP53 as violin plot", 1)
#' print(plot_type)  # "violin"
#'
#' @details
#' TODO for Zaki + Udhaya:
#' 1. Write LLM prompt to detect plot type from query
#' 2. Provide options: boxplot, violin, heatmap, barplot
#' 3. Use num_genes as context (e.g., 3+ genes → suggest heatmap)
#' 4. Default to sensible choice if not specified
detect_plot_type <- function(user_input, num_genes = 1) {

  prompt <- paste0(
    "What type of plot does the user want?\n",
    "Context: ", num_genes, " gene(s) to plot.\n\n",
    "Choose ONE from: boxplot, violin, heatmap, barplot\n",
    "If not specified, suggest:\n",
    "- 1 gene → boxplot or violin\n",
    "- 2-3 genes → heatmap\n",
    "- 4+ genes → heatmap\n\n",
    "Return ONLY the plot type name.\n\n",
    "Query: ", user_input, "\n\n",
    "Plot type:"
  )

  response <- call_ollama(prompt, temperature = 0.1)

  if (is.null(response)) {
    # Fallback logic
    if (num_genes == 1) {
      return("boxplot")
    } else {
      return("heatmap")
    }
  }

  # Clean response
  plot_type <- tolower(trimws(response))

  # Validate
  valid_types <- c("boxplot", "violin", "heatmap", "barplot")

  for (valid in valid_types) {
    if (grepl(valid, plot_type)) {
      return(valid)
    }
  }

  # Default fallback
  return(if (num_genes == 1) "boxplot" else "heatmap")
}


# ============================================
# Plotting Functions (Hardcoded)
# ============================================

#' Create Boxplot for Single Gene
#'
#' @param gene_name Character string, gene to plot
#' @param expr_data Data frame with expression values (genes × samples)
#' @param metadata Data frame with sample conditions
#' @return ggplot object
plot_gene_boxplot <- function(gene_name, expr_data, metadata) {

  # Get expression for this gene
  gene_expr <- as.numeric(expr_data[gene_name, ])

  # Combine with metadata
  plot_data <- data.frame(
    sample = colnames(expr_data),
    expression = gene_expr,
    condition = metadata$condition
  )

  # Create ggplot
  p <- ggplot(plot_data, aes(x = condition, y = expression, fill = condition)) +
    geom_boxplot(alpha = 0.7, outlier.shape = NA) +
    geom_jitter(width = 0.2, alpha = 0.5, size = 2) +
    labs(
      title = paste0(gene_name, " Expression Across Conditions"),
      x = "Condition",
      y = "Expression (TPM)",
      fill = "Condition"
    ) +
    theme_minimal() +
    theme(
      plot.title = element_text(size = 16, face = "bold", hjust = 0.5),
      axis.title = element_text(size = 12),
      legend.position = "right"
    ) +
    scale_fill_brewer(palette = "Set2")

  return(p)
}


#' Create Violin Plot for Single Gene
#'
#' @param gene_name Character string
#' @param expr_data Data frame
#' @param metadata Data frame
#' @return ggplot object
plot_gene_violin <- function(gene_name, expr_data, metadata) {

  # Get expression
  gene_expr <- as.numeric(expr_data[gene_name, ])

  plot_data <- data.frame(
    expression = gene_expr,
    condition = metadata$condition
  )

  # Create violin plot
  p <- ggplot(plot_data, aes(x = condition, y = expression, fill = condition)) +
    geom_violin(alpha = 0.7) +
    geom_jitter(width = 0.1, alpha = 0.5, size = 1.5) +
    labs(
      title = paste0(gene_name, " Expression (Violin Plot)"),
      x = "Condition",
      y = "Expression (TPM)"
    ) +
    theme_minimal() +
    scale_fill_brewer(palette = "Set2")

  return(p)
}


#' Create Heatmap for Multiple Genes
#'
#' @param gene_list Character vector of gene names
#' @param expr_data Data frame
#' @param metadata Data frame
#' @return ggplot object
plot_genes_heatmap <- function(gene_list, expr_data, metadata) {

  # Subset expression data
  subset_expr <- expr_data[gene_list, , drop = FALSE]

  # Convert to long format for ggplot
  expr_long <- subset_expr %>%
    as.data.frame() %>%
    rownames_to_column("gene") %>%
    pivot_longer(
      cols = -gene,
      names_to = "sample",
      values_to = "expression"
    )

  # Add condition info
  expr_long <- expr_long %>%
    left_join(
      metadata %>%
        rownames_to_column("sample") %>%
        select(sample, condition),
      by = "sample"
    )

  # Order samples by condition
  expr_long$sample <- factor(
    expr_long$sample,
    levels = rownames(metadata)[order(metadata$condition)]
  )

  # Create heatmap
  p <- ggplot(expr_long, aes(x = sample, y = gene, fill = expression)) +
    geom_tile() +
    scale_fill_gradient2(
      low = "blue",
      mid = "white",
      high = "red",
      midpoint = median(expr_long$expression),
      name = "Expression"
    ) +
    labs(
      title = paste0("Expression Heatmap (", length(gene_list), " genes)"),
      x = "Samples",
      y = "Genes"
    ) +
    theme_minimal() +
    theme(
      axis.text.x = element_blank(),  # Too many samples to show names
      axis.ticks.x = element_blank(),
      plot.title = element_text(size = 14, face = "bold", hjust = 0.5)
    ) +
    # Add condition annotation at top
    facet_grid(. ~ condition, scales = "free_x", space = "free")

  return(p)
}


# ============================================
# Main Smart Plotting Function
# ============================================

#' Smart Gene Plotting with LLM
#'
#' Automatically extract genes, choose plot type, and create visualization.
#'
#' @param user_query Character string, user's natural language query
#' @param expr_data Data frame with expression (genes × samples)
#' @param metadata Data frame with sample info
#' @return ggplot object
#'
#' @examples
#' plot <- plot_genes_smart("Compare TP53, BRCA1, EGFR", expr_data, metadata)
#' print(plot)
#'
#' @details
#' This is the main function that orchestrates LLM-powered plotting:
#' 1. Extract gene names using LLM
#' 2. Detect plot type using LLM
#' 3. Route to appropriate plotting function
#' 4. Return the plot
plot_genes_smart <- function(user_query, expr_data, metadata) {

  # Step 1: Extract genes
  cat("Extracting gene names from query...\n")
  genes <- extract_gene_names(user_query)

  if (length(genes) == 0) {
    stop("Could not extract any gene names from the query")
  }

  cat("Found", length(genes), "gene(s):", paste(genes, collapse = ", "), "\n")

  # Validate genes exist
  available_genes <- rownames(expr_data)
  missing_genes <- genes[!genes %in% available_genes]

  if (length(missing_genes) > 0) {
    warning("These genes not found: ", paste(missing_genes, collapse = ", "))
    genes <- genes[genes %in% available_genes]
  }

  if (length(genes) == 0) {
    stop("None of the requested genes are in the dataset")
  }

  # Step 2: Detect plot type
  cat("Detecting requested plot type...\n")
  plot_type <- detect_plot_type(user_query, length(genes))
  cat("Using plot type:", plot_type, "\n")

  # Step 3: Create appropriate plot
  if (length(genes) == 1) {
    # Single gene - boxplot or violin
    if (plot_type == "violin") {
      return(plot_gene_violin(genes[1], expr_data, metadata))
    } else {
      return(plot_gene_boxplot(genes[1], expr_data, metadata))
    }
  } else {
    # Multiple genes - heatmap
    return(plot_genes_heatmap(genes, expr_data, metadata))
  }
}


# ============================================
# Testing / Development
# ============================================

if (interactive()) {
  cat("=== Multi-Gene Visualization Module ===\n\n")

  cat("Tips for Zaki + Udhayakumar:\n")
  cat("1. Test extract_gene_names() with different phrasings\n")
  cat("2. Iterate on LLM prompts to improve accuracy\n")
  cat("3. Add more plot types if time permits (barplot, etc.)\n")
  cat("4. Consider adding plot customization options\n")
  cat("5. Test with edge cases: typos, ambiguous gene names\n\n")

  cat("Example usage:\n")
  cat("  genes <- extract_gene_names('Compare TP53 and BRCA1')\n")
  cat("  plot <- plot_genes_smart('Show me TP53 as violin', expr_data, metadata)\n")
}
