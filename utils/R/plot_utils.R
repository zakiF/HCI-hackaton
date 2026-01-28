##### Plotting Utility Functions for R #####
# Functions to create expression visualizations

library(tidyverse)
library(ggplot2)


#' Load Expression Data
#'
#' @param data_dir Path to data directory (default: "data")
#' @return List with expr_matrix (wide), expr_long (long format), and metadata
#' @export
load_expression_data <- function(data_dir = "data") {

  # Read normalized expression
  expr_matrix <- read.csv(
    file.path(data_dir, "Normalized_expression.csv"),
    row.names = 1,
    check.names = FALSE
  )

  # Read metadata
  metadata <- read.csv(
    file.path(data_dir, "Metadata.csv"),
    stringsAsFactors = FALSE
  )

  # Clean up group names for better display
  metadata <- metadata %>%
    mutate(
      condition = case_when(
        grepl("normal", group, ignore.case = TRUE) ~ "Normal",
        grepl("tumor", group, ignore.case = TRUE) ~ "Primary Tumor",
        grepl("met", group, ignore.case = TRUE) ~ "Metastatic",
        TRUE ~ group
      ),
      condition = factor(condition, levels = c("Normal", "Primary Tumor", "Metastatic"))
    )

  # Convert to long format for plotting
  expr_long <- expr_matrix %>%
    rownames_to_column("gene") %>%
    pivot_longer(
      cols = -gene,
      names_to = "Run",
      values_to = "expression"
    ) %>%
    left_join(metadata, by = "Run")

  return(list(
    expr_matrix = expr_matrix,
    expr_long = expr_long,
    metadata = metadata
  ))
}


#' Plot Gene Expression Boxplot
#'
#' @param gene_name Character string with gene name (case-sensitive)
#' @param expr_data List from load_expression_data()
#' @return ggplot object, or NULL if gene not found
#' @export
plot_gene_boxplot <- function(gene_name, expr_data) {

  # Filter for the specific gene
  gene_data <- expr_data$expr_long %>%
    filter(gene == gene_name)

  # Check if gene exists
  if (nrow(gene_data) == 0) {
    warning(paste("Gene", gene_name, "not found in dataset"))
    return(NULL)
  }

  # Create boxplot
  p <- ggplot(gene_data, aes(x = condition, y = expression, fill = condition)) +
    geom_boxplot(alpha = 0.7, outlier.shape = NA) +
    geom_jitter(width = 0.2, alpha = 0.5, size = 2) +
    scale_fill_manual(
      values = c(
        "Normal" = "#66C2A5",
        "Primary Tumor" = "#FC8D62",
        "Metastatic" = "#8DA0CB"
      )
    ) +
    labs(
      title = paste("Expression of", gene_name),
      subtitle = paste(nrow(gene_data), "samples across 3 conditions"),
      x = "Condition",
      y = "Normalized Expression (log2)",
      fill = "Condition"
    ) +
    theme_minimal() +
    theme(
      plot.title = element_text(size = 16, face = "bold"),
      plot.subtitle = element_text(size = 12, color = "gray40"),
      axis.title = element_text(size = 12),
      axis.text.x = element_text(size = 11, angle = 0),
      axis.text.y = element_text(size = 11),
      legend.position = "none"
    )

  return(p)
}


#' Get Available Genes
#'
#' @param expr_data List from load_expression_data()
#' @return Character vector of gene names
#' @export
get_available_genes <- function(expr_data) {
  return(sort(unique(expr_data$expr_long$gene)))
}


#' Search for Genes by Pattern
#'
#' @param pattern Character string to search for (case-insensitive)
#' @param expr_data List from load_expression_data()
#' @param max_results Maximum number of results to return
#' @return Character vector of matching gene names
#' @export
search_genes <- function(pattern, expr_data, max_results = 10) {
  all_genes <- get_available_genes(expr_data)
  matches <- grep(pattern, all_genes, ignore.case = TRUE, value = TRUE)
  return(head(matches, max_results))
}
