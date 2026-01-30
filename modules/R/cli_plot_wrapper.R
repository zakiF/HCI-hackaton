#!/usr/bin/env Rscript
# CLI wrapper to call plot_genes_save_png from the command line.
# Usage:
# Rscript cli_plot_wrapper.R "TP53,BRCA1" expr.csv metadata.csv out.png 10 6 150 group

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 4) {
  stop("Usage: cli_plot_wrapper.R <genes_comma_sep> <expr_csv> <metadata_csv> <out_png> [width] [height] [dpi] [group_col]")
}

genes_str <- args[1]
expr_csv <- args[2]
meta_csv <- args[3]
out_png <- args[4]
width <- ifelse(length(args) >= 5, as.numeric(args[5]), 10)
height <- ifelse(length(args) >= 6, as.numeric(args[6]), 6)
dpi <- ifelse(length(args) >= 7, as.numeric(args[7]), 150)
group_col <- ifelse(length(args) >= 8, args[8], "group")

# Parse genes
genes <- unlist(strsplit(genes_str, ","))
genes <- trimws(genes)

# Find script dir (robust when called via Rscript)
arg_vals <- commandArgs(trailingOnly = FALSE)
file_arg <- grep("--file=", arg_vals)
if (length(file_arg) > 0) {
  script_path <- sub("--file=", "", arg_vals[file_arg])
  script_dir <- dirname(normalizePath(script_path))
} else {
  # Fallback to current working directory
  script_dir <- getwd()
}

# Source the multi-gene module
source(file.path(script_dir, "multi_gene_viz.R"))

# Read inputs
expr_matrix <- read.csv(expr_csv, row.names = 1, check.names = FALSE)
metadata <- read.csv(meta_csv, check.names = FALSE)

# Call the saving wrapper (handles printing any errors)
tryCatch({
  plot_genes_save_png(genes, expr_matrix, metadata, out_png, width = width, height = height, dpi = dpi, group_col = group_col)
  cat("OK\n")
}, error = function(e) {
  cat("ERROR: ", conditionMessage(e), "\n", sep = "")
  quit(status = 2)
})
