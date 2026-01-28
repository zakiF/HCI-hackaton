# R Package Requirements for ChatSeq
# Install with: source("environment/requirements.R")

# List of required packages
required_packages <- c(
  "shiny",         # Web app framework
  "httr",          # HTTP requests to Ollama
  "jsonlite",      # JSON parsing
  "tidyverse",     # Data manipulation (includes dplyr, tidyr, ggplot2)
  "ggplot2"        # Plotting (included in tidyverse but listed explicitly)
)

# Function to install missing packages
install_if_missing <- function(packages) {
  new_packages <- packages[!(packages %in% installed.packages()[,"Package"])]

  if(length(new_packages) > 0) {
    cat("Installing missing packages:", paste(new_packages, collapse = ", "), "\n")
    install.packages(new_packages, dependencies = TRUE)
  } else {
    cat("All required packages are already installed!\n")
  }
}

# Install missing packages
install_if_missing(required_packages)

# Verify installation
cat("\nVerifying installation...\n")
for(pkg in required_packages) {
  if(require(pkg, character.only = TRUE, quietly = TRUE)) {
    cat("✓", pkg, "\n")
  } else {
    cat("✗", pkg, "FAILED\n")
  }
}

cat("\nSetup complete!\n")
