##### Shared Ollama Utility Functions for R #####
# All team members can source and use these functions
#
# This provides a consistent interface for calling the local LLM.

library(httr)
library(jsonlite)


# ============================================
# Configuration
# ============================================

OLLAMA_URL <- "http://localhost:11434"
DEFAULT_MODEL <- "llama3.2"  # Everyone uses this model


# ============================================
# Basic LLM Calling
# ============================================

#' Call Ollama LLM
#'
#' Send a prompt to the local Ollama LLM and get a text response.
#'
#' @param prompt Character string with the question/instruction
#' @param model Character string, which model to use (default: llama3.2)
#' @param temperature Numeric 0-1, how creative (0 = focused, 1 = creative)
#' @param max_tokens Integer, maximum length of response
#' @return Character string with LLM response, or NULL on error
#' @export
#'
#' @examples
#' response <- call_ollama("What is the capital of France?")
#' print(response)  # "Paris"
call_ollama <- function(prompt,
                       model = DEFAULT_MODEL,
                       temperature = 0.1,
                       max_tokens = 500) {

  url <- paste0(OLLAMA_URL, "/api/generate")

  # Prepare request data
  data <- list(
    model = model,
    prompt = prompt,
    stream = FALSE,
    options = list(
      temperature = temperature,
      num_predict = max_tokens
    )
  )

  # Make POST request
  tryCatch({
    response <- POST(
      url,
      body = toJSON(data, auto_unbox = TRUE),
      encode = "json",
      content_type_json(),
      timeout(120)
    )

    # Check status
    if (status_code(response) == 200) {
      result <- content(response, as = "parsed", simplifyVector = TRUE)
      return(trimws(result$response))
    } else {
      warning(paste("Ollama returned status code:", status_code(response)))
      return(NULL)
    }

  }, error = function(e) {
    if (grepl("Connection refused", e$message)) {
      warning("Can't connect to Ollama. Is it running? Try: ollama serve")
    } else if (grepl("Timeout", e$message)) {
      warning("Request timed out (LLM took too long)")
    } else {
      warning(paste("Error calling Ollama:", e$message))
    }
    return(NULL)
  })
}


# ============================================
# Advanced: Structured Output
# ============================================

#' Call Ollama and Parse JSON Response
#'
#' Call Ollama and automatically parse the response as JSON.
#' Useful for structured extraction tasks.
#'
#' @param prompt Character string asking LLM to return JSON
#' @param model Character string, which model to use
#' @param temperature Numeric, lower is better for structured output
#' @return List (parsed JSON), or NULL on error
#' @export
#'
#' @examples
#' prompt <- "
#' Extract gene name and plot type. Return as JSON.
#' Query: Show me TP53 as a violin plot
#' JSON:
#' "
#' result <- call_ollama_json(prompt)
#' print(result$gene_name)  # "TP53"
call_ollama_json <- function(prompt,
                             model = DEFAULT_MODEL,
                             temperature = 0.1) {

  # Call LLM
  response_text <- call_ollama(prompt, model = model, temperature = temperature)

  if (is.null(response_text)) {
    return(NULL)
  }

  # Try to parse as JSON
  tryCatch({
    # Remove markdown code blocks if present
    response_text <- trimws(response_text)
    response_text <- gsub("^```json\n?", "", response_text)
    response_text <- gsub("^```\n?", "", response_text)
    response_text <- gsub("\n?```$", "", response_text)

    # Parse JSON
    result <- fromJSON(response_text, simplifyVector = TRUE)
    return(result)

  }, error = function(e) {
    warning(paste("Could not parse LLM response as JSON:", e$message))
    warning(paste("Raw response:", response_text))
    return(NULL)
  })
}


# ============================================
# Status Checking
# ============================================

#' Check if Ollama is Running
#'
#' Test connection to Ollama server.
#'
#' @return Logical TRUE if Ollama is accessible, FALSE otherwise
#' @export
#'
#' @examples
#' if (check_ollama_status()) {
#'   cat("Ollama is ready!\n")
#' } else {
#'   cat("Start Ollama with: ollama serve\n")
#' }
check_ollama_status <- function() {
  tryCatch({
    response <- GET(paste0(OLLAMA_URL, "/api/tags"), timeout(5))
    return(status_code(response) == 200)
  }, error = function(e) {
    return(FALSE)
  })
}


#' Get Available Ollama Models
#'
#' List all models available in local Ollama installation.
#'
#' @return Character vector of model names, or empty vector on error
#' @export
#'
#' @examples
#' models <- get_available_models()
#' print(models)  # c("llama3.2", "codellama", "mistral")
get_available_models <- function() {
  tryCatch({
    response <- GET(paste0(OLLAMA_URL, "/api/tags"), timeout(5))
    if (status_code(response) == 200) {
      data <- content(response, as = "parsed")
      return(sapply(data$models, function(m) m$name))
    }
    return(character(0))
  }, error = function(e) {
    return(character(0))
  })
}


# ============================================
# Convenience Functions for Common Tasks
# ============================================

#' Ask LLM to Classify Text
#'
#' Have the LLM classify text into one of several categories.
#'
#' @param text Character string to classify
#' @param options Character vector of possible categories
#' @param context Character string, additional context
#' @return Character string (one of options), or NULL
#' @export
#'
#' @examples
#' query <- "Show me TP53 expression"
#' category <- ask_llm_to_classify(
#'   query,
#'   options = c("plot", "stats", "question"),
#'   context = "Classify this user query"
#' )
#' print(category)  # "plot"
ask_llm_to_classify <- function(text, options, context = "") {

  options_str <- paste(options, collapse = ", ")

  prompt <- paste0(
    context, "\n\n",
    "Choose ONLY ONE from these options: ", options_str, "\n",
    "Return ONLY the option name, nothing else.\n\n",
    "Text: ", text, "\n\n",
    "Choice:"
  )

  response <- call_ollama(prompt, temperature = 0.0)

  if (is.null(response)) {
    return(NULL)
  }

  # Clean and validate response
  response <- tolower(trimws(response))

  for (option in options) {
    if (grepl(tolower(option), response)) {
      return(option)
    }
  }

  return(NULL)
}


#' Ask LLM to Extract Information
#'
#' Have the LLM extract specific information from text.
#'
#' @param text Character string to extract from
#' @param what_to_extract Character string, description of what to extract
#' @param example Character string, optional example output
#' @return Character string with extracted info, or NULL
#' @export
#'
#' @examples
#' query <- "Show me TP53 and BRCA1 expression"
#' genes <- ask_llm_to_extract(
#'   query,
#'   what_to_extract = "gene names",
#'   example = "TP53, BRCA1"
#' )
#' print(genes)  # "TP53, BRCA1"
ask_llm_to_extract <- function(text, what_to_extract, example = "") {

  prompt <- paste0(
    "Extract ", what_to_extract, " from this text.\n",
    "Return ONLY the extracted information, nothing else.\n"
  )

  if (example != "") {
    prompt <- paste0(prompt, "Example output: ", example, "\n")
  }

  prompt <- paste0(
    prompt,
    "\nText: ", text, "\n\n",
    "Extracted ", what_to_extract, ":"
  )

  response <- call_ollama(prompt, temperature = 0.1)

  return(if (is.null(response)) NULL else trimws(response))
}


# ============================================
# Learning Objectives Helper
# ============================================

#' Demonstrate LLM Skills
#'
#' Show examples of the 4 main LLM skills for the hackathon:
#' 1. Intent classification
#' 2. Parameter extraction
#' 3. Natural language generation
#' 4. Structured output (JSON)
#'
#' @export
demonstrate_llm_skills <- function() {

  cat("=== LLM Skills Demonstration ===\n\n")

  # Skill 1: Intent Classification
  cat("1. INTENT CLASSIFICATION\n")
  user_query <- "Is TP53 significantly different between groups?"
  intent <- ask_llm_to_classify(
    user_query,
    options = c("plot", "stats", "question"),
    context = "What is the user's intent?"
  )
  cat("   Query:", user_query, "\n")
  cat("   Intent:", intent, "\n\n")

  # Skill 2: Parameter Extraction
  cat("2. PARAMETER EXTRACTION\n")
  user_query <- "Show TP53 in tumor samples only"
  genes <- ask_llm_to_extract(user_query, "gene names")
  filters <- ask_llm_to_extract(user_query, "filter conditions")
  cat("   Query:", user_query, "\n")
  cat("   Genes:", genes, "\n")
  cat("   Filters:", filters, "\n\n")

  # Skill 3: Natural Language Generation
  cat("3. NATURAL LANGUAGE GENERATION\n")
  stats_data <- "t-test: statistic=-2.45, p-value=0.023"
  prompt <- paste("Explain these statistical results in one sentence:", stats_data)
  explanation <- call_ollama(prompt, temperature = 0.3)
  cat("   Data:", stats_data, "\n")
  cat("   Explanation:", explanation, "\n\n")

  # Skill 4: Structured Output (JSON)
  cat("4. STRUCTURED OUTPUT (JSON)\n")
  user_query <- "Compare TP53 and BRCA1 with a heatmap"
  prompt <- paste0(
    "Extract information from this query and return as JSON with these keys:\n",
    "- genes: list of gene names\n",
    "- plot_type: type of plot requested\n\n",
    "Query: ", user_query, "\n\n",
    "JSON:"
  )
  result <- call_ollama_json(prompt)
  cat("   Query:", user_query, "\n")
  cat("   JSON:\n")
  print(result)
  cat("\n")
}


# ============================================
# Main (for testing)
# ============================================

if (interactive()) {
  # Test when sourced interactively
  cat("Testing Ollama connection...\n")

  if (check_ollama_status()) {
    cat("✓ Ollama is running!\n\n")

    # Show available models
    models <- get_available_models()
    cat("Available models:", paste(models, collapse = ", "), "\n\n")

    # Run demonstration
    demonstrate_llm_skills()

  } else {
    cat("✗ Ollama is not running. Start it with: ollama serve\n")
  }
}
