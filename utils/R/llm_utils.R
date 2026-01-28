##### LLM Utility Functions for R #####
# Functions to interact with local Ollama LLM

library(httr)
library(jsonlite)

# Configuration
OLLAMA_URL <- "http://localhost:11434"
DEFAULT_MODEL <- "llama3.2"


#' Call Ollama LLM
#'
#' @param prompt Character string with the question/prompt
#' @param model Character string with model name (default: llama3.2)
#' @param temperature Numeric between 0-1 (default: 0.1 for focused responses)
#' @return Character string with LLM response, or NULL if error
#' @export
ask_ollama <- function(prompt, model = DEFAULT_MODEL, temperature = 0.1) {

  url <- paste0(OLLAMA_URL, "/api/generate")

  # Prepare request data
  data <- list(
    model = model,
    prompt = prompt,
    stream = FALSE,
    options = list(
      temperature = temperature
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
    warning(paste("Error calling Ollama:", e$message))
    return(NULL)
  })
}


#' Extract Gene Name from Natural Language Query
#'
#' Uses LLM to parse user question and extract gene name
#'
#' @param user_question Character string with user's question
#' @return Character string with extracted gene name (uppercase), or NULL
#' @export
extract_gene_name <- function(user_question) {

  prompt <- paste0(
    "Extract ONLY the gene name from this question. ",
    "Return ONLY the gene name in uppercase, nothing else. ",
    "If no gene name is found, return 'NONE'.\n\n",
    "Question: ", user_question, "\n",
    "Gene name:"
  )

  response <- ask_ollama(prompt, temperature = 0.1)

  if (is.null(response)) {
    return(NULL)
  }

  # Clean up response
  gene_name <- toupper(trimws(response))

  # Remove any extra text
  gene_name <- gsub("[^A-Z0-9-]", "", gene_name)

  # Check if valid
  if (gene_name == "NONE" || gene_name == "" || nchar(gene_name) == 0) {
    return(NULL)
  }

  return(gene_name)
}


#' Extract Gene Name with Details
#'
#' Uses LLM to parse user question and extract gene name.
#' Returns detailed information about the LLM interaction.
#'
#' @param user_question Character string with user's question
#' @return List with prompt, llm_response, gene_name, and success
#' @export
extract_gene_name_with_details <- function(user_question) {

  prompt <- paste0(
    "Extract ONLY the gene name from this question. ",
    "Return ONLY the gene name in uppercase, nothing else. ",
    "If no gene name is found, return 'NONE'.\n\n",
    "Question: ", user_question, "\n",
    "Gene name:"
  )

  response <- ask_ollama(prompt, temperature = 0.1)

  result <- list(
    prompt = prompt,
    llm_response = if (is.null(response)) "No response from LLM" else response,
    gene_name = NULL,
    success = FALSE
  )

  if (is.null(response)) {
    return(result)
  }

  # Store original response for display
  result$llm_response <- response

  # Clean up response
  gene_name <- toupper(trimws(response))

  # Remove any extra text
  gene_name <- gsub("[^A-Z0-9-]", "", gene_name)

  # Check if valid
  if (gene_name == "NONE" || gene_name == "" || nchar(gene_name) == 0) {
    return(result)
  }

  result$gene_name <- gene_name
  result$success <- TRUE

  return(result)
}


#' Check if Ollama is Running
#'
#' @return Logical TRUE if Ollama is accessible, FALSE otherwise
#' @export
check_ollama_status <- function() {
  tryCatch({
    response <- GET(paste0(OLLAMA_URL, "/api/tags"), timeout(5))
    return(status_code(response) == 200)
  }, error = function(e) {
    return(FALSE)
  })
}
