##### ChatSeq - Basic R Shiny Chatbot #####
# A simple chatbot for visualizing gene expression
# Usage: shiny::runApp("app/R/chatbot_basic.R")

library(shiny)
library(ggplot2)

# Source utility functions
source("utils/R/llm_utils.R")
source("utils/R/plot_utils.R")

# Load data at startup
cat("Loading expression data...\n")
expr_data <- load_expression_data("data")
cat("Data loaded! Found", length(get_available_genes(expr_data)), "genes\n")


# ============================================
# UI
# ============================================

ui <- fluidPage(
  # Custom CSS
  tags$head(
    tags$style(HTML("
      .main-title {
        color: #2C3E50;
        font-weight: bold;
        margin-bottom: 10px;
      }
      .subtitle {
        color: #7F8C8D;
        margin-bottom: 30px;
      }
      .chat-box {
        background-color: #F8F9FA;
        border: 1px solid #DEE2E6;
        border-radius: 5px;
        padding: 15px;
        margin-bottom: 20px;
        min-height: 100px;
      }
      .user-message {
        background-color: #E3F2FD;
        padding: 10px;
        border-radius: 5px;
        margin-bottom: 10px;
      }
      .bot-message {
        background-color: #F1F8E9;
        padding: 10px;
        border-radius: 5px;
        margin-bottom: 10px;
      }
      .error-message {
        background-color: #FFEBEE;
        padding: 10px;
        border-radius: 5px;
        margin-bottom: 10px;
        color: #C62828;
      }
    "))
  ),

  # Title
  titlePanel(
    div(
      h1("ChatSeq", class = "main-title"),
      p("Interactive Gene Expression Visualization", class = "subtitle")
    )
  ),

  # Main layout
  sidebarLayout(

    # Sidebar
    sidebarPanel(
      width = 3,

      h4("About"),
      p("Ask questions about gene expression in natural language!"),

      hr(),

      h5("Example Queries:"),
      tags$ul(
        tags$li("Show me expression of TP53"),
        tags$li("Plot BRCA1 across conditions"),
        tags$li("Display A1CF gene")
      ),

      hr(),

      h5("Dataset Info:"),
      p(strong("Genes:"), length(get_available_genes(expr_data))),
      p(strong("Samples:"), nrow(expr_data$metadata)),
      p(strong("Conditions:"), "Normal, Tumor, Metastatic"),

      hr(),

      # Status indicator
      uiOutput("ollama_status"),

      hr(),

      # Quick plot
      selectInput(
        "quick_gene",
        "Quick Plot:",
        choices = c("Select a gene..." = "", head(get_available_genes(expr_data), 20)),
        selected = NULL
      ),
      actionButton("quick_plot_btn", "Plot", class = "btn-primary btn-sm", width = "100%")
    ),

    # Main panel
    mainPanel(
      width = 9,

      # Chat interface
      h3("Chat Interface"),

      # Input area
      fluidRow(
        column(
          10,
          textInput(
            "user_question",
            NULL,
            placeholder = "Ask about a gene... (e.g., 'Show me TP53 expression')",
            width = "100%"
          )
        ),
        column(
          2,
          actionButton("send_btn", "Send", class = "btn-primary", width = "100%")
        )
      ),

      # Chat history
      div(class = "chat-box", uiOutput("chat_history")),

      hr(),

      # Plot output
      h3("Visualization"),
      plotOutput("plot_output", height = "500px"),

      hr(),

      # Additional info
      verbatimTextOutput("plot_info"),

      hr(),

      # Code display section
      h3("LLM Code & Details"),
      p("See how the LLM processed your query", style = "color: gray;"),
      uiOutput("code_details_ui")
    )
  )
)


# ============================================
# Server
# ============================================

server <- function(input, output, session) {

  # Reactive values for chat history and current plot
  rv <- reactiveValues(
    messages = list(),
    current_plot = NULL,
    current_gene = NULL,
    last_llm_details = NULL
  )

  # Check Ollama status
  output$ollama_status <- renderUI({
    if (check_ollama_status()) {
      div(
        icon("check-circle", style = "color: green;"),
        " Ollama is running"
      )
    } else {
      div(
        icon("times-circle", style = "color: red;"),
        " Ollama is NOT running",
        br(),
        tags$small("Run: ollama serve")
      )
    }
  })

  # Handle send button
  observeEvent(input$send_btn, {
    req(input$user_question)

    user_q <- trimws(input$user_question)

    if (user_q == "") {
      return()
    }

    # Add user message
    rv$messages <- append(rv$messages, list(list(
      type = "user",
      text = user_q
    )))

    # Clear input
    updateTextInput(session, "user_question", value = "")

    # Extract gene name using LLM (with details)
    llm_details <- extract_gene_name_with_details(user_q)
    gene_name <- llm_details$gene_name

    if (is.null(gene_name)) {
      rv$messages <- append(rv$messages, list(list(
        type = "error",
        text = "Sorry, I couldn't extract a gene name from your question. Please try again with a specific gene name (e.g., 'Show me TP53')."
      )))
      rv$last_llm_details <- llm_details  # Store for debugging
      return()
    }

    # Check if gene exists
    if (!gene_name %in% get_available_genes(expr_data)) {
      rv$messages <- append(rv$messages, list(list(
        type = "error",
        text = paste0("Gene '", gene_name, "' not found in dataset. Please check the spelling or try a different gene.")
      )))
      rv$last_llm_details <- llm_details
      return()
    }

    # Create plot
    plot_obj <- plot_gene_boxplot(gene_name, expr_data)

    if (!is.null(plot_obj)) {
      rv$current_plot <- plot_obj
      rv$current_gene <- gene_name

      rv$messages <- append(rv$messages, list(list(
        type = "bot",
        text = paste0("Here's the expression plot for ", gene_name, " across all conditions.")
      )))

      # Store LLM details with executed code
      llm_details$code_executed <- paste0("plot_gene_boxplot('", gene_name, "', expr_data)")
      rv$last_llm_details <- llm_details
    } else {
      rv$messages <- append(rv$messages, list(list(
        type = "error",
        text = "Failed to create plot. Please try again."
      )))
    }
  })

  # Handle quick plot button
  observeEvent(input$quick_plot_btn, {
    req(input$quick_gene)

    if (input$quick_gene == "") {
      return()
    }

    gene_name <- input$quick_gene

    # Create plot
    plot_obj <- plot_gene_boxplot(gene_name, expr_data)

    if (!is.null(plot_obj)) {
      rv$current_plot <- plot_obj
      rv$current_gene <- gene_name

      rv$messages <- append(rv$messages, list(list(
        type = "bot",
        text = paste0("Plotted ", gene_name, " from quick select.")
      )))
    }
  })

  # Render chat history
  output$chat_history <- renderUI({
    if (length(rv$messages) == 0) {
      return(p("No messages yet. Ask a question to get started!", style = "color: gray;"))
    }

    message_divs <- lapply(rv$messages, function(msg) {
      if (msg$type == "user") {
        div(class = "user-message", strong("You: "), msg$text)
      } else if (msg$type == "bot") {
        div(class = "bot-message", strong("ChatSeq: "), msg$text)
      } else {
        div(class = "error-message", strong("Error: "), msg$text)
      }
    })

    do.call(tagList, message_divs)
  })

  # Render plot
  output$plot_output <- renderPlot({
    if (is.null(rv$current_plot)) {
      plot.new()
      text(0.5, 0.5, "No plot yet. Ask about a gene!", cex = 1.5, col = "gray")
    } else {
      print(rv$current_plot)
    }
  })

  # Render plot info
  output$plot_info <- renderText({
    if (is.null(rv$current_gene)) {
      "Waiting for your first query..."
    } else {
      paste0("Currently showing: ", rv$current_gene)
    }
  })

  # Render code details
  output$code_details_ui <- renderUI({
    if (is.null(rv$last_llm_details)) {
      return(p("No LLM query yet. Ask a question to see the details!", style = "color: gray; font-style: italic;"))
    }

    details <- rv$last_llm_details

    tagList(
      # 1. Prompt sent
      h4("1️⃣ Prompt Sent to Ollama"),
      pre(style = "background-color: #f4f4f4; padding: 10px; border-radius: 5px; overflow-x: auto;",
          details$prompt
      ),

      br(),

      # 2. LLM Response
      h4("2️⃣ LLM Response"),
      pre(style = "background-color: #f4f4f4; padding: 10px; border-radius: 5px; overflow-x: auto;",
          details$llm_response
      ),

      br(),

      # 3. Extracted gene name
      h4("3️⃣ Extracted Gene Name"),
      if (!is.null(details$gene_name)) {
        div(style = "background-color: #d4edda; color: #155724; padding: 10px; border-radius: 5px;",
            icon("check-circle"),
            paste0(" Successfully extracted: ", strong(details$gene_name))
        )
      } else {
        div(style = "background-color: #f8d7da; color: #721c24; padding: 10px; border-radius: 5px;",
            icon("times-circle"),
            " Failed to extract gene name"
        )
      },

      br(),

      # 4. Code executed
      if (!is.null(details$code_executed)) {
        tagList(
          h4("4️⃣ R Code Executed"),
          pre(style = "background-color: #e8f4f8; padding: 10px; border-radius: 5px; overflow-x: auto;",
              code(details$code_executed)
          )
        )
      }
    )
  })
}


# ============================================
# Run App
# ============================================

shinyApp(ui = ui, server = server)
