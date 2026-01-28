"""
ChatSeq - Basic Python Streamlit Chatbot
A simple chatbot for visualizing gene expression

Usage:
    streamlit run app/python/chatbot_basic.py
"""

import streamlit as st
import sys
import os

# Add utils to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from utils.python.llm_utils import extract_gene_name, extract_gene_name_with_details, check_ollama_status
from utils.python.plot_utils import load_expression_data, plot_gene_boxplot, get_available_genes


# ============================================
# Page Configuration
# ============================================

st.set_page_config(
    page_title="ChatSeq",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================
# Custom CSS
# ============================================

st.markdown("""
<style>
    .main {
        background-color: #f8f9fa;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        display: flex;
        flex-direction: column;
    }
    .chat-message.user {
        background-color: #e3f2fd;
        margin-left: 20%;
    }
    .chat-message.bot {
        background-color: #f1f8e9;
        margin-right: 20%;
    }
    .chat-message.error {
        background-color: #ffebee;
        color: #c62828;
    }
    .chat-message .message-header {
        font-weight: bold;
        margin-bottom: 0.5rem;
        font-size: 0.9rem;
        color: #666;
    }
</style>
""", unsafe_allow_html=True)


# ============================================
# Load Data (Cached)
# ============================================

@st.cache_data
def load_data():
    """Load expression data (cached)"""
    return load_expression_data("data")

# Load data
with st.spinner("Loading expression data..."):
    expr_data = load_data()


# ============================================
# Initialize Session State
# ============================================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "current_gene" not in st.session_state:
    st.session_state.current_gene = None

if "last_llm_details" not in st.session_state:
    st.session_state.last_llm_details = None


# ============================================
# Sidebar
# ============================================

with st.sidebar:
    st.title("⚙️ ChatSeq")

    st.markdown("### About")
    st.info("Ask questions about gene expression in natural language!")

    st.markdown("### Example Queries")
    st.markdown("""
    - Show me expression of TP53
    - Plot BRCA1 across conditions
    - Display A1CF gene
    """)

    st.divider()

    # Dataset info
    st.markdown("### Dataset Info")
    st.metric("Genes", len(get_available_genes(expr_data)))
    st.metric("Samples", len(expr_data['metadata']))
    st.caption("**Conditions:** Normal, Primary Tumor, Metastatic")

    st.divider()

    # Ollama status
    is_running = check_ollama_status()
    if is_running:
        st.success("✓ Ollama is running")
    else:
        st.error("✗ Ollama is NOT running")
        st.caption("Run: `ollama serve`")

    st.divider()

    # Quick plot
    st.markdown("### Quick Plot")
    available_genes = get_available_genes(expr_data)
    quick_gene = st.selectbox(
        "Select a gene:",
        options=[""] + available_genes[:20],
        index=0
    )

    if st.button("Plot", use_container_width=True, type="primary"):
        if quick_gene and quick_gene != "":
            st.session_state.current_gene = quick_gene
            st.session_state.messages.append({
                "type": "bot",
                "text": f"Plotted {quick_gene} from quick select."
            })
            st.rerun()

    st.divider()

    # Clear button
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.current_gene = None
        st.rerun()


# ============================================
# Main Content
# ============================================

st.title("🧬 ChatSeq")
st.markdown("*Interactive Gene Expression Visualization*")

st.divider()

# Chat interface
st.header("Chat Interface")

# Chat history
chat_container = st.container()

with chat_container:
    if len(st.session_state.messages) == 0:
        st.info("No messages yet. Ask a question to get started!")
    else:
        for message in st.session_state.messages:
            msg_type = message["type"]
            msg_text = message["text"]

            if msg_type == "user":
                st.markdown(f"""
                <div class="chat-message user">
                    <div class="message-header">You</div>
                    <div>{msg_text}</div>
                </div>
                """, unsafe_allow_html=True)

            elif msg_type == "bot":
                st.markdown(f"""
                <div class="chat-message bot">
                    <div class="message-header">ChatSeq</div>
                    <div>{msg_text}</div>
                </div>
                """, unsafe_allow_html=True)

            elif msg_type == "error":
                st.markdown(f"""
                <div class="chat-message error">
                    <div class="message-header">Error</div>
                    <div>{msg_text}</div>
                </div>
                """, unsafe_allow_html=True)

# Chat input
user_input = st.chat_input("Ask about a gene... (e.g., 'Show me TP53 expression')")

if user_input:
    # Add user message
    st.session_state.messages.append({
        "type": "user",
        "text": user_input
    })

    # Extract gene name using LLM (with details)
    with st.spinner("Processing your question..."):
        llm_details = extract_gene_name_with_details(user_input)

    gene_name = llm_details['gene_name']

    if gene_name is None:
        st.session_state.messages.append({
            "type": "error",
            "text": "Sorry, I couldn't extract a gene name from your question. Please try again with a specific gene name (e.g., 'Show me TP53')."
        })
        st.session_state.last_llm_details = llm_details  # Store for debugging
    else:
        # Check if gene exists
        available_genes = get_available_genes(expr_data)

        if gene_name not in available_genes:
            st.session_state.messages.append({
                "type": "error",
                "text": f"Gene '{gene_name}' not found in dataset. Please check the spelling or try a different gene."
            })
            st.session_state.last_llm_details = llm_details
        else:
            # Success!
            st.session_state.current_gene = gene_name
            st.session_state.messages.append({
                "type": "bot",
                "text": f"Here's the expression plot for {gene_name} across all conditions."
            })
            # Store LLM details with the executed code
            llm_details['code_executed'] = f"plot_gene_boxplot('{gene_name}', expr_data)"
            st.session_state.last_llm_details = llm_details

    st.rerun()

st.divider()

# Visualization section
st.header("Visualization")

if st.session_state.current_gene is None:
    st.info("No plot yet. Ask about a gene to see its expression!")
else:
    gene_name = st.session_state.current_gene

    # Create and display plot
    fig = plot_gene_boxplot(gene_name, expr_data)

    if fig is not None:
        st.pyplot(fig)
        st.caption(f"Currently showing: **{gene_name}**")
    else:
        st.error(f"Failed to create plot for {gene_name}")

st.divider()

# Code display section
if st.session_state.last_llm_details is not None:
    with st.expander("🔍 Show LLM Code & Details", expanded=False):
        st.markdown("### How the LLM Processed Your Query")

        llm_details = st.session_state.last_llm_details

        # 1. Prompt sent to LLM
        st.markdown("#### 1️⃣ Prompt Sent to Ollama")
        st.code(llm_details['prompt'], language="text")

        # 2. LLM Response
        st.markdown("#### 2️⃣ LLM Response")
        st.code(llm_details['llm_response'], language="text")

        # 3. Extracted gene name
        st.markdown("#### 3️⃣ Extracted Gene Name")
        if llm_details['gene_name']:
            st.success(f"✓ Successfully extracted: **{llm_details['gene_name']}**")
        else:
            st.error("✗ Failed to extract gene name")

        # 4. Code executed (if successful)
        if 'code_executed' in llm_details:
            st.markdown("#### 4️⃣ Python Code Executed")
            st.code(llm_details['code_executed'], language="python")

st.divider()

# Footer
st.markdown("""
<div style='text-align: center; color: #666; font-size: 0.8rem;'>
    Built with Streamlit | Powered by Ollama<br>
    ChatSeq - A hackathon project
</div>
""", unsafe_allow_html=True)
