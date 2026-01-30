"""
ChatSeq - HACKATHON VERSION with Feature Flags
LLM-powered chatbot for visualizing gene expression

This version includes experimental features from the hackathon team.
Each feature can be toggled on/off via feature flags below.

Usage:
    streamlit run app/python/chatbot.py
"""

import streamlit as st
import sys
import os

# Add utils to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from utils.python.llm_utils import extract_gene_name, extract_gene_name_with_details, check_ollama_status
from utils.python.plot_utils import load_expression_data, plot_gene_boxplot, get_available_genes, create_pca_plot


# ============================================
# FEATURE FLAGS - Toggle experimental features
# ============================================

# Set to True to enable all features, False for safe mode
SAFE_MODE = False

if SAFE_MODE:
    # Emergency fallback - disable all experimental features
    ENABLE_MULTI_GENE = True
    ENABLE_FILTERS = False
    ENABLE_CONVERSATION = False
    ENABLE_STATS = False
    ENABLE_RAG = False
else:
    # Individual feature toggles
    ENABLE_MULTI_GENE = True     # Zaki + Udhayakumar: Multi-gene plotting
    ENABLE_FILTERS = True        # Qing: Natural language data filtering
    ENABLE_CONVERSATION = True   # Miao: Conversational context/follow-ups
    ENABLE_STATS = True          # Tayler: Statistical testing
    ENABLE_RAG = True            # David: Gene information RAG
    ENABLE_PCA = True            # Javier: PCA Visualization


# ============================================
# IMPORT EXPERIMENTAL FEATURES (with safe fallbacks)
# ============================================

# Multi-gene visualization (Zaki + Udhayakumar)
if ENABLE_MULTI_GENE:
    # We enable multi-gene plotting even if rpy2 isn't installed. The bridge
    # will fall back to calling Rscript if needed, so users do not need rpy2.
    print("✓ Multi-gene feature enabled (uses Rscript fallback if rpy2 missing)")

# Filter extraction (Qing)
if ENABLE_FILTERS:
    try:
        from modules.python.llm_filters import extract_filters, apply_filters
        print("✓ Filter feature loaded")
    except Exception as e:
        print(f"✗ Filter feature disabled: {e}")
        ENABLE_FILTERS = False

# Conversation manager (Miao)
if ENABLE_CONVERSATION:
    try:
        from modules.python.conversation import ConversationManager
        print("✓ Conversation feature loaded")
    except Exception as e:
        print(f"✗ Conversation feature disabled: {e}")
        ENABLE_CONVERSATION = False

# Statistical testing (Tayler)
if ENABLE_STATS:
    try:
        from modules.python.llm_stats import handle_stats_query, is_stats_query
        print("✓ Stats feature loaded")
    except Exception as e:
        print(f"✗ Stats feature disabled: {e}")
        ENABLE_STATS = False

# RAG for gene information (David)
if ENABLE_RAG:
    try:
        from modules.python.llm_rag import answer_gene_question, is_gene_question
        print("✓ RAG feature loaded")
    except Exception as e:
        print(f"✗ RAG feature disabled: {e}")
        ENABLE_RAG = False

# PCA visualization flag (Javier's feature)
if "show_pca" not in st.session_state:
    st.session_state.show_pca = False

# ============================================
# Page Configuration
# ============================================

st.set_page_config(
    page_title="ChatSeq - Hackathon Version",
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

# Conversation manager (Miao's feature)
if ENABLE_CONVERSATION and "conversation_mgr" not in st.session_state:
    try:
        st.session_state.conversation_mgr = ConversationManager()
    except:
        st.session_state.conversation_mgr = None


# ============================================
# Sidebar
# ============================================

with st.sidebar:
    st.title("⚙️ ChatSeq")
    st.caption("Hackathon Version with Experimental Features")

    st.markdown("### About")
    st.info("Ask questions about gene expression using natural language and local LLMs!")

    st.markdown("### Example Queries")

    # Base examples
    st.markdown("**Basic:**")
    st.markdown("- Show me expression of TP53")
    st.markdown("- Plot BRCA1 across conditions")

    # Feature-specific examples
    if ENABLE_MULTI_GENE:
        st.markdown("**Multi-gene (Zaki+Udhaya):**")
        st.markdown("- Compare TP53, BRCA1, and EGFR")

    if ENABLE_FILTERS:
        st.markdown("**Filtering (Qing):**")
        st.markdown("- Show TP53 in only tumor samples")

    if ENABLE_STATS:
        st.markdown("**Statistics (Tayler):**")
        st.markdown("- Is TP53 significant between groups?")

    if ENABLE_RAG:
        st.markdown("**Gene Info (David):**")
        st.markdown("- What does TP53 do?")

    if ENABLE_PCA:
        st.markdown("**PCA (Javier):**")
        st.markdown("- Run PCA on this dataset")

    st.divider()

    # Dataset info
    st.markdown("### Dataset Info")
    st.metric("Genes", len(get_available_genes(expr_data)))
    st.metric("Samples", len(expr_data['metadata']))
    st.caption("**Conditions:** Normal, Primary Tumor, Metastatic")

    st.divider()

    # Feature status
    st.markdown("### Feature Status")
    features = {
        "Multi-gene": ENABLE_MULTI_GENE,
        "Filters": ENABLE_FILTERS,
        "Conversation": ENABLE_CONVERSATION,
        "Statistics": ENABLE_STATS,
        "Gene Info (RAG)": ENABLE_RAG,
        "PCA": ENABLE_PCA
    }

    for feature_name, enabled in features.items():
        if enabled:
            st.success(f"✓ {feature_name}")
        else:
            st.error(f"✗ {feature_name}")

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

    # PCA Quick Action (Javier's feature)
    if ENABLE_PCA:
        st.markdown("### Quick Actions")
        if st.button("Run PCA", use_container_width=True, type="primary"):
            st.session_state.show_pca = True
            st.session_state.current_gene = None
            st.session_state.messages.append({
                "type": "user",
                "text": "Run PCA on this dataset"
            })
            st.session_state.messages.append({
                "type": "bot",
                "text": "I've created a PCA visualization using all genes."
            })
            st.rerun()

    st.divider()

    # Clear button
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.current_gene = None
        if ENABLE_CONVERSATION and st.session_state.conversation_mgr:
            st.session_state.conversation_mgr.reset()
        st.rerun()


# ============================================
# Main Content
# ============================================

st.title("🧬 ChatSeq")
st.markdown("*Interactive Gene Expression Visualization with Local LLMs*")

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


# ============================================
# MAIN QUERY HANDLER
# ============================================

user_input = st.chat_input("Ask about a gene... (e.g., 'Show me TP53 expression')")

if user_input:
    # Add user message
    st.session_state.messages.append({
        "type": "user",
        "text": user_input
    })

    # ========================================
    # ROUTE 1: Gene Information Question (David's RAG)
    # ========================================

    if ENABLE_RAG:
        try:
            if is_gene_question(user_input):
                with st.spinner("Looking up gene information..."):
                    response = answer_gene_question(user_input)

                st.session_state.messages.append({
                    "type": "bot",
                    "text": response
                })
                st.rerun()
                # Stop here - don't proceed to plotting
        except Exception as e:
            st.session_state.messages.append({
                "type": "error",
                "text": f"RAG feature error: {str(e)}"
            })
            # Continue to next route

    # ========================================
    # ROUTE 2: Statistical Question (Tayler's Stats)
    # ========================================

    if ENABLE_STATS:
        try:
            if is_stats_query(user_input):
                with st.spinner("Running statistical analysis..."):
                    result = handle_stats_query(user_input, expr_data)

                # Display results
                st.session_state.messages.append({
                    "type": "bot",
                    "text": result['summary']
                })

                # Store stats results for display
                if 'stats_results' not in st.session_state:
                    st.session_state.stats_results = []
                st.session_state.stats_results.append(result)

                st.rerun()
                # Stop here - don't proceed to plotting
        except Exception as e:
            st.session_state.messages.append({
                "type": "error",
                "text": f"Stats feature error: {str(e)}"
            })
            # Continue to plotting route

    # ========================================
    # ROUTE 2.5: PCA Request (Javier's PCA Feature)
    # ========================================
    
    if ENABLE_PCA and 'pca' in user_input.lower():
        st.session_state.show_pca = True
        st.session_state.current_gene = None
        st.session_state.messages.append({
            "type": "bot",
            "text": "I've created a PCA visualization using all genes."
        })
        st.rerun()

    # ========================================
    # ROUTE 3: Visualization/Plotting Request
    # ========================================

    # --- STEP 1: Resolve context (Miao's Conversation Feature) ---

    resolved_input = user_input  # Default to original input

    if ENABLE_CONVERSATION and st.session_state.conversation_mgr:
        try:
            resolved_input = st.session_state.conversation_mgr.resolve_context(
                user_input,
                st.session_state.get('current_gene', None)
            )
        except Exception as e:
            st.warning(f"Conversation feature skipped: {str(e)}")
            # Continue with original input

    # --- STEP 2: Extract gene name(s) using LLM ---

    with st.spinner("Processing your question..."):
        llm_details = extract_gene_name_with_details(resolved_input)

    gene_names = llm_details.get('gene_names', [])
    gene_name = llm_details['gene_name']  # First gene for backwards compatibility

    if not gene_names or gene_name is None:
        st.session_state.messages.append({
            "type": "error",
            "text": "Sorry, I couldn't extract a gene name from your question. Please try again with a specific gene name (e.g., 'Show me TP53')."
        })
        st.session_state.last_llm_details = llm_details
        st.rerun()

    # --- STEP 3: Check if gene exists ---

    available_genes = get_available_genes(expr_data)

    # Check which genes are valid
    valid_genes = [g for g in gene_names if g in available_genes]
    invalid_genes = [g for g in gene_names if g not in available_genes]

    if not valid_genes:
        st.session_state.messages.append({
            "type": "error",
            "text": f"Gene(s) '{', '.join(gene_names)}' not found in dataset. Please check the spelling or try a different gene."
        })
        st.session_state.last_llm_details = llm_details
        st.rerun()

    # If multiple genes detected, notify user
    if len(gene_names) > 1:
        if invalid_genes:
            st.session_state.messages.append({
                "type": "bot",
                "text": f"Found {len(gene_names)} genes: {', '.join(gene_names)}. Note: {', '.join(invalid_genes)} not in dataset. Plotting {valid_genes[0]}."
            })
        else:
            st.session_state.messages.append({
                "type": "bot",
                "text": f"Found {len(gene_names)} genes: {', '.join(gene_names)}. Currently plotting {valid_genes[0]} (multi-gene visualization coming soon!)."
            })

    # Use first valid gene for now
    gene_name = valid_genes[0]

    # --- STEP 4: Apply filters (Qing's Filter Feature) ---

    filtered_data = expr_data  # Default to full dataset
    filter_info = None

    if ENABLE_FILTERS:
        try:
            filter_info = extract_filters(resolved_input)
            if filter_info and filter_info.get('has_filter'):
                filtered_data = apply_filters(expr_data, filter_info)
                # Update LLM details
                llm_details['filters_applied'] = filter_info
        except Exception as e:
            st.warning(f"Filter feature skipped: {str(e)}")
            # Continue with unfiltered data

    # --- STEP 5: Create visualization ---

    # Multi-gene plotting: if multiple valid genes detected, try R bridge
    if ENABLE_MULTI_GENE and len(valid_genes) > 1:
        try:
            from modules.python.r_bridge import plot_multiple_genes as r_plot_multi
            img_path = r_plot_multi(valid_genes, filtered_data['expr_matrix'], filtered_data['metadata'])

            # Store image path and context for display
            st.session_state.current_plot_image = img_path
            st.session_state.last_multi_genes = valid_genes

            # Keep first gene as conversation context/backwards compatibility
            st.session_state.current_gene = valid_genes[0]

            llm_details['code_executed'] = f"plot_multiple_genes({valid_genes})"
        except Exception as e:
            # Try Python fallback before failing
            try:
                from utils.python.plot_utils import plot_multiple_genes as py_plot_multi
                fig = py_plot_multi(valid_genes, filtered_data)
                if fig is not None:
                    st.session_state.current_plot_image = None
                    st.session_state.current_plot_figure = fig
                    st.session_state.last_multi_genes = valid_genes
                    st.session_state.current_gene = valid_genes[0]
                    llm_details['code_executed'] = f"python_plot_multiple_genes({valid_genes})"
                else:
                    raise RuntimeError("Python fallback returned no figure")
            except Exception as e2:
                st.session_state.messages.append({
                    "type": "error",
                    "text": f"Multi-gene plotting failed: {str(e)}; Python fallback: {str(e2)}. Showing single-gene plot for {gene_name}."
                })
                st.session_state.current_plot_image = None
                st.session_state.current_plot_figure = None
                st.session_state.current_gene = gene_name
    else:
        # Default single-gene plotting
        st.session_state.current_plot_image = None
        st.session_state.current_plot_figure = None
        st.session_state.current_gene = gene_name
        st.session_state.show_pca = False 

    # Build response message
    response_text = f"Here's the expression plot for {gene_name}"
    if filter_info and filter_info.get('has_filter'):
        response_text += f" (filtered: {filter_info.get('description', 'custom filter')})"
    response_text += " across all conditions."

    st.session_state.messages.append({
        "type": "bot",
        "text": response_text
    })

    # Store LLM details
    llm_details['code_executed'] = f"plot_gene_boxplot('{gene_name}', filtered_data)"
    st.session_state.last_llm_details = llm_details

    # Update conversation context (Miao)
    if ENABLE_CONVERSATION and st.session_state.conversation_mgr:
        try:
            st.session_state.conversation_mgr.add_turn(user_input, gene_name)
        except:
            pass

    st.rerun()


# ============================================
# Visualization Section
# ============================================

st.divider()
st.header("Visualization")

if st.session_state.get('show_pca', False):
    # Show PCA plot (Javier's feature)
    with st.spinner("Creating PCA visualization..."):
        try:
            pca_fig = create_pca_plot(expr_data)
            st.plotly_chart(pca_fig, use_container_width=True)
            
            with st.expander("How to interpret this PCA plot"):
                st.markdown("""
                **PCA (Principal Component Analysis):**
                
                - Each point represents one sample
                - Colors: Blue (Normal), Red (Tumor), Orange (Metastatic)
                - PC1 (x-axis): Largest source of variation
                - PC2 (y-axis): Second-largest source of variation
                - Samples close together have similar gene expression
                
                Data is already log2-transformed and standardized before PCA.
                """)
        except Exception as e:
            st.error(f"Error creating PCA plot: {str(e)}")

elif st.session_state.current_gene is None and not st.session_state.get('current_plot_image'):
    st.info("No plot yet. Ask about a gene or run PCA!")
else:
    # If we have an image saved by the R bridge for multi-gene plot, show it
    if st.session_state.get('current_plot_image'):
        st.image(st.session_state['current_plot_image'], use_column_width=True)
        multi_genes = st.session_state.get('last_multi_genes', [])
        if multi_genes:
            st.caption(f"Currently showing: **{', '.join(multi_genes)}**")
        else:
            st.caption("Currently showing multi-gene plot")
    elif st.session_state.get('current_plot_figure'):
        st.pyplot(st.session_state['current_plot_figure'])
        multi_genes = st.session_state.get('last_multi_genes', [])
        if multi_genes:
            st.caption(f"Currently showing: **{', '.join(multi_genes)}**")
        else:
            st.caption("Currently showing multi-gene plot")
    else:
        gene_name = st.session_state.current_gene

        # Create and display plot (Python fallback single-gene)
        fig = plot_gene_boxplot(gene_name, expr_data)

        if fig is not None:
            st.pyplot(fig)
            st.caption(f"Currently showing: **{gene_name}**")
        else:
            st.error(f"Failed to create plot for {gene_name}")


# ============================================
# Statistics Results Display (Tayler's Feature)
# ============================================

if ENABLE_STATS and 'stats_results' in st.session_state and len(st.session_state.stats_results) > 0:
    st.divider()
    st.header("Statistical Results")

    for idx, result in enumerate(st.session_state.stats_results[-3:]):  # Show last 3
        with st.expander(f"Result {idx + 1}: {result.get('results', {}).get('gene', 'Unknown')}", expanded=(idx == 0)):
            res = result.get('results', {})

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Test", res.get('test', 'N/A'))
            with col2:
                st.metric("P-value", f"{res.get('pvalue', 0):.4f}")
            with col3:
                significance = "Yes" if res.get('significant', False) else "No"
                st.metric("Significant (p<0.05)", significance)


# ============================================
# LLM Debug Section
# ============================================

st.divider()

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

        # 3. Extracted gene name(s)
        st.markdown("#### 3️⃣ Extracted Gene Name(s)")
        gene_names = llm_details.get('gene_names', [])
        if gene_names:
            if len(gene_names) == 1:
                st.success(f"✓ Successfully extracted: **{gene_names[0]}**")
            else:
                st.success(f"✓ Successfully extracted {len(gene_names)} genes: **{', '.join(gene_names)}**")
        elif llm_details['gene_name']:
            st.success(f"✓ Successfully extracted: **{llm_details['gene_name']}**")
        else:
            st.error("✗ Failed to extract gene name")

        # 4. Filters (if any)
        if 'filters_applied' in llm_details:
            st.markdown("#### 4️⃣ Filters Applied (Qing's Feature)")
            st.json(llm_details['filters_applied'])

        # 5. Code executed
        if 'code_executed' in llm_details:
            st.markdown("#### 5️⃣ Python Code Executed")
            st.code(llm_details['code_executed'], language="python")


# ============================================
# Footer
# ============================================

st.divider()

st.markdown("""
<div style='text-align: center; color: #666; font-size: 0.8rem;'>
    Built with Streamlit | Powered by Ollama (llama3.2)<br>
    ChatSeq - Hackathon Version | LLM Learning Project
</div>
""", unsafe_allow_html=True)
