"""
Plotting Utility Functions for Python
Functions to create expression visualizations
"""

import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Optional
import os


def load_expression_data(data_dir: str = "data") -> Dict:
    """
    Load expression data and metadata.

    Parameters
    ----------
    data_dir : str
        Path to data directory (default: "data")

    Returns
    -------
    dict
        Dictionary with keys:
        - 'expr_matrix': DataFrame with genes as rows, samples as columns
        - 'expr_long': DataFrame in long format for plotting
        - 'metadata': DataFrame with sample annotations
    """
    # Read normalized expression
    expr_matrix = pd.read_csv(
        os.path.join(data_dir, "Normalized_expression.csv"),
        index_col=0
    )

    # Read metadata
    metadata = pd.read_csv(
        os.path.join(data_dir, "Metadata.csv")
    )

    # Clean up group names for better display
    def clean_condition(group):
        if 'normal' in group.lower():
            return 'Normal'
        elif 'tumor' in group.lower():
            return 'Primary Tumor'
        elif 'met' in group.lower():
            return 'Metastatic'
        else:
            return group

    metadata['condition'] = metadata['group'].apply(clean_condition)

    # Set condition order
    condition_order = ['Normal', 'Primary Tumor', 'Metastatic']
    metadata['condition'] = pd.Categorical(
        metadata['condition'],
        categories=condition_order,
        ordered=True
    )

    # Convert to long format for plotting
    expr_long = expr_matrix.T.reset_index()
    expr_long.columns.name = None
    expr_long = expr_long.rename(columns={'index': 'Run'})
    expr_long = expr_long.melt(
        id_vars=['Run'],
        var_name='gene',
        value_name='expression'
    )
    expr_long = expr_long.merge(metadata, on='Run')

    return {
        'expr_matrix': expr_matrix,
        'expr_long': expr_long,
        'metadata': metadata
    }


def plot_gene_boxplot(gene_name: str, expr_data: Dict) -> Optional[plt.Figure]:
    """
    Create a boxplot of gene expression across conditions.

    Parameters
    ----------
    gene_name : str
        Gene name (case-sensitive)
    expr_data : dict
        Dictionary from load_expression_data()

    Returns
    -------
    matplotlib.figure.Figure or None
        Figure object, or None if gene not found
    """
    # Filter for the specific gene
    gene_data = expr_data['expr_long'][expr_data['expr_long']['gene'] == gene_name]

    # Check if gene exists
    if gene_data.empty:
        print(f"Warning: Gene {gene_name} not found in dataset")
        return None

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))

    # Define colors
    colors = {
        'Normal': '#66C2A5',
        'Primary Tumor': '#FC8D62',
        'Metastatic': '#8DA0CB'
    }

    # Create boxplot
    conditions = ['Normal', 'Primary Tumor', 'Metastatic']
    data_by_condition = [
        gene_data[gene_data['condition'] == cond]['expression'].values
        for cond in conditions
    ]

    bp = ax.boxplot(
        data_by_condition,
        labels=conditions,
        patch_artist=True,
        showfliers=False
    )

    # Color the boxes
    for patch, condition in zip(bp['boxes'], conditions):
        patch.set_facecolor(colors[condition])
        patch.set_alpha(0.7)

    # Add individual points
    for i, condition in enumerate(conditions, 1):
        cond_data = gene_data[gene_data['condition'] == condition]['expression']
        x = [i] * len(cond_data)
        ax.scatter(
            x, cond_data,
            alpha=0.5,
            s=50,
            color=colors[condition],
            zorder=3
        )

    # Styling
    ax.set_title(
        f"Expression of {gene_name}",
        fontsize=16,
        fontweight='bold'
    )
    ax.set_xlabel("Condition", fontsize=12)
    ax.set_ylabel("Normalized Expression (log2)", fontsize=12)
    ax.grid(axis='y', alpha=0.3)

    # Add subtitle
    fig.text(
        0.5, 0.92,
        f"{len(gene_data)} samples across 3 conditions",
        ha='center',
        fontsize=10,
        color='gray'
    )

    plt.tight_layout()

    return fig


def get_available_genes(expr_data: Dict) -> List[str]:
    """
    Get list of all available gene names.

    Parameters
    ----------
    expr_data : dict
        Dictionary from load_expression_data()

    Returns
    -------
    list
        Sorted list of gene names
    """
    return sorted(expr_data['expr_long']['gene'].unique())


def search_genes(pattern: str, expr_data: Dict, max_results: int = 10) -> List[str]:
    """
    Search for genes matching a pattern.

    Parameters
    ----------
    pattern : str
        Search pattern (case-insensitive)
    expr_data : dict
        Dictionary from load_expression_data()
    max_results : int
        Maximum number of results to return

    Returns
    -------
    list
        List of matching gene names
    """
    all_genes = get_available_genes(expr_data)
    matches = [g for g in all_genes if pattern.upper() in g.upper()]
    return matches[:max_results]
