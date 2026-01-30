"""
Plotting Utility Functions for Python
Functions to create expression visualizations
"""

import os
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import pandas as pd

import numpy as np
import plotly.express as px
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


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


def plot_multiple_genes(genes: List[str], expr_data: Dict) -> Optional[plt.Figure]:
    """
    Create facetted boxplots (one facet per gene) for multiple genes using matplotlib.

    Parameters
    ----------
    genes : list[str]
    expr_data : dict
        Dictionary from load_expression_data()

    Returns
    -------
    matplotlib.figure.Figure or None
    """
    # Filter long data for requested genes
    df = expr_data['expr_long']
    sel = df[df['gene'].isin(genes)].copy()

    if sel.empty:
        print(f"Warning: None of genes {genes} found in dataset")
        return None

    # Number of facets
    n = len(genes)
    ncols = min(3, n)
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False)
    axes = axes.flatten()

    colors = {
        'Normal': '#66C2A5',
        'Primary Tumor': '#FC8D62',
        'Metastatic': '#8DA0CB'
    }

    for i, gene in enumerate(genes):
        ax = axes[i]
        gene_df = sel[sel['gene'] == gene]

        # Prepare data per condition
        conditions = ['Normal', 'Primary Tumor', 'Metastatic']
        data_by_condition = [
            gene_df[gene_df['condition'] == cond]['expression'].values
            for cond in conditions
        ]

        bp = ax.boxplot(data_by_condition, labels=conditions, patch_artist=True, showfliers=False)
        for patch, cond in zip(bp['boxes'], conditions):
            patch.set_facecolor(colors[cond])
            patch.set_alpha(0.7)

        for j, cond in enumerate(conditions, 1):
            cond_data = gene_df[gene_df['condition'] == cond]['expression']
            x = [j] * len(cond_data)
            ax.scatter(x, cond_data, alpha=0.5, s=20, color=colors[cond], zorder=3)

        ax.set_title(gene, fontsize=12)
        ax.set_xlabel('')
        ax.set_ylabel('Expression')

    # Remove extra axes
    for k in range(i + 1, len(axes)):
        fig.delaxes(axes[k])

    plt.tight_layout()
    return fig


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

#PCA function for ChatSeq - log/scaling normalization for TPM data

def create_pca_plot(expr_data):
    """
    Create PCA visualization using all genes from already-loaded data
    
    For TPM-normalized data:
    Standardize (for PCA)
    
    Parameters:
    -----------
    expr_data : dict
        Expression data dictionary from load_expression_data()
        Should contain 'normalized' (TPM values) and 'metadata' keys
        
    Returns:
    --------
    plotly.graph_objects.Figure
        Interactive PCA plot
    """
    # Get data from the dict that's already loaded
    expr_df = expr_data['expr_matrix']  # genes x samples (TPM values)
    metadata_df = expr_data['metadata']
    
    # Transpose to samples x genes
    expr_t = expr_df.T
    
    # Standardize the data (mean=0, std=1 for each gene)
    scaler = StandardScaler()
    expr_scaled = scaler.fit_transform(expr_t)
    
    # Run PCA
    pca = PCA(n_components=2)
    pca_coords = pca.fit_transform(expr_scaled)
    
    # Create results dataframe
    pca_df = pd.DataFrame({
        'PC1': pca_coords[:, 0],
        'PC2': pca_coords[:, 1],
        'Sample': expr_t.index
    })
    
    # Add metadata (match by Sample/Run)
    metadata_indexed = metadata_df.set_index('Run')
    pca_df['group'] = metadata_indexed.loc[pca_df['Sample'], 'group'].values
    
    # Get variance explained
    var1 = pca.explained_variance_ratio_[0] * 100
    var2 = pca.explained_variance_ratio_[1] * 100
    
    # Color mapping
    color_map = {
        'g1.normal': '#2E86AB',   # Blue
        'g2.tumor': '#E63946',    # Red
        'g3.mets': '#F77F00'      # Orange
    }
    
    # Create plot
    fig = px.scatter(
        pca_df,
        x='PC1',
        y='PC2',
        color='group',
        hover_data=['Sample'],
        title=f'PCA of Gene Expression Data (All {expr_df.shape[0]} genes, log2-transformed)',
        labels={
            'PC1': f'PC1 ({var1:.1f}% variance)',
            'PC2': f'PC2 ({var2:.1f}% variance)'
        },
        color_discrete_map=color_map
    )
    
    fig.update_traces(marker=dict(size=10, line=dict(width=1, color='white')))
    
    fig.update_layout(
        width=800,
        height=600,
        template='plotly_white',
        font=dict(size=12),
        legend=dict(
            title='Sample Group',
            yanchor="top",
            y=0.99,
            xanchor="right",
            x=0.99
        )
    )
    
    return fig
