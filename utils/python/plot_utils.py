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

    # Get conditions present in the data (preserves order)
    all_conditions = ['Normal', 'Primary Tumor', 'Metastatic']
    conditions = [c for c in all_conditions if c in gene_data['condition'].values]

    # If no conditions found, fall back to all conditions
    if not conditions:
        conditions = all_conditions

    # Create boxplot
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
        f"{len(gene_data)} samples across {len(conditions)} condition(s)",
        ha='center',
        fontsize=10,
        color='gray'
    )

    plt.tight_layout()

    return fig


def plot_gene_violin(gene_name: str, expr_data: Dict) -> Optional[plt.Figure]:
    """
    Create a violin plot of gene expression across conditions.

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

    # Get conditions present in the data (preserves order)
    all_conditions = ['Normal', 'Primary Tumor', 'Metastatic']
    conditions = [c for c in all_conditions if c in gene_data['condition'].values]

    # If no conditions found, fall back to all conditions
    if not conditions:
        conditions = all_conditions

    # Prepare data for violin plot
    positions = list(range(1, len(conditions) + 1))

    # Create violin plots
    parts = ax.violinplot(
        [gene_data[gene_data['condition'] == cond]['expression'].values for cond in conditions],
        positions=positions,
        showmeans=True,
        showmedians=True,
        widths=0.7
    )

    # Color the violins
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(colors[conditions[i]])
        pc.set_alpha(0.7)

    # Add individual points
    for i, condition in enumerate(conditions, 1):
        cond_data = gene_data[gene_data['condition'] == condition]['expression']
        x = [i] * len(cond_data)
        ax.scatter(
            x, cond_data,
            alpha=0.4,
            s=30,
            color=colors[condition],
            zorder=3
        )

    # Styling
    ax.set_xticks(positions)
    ax.set_xticklabels(conditions)
    ax.set_title(
        f"Expression of {gene_name} (Violin Plot)",
        fontsize=16,
        fontweight='bold'
    )
    ax.set_xlabel("Condition", fontsize=12)
    ax.set_ylabel("Normalized Expression (log2)", fontsize=12)
    ax.grid(axis='y', alpha=0.3)

    # Add subtitle
    fig.text(
        0.5, 0.92,
        f"{len(gene_data)} samples across {len(conditions)} condition(s)",
        ha='center',
        fontsize=10,
        color='gray'
    )

    plt.tight_layout()

    return fig


def plot_genes_heatmap(genes: List[str], expr_data: Dict) -> Optional[plt.Figure]:
    """
    Create a heatmap of expression for multiple genes.

    Parameters
    ----------
    genes : list[str]
        List of gene names
    expr_data : dict
        Dictionary from load_expression_data()

    Returns
    -------
    matplotlib.figure.Figure or None
        Figure object, or None if genes not found
    """
    import numpy as np

    # Get expression matrix and metadata
    expr_matrix = expr_data['expr_matrix']
    metadata = expr_data['metadata']

    # Filter for requested genes
    available_genes = [g for g in genes if g in expr_matrix.index]

    if not available_genes:
        print(f"Warning: None of genes {genes} found in dataset")
        return None

    # Subset expression data
    subset_expr = expr_matrix.loc[available_genes, :]

    # Sort samples by condition
    metadata_sorted = metadata.sort_values('condition')
    subset_expr = subset_expr[metadata_sorted['Run'].values]

    # Create figure
    n_genes = len(available_genes)
    n_samples = len(metadata_sorted)

    fig, ax = plt.subplots(figsize=(max(12, n_samples * 0.3), max(6, n_genes * 0.5)))

    # Create heatmap
    im = ax.imshow(subset_expr.values, aspect='auto', cmap='RdBu_r', interpolation='nearest')

    # Set ticks and labels
    ax.set_yticks(range(n_genes))
    ax.set_yticklabels(available_genes, fontsize=10)

    # Don't show all sample names (too many), just condition blocks
    ax.set_xticks([])
    ax.set_xlabel("Samples (ordered by condition)", fontsize=12)

    # Add condition annotation at top
    condition_colors = {
        'Normal': '#66C2A5',
        'Primary Tumor': '#FC8D62',
        'Metastatic': '#8DA0CB'
    }

    # Create condition color bar at top
    condition_array = metadata_sorted['condition'].values
    unique_conditions = ['Normal', 'Primary Tumor', 'Metastatic']

    # Add thick colored bar for conditions above heatmap
    for i, sample_cond in enumerate(condition_array):
        ax.axvline(i, color=condition_colors[sample_cond], alpha=0.9, linewidth=6, ymin=1.02, ymax=1.15, clip_on=False)

    # Add vertical separator lines between condition groups
    # Find where conditions change
    prev_cond = None
    for i, sample_cond in enumerate(condition_array):
        if prev_cond is not None and sample_cond != prev_cond:
            # Draw separator line at condition boundary
            ax.axvline(i - 0.5, color='black', linewidth=2.5, linestyle='-', alpha=0.8)
        prev_cond = sample_cond

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Expression (log2)', rotation=270, labelpad=20, fontsize=10)

    # Title
    ax.set_title(
        f"Expression Heatmap ({len(available_genes)} genes)",
        fontsize=16,
        fontweight='bold',
        pad=20
    )

    # Add legend for conditions
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=condition_colors[c], label=c, alpha=0.7) for c in unique_conditions]
    ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.1, 1), frameon=True)

    plt.tight_layout()

    return fig


def plot_gene_barplot(gene_name: str, expr_data: Dict) -> Optional[plt.Figure]:
    """
    Create a barplot (mean ± SEM) of gene expression across conditions.

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
    import numpy as np

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

    # Get conditions present in the data (preserves order)
    all_conditions = ['Normal', 'Primary Tumor', 'Metastatic']
    conditions = [c for c in all_conditions if c in gene_data['condition'].values]

    # If no conditions found, fall back to all conditions
    if not conditions:
        conditions = all_conditions

    # Calculate mean and SEM for each condition
    means = []
    sems = []
    for cond in conditions:
        cond_data = gene_data[gene_data['condition'] == cond]['expression']
        means.append(cond_data.mean())
        sems.append(cond_data.sem())  # Standard error of mean

    # Create bar positions
    x_pos = np.arange(len(conditions))

    # Create bars
    bars = ax.bar(x_pos, means, yerr=sems, capsize=5, alpha=0.8,
                   color=[colors[c] for c in conditions], edgecolor='black', linewidth=1.5)

    # Styling
    ax.set_xticks(x_pos)
    ax.set_xticklabels(conditions)
    ax.set_title(
        f"Expression of {gene_name} (Bar Plot)",
        fontsize=16,
        fontweight='bold'
    )
    ax.set_xlabel("Condition", fontsize=12)
    ax.set_ylabel("Mean Expression ± SEM (log2)", fontsize=12)
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    # Add sample counts as text on bars
    for i, (bar, cond) in enumerate(zip(bars, conditions)):
        n_samples = len(gene_data[gene_data['condition'] == cond])
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + sems[i] + 0.05,
                f'n={n_samples}', ha='center', va='bottom', fontsize=9, color='gray')

    plt.tight_layout()
    return fig


def plot_multiple_barplots(genes: List[str], expr_data: Dict) -> Optional[plt.Figure]:
    """
    Create facetted barplots (one facet per gene) for multiple genes.

    Parameters
    ----------
    genes : list[str]
        List of gene names
    expr_data : dict
        Dictionary from load_expression_data()

    Returns
    -------
    matplotlib.figure.Figure or None
    """
    import numpy as np

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

    conditions = ['Normal', 'Primary Tumor', 'Metastatic']

    for i, gene in enumerate(genes):
        ax = axes[i]
        gene_df = sel[sel['gene'] == gene]

        # Calculate mean and SEM for each condition
        means = []
        sems = []
        for cond in conditions:
            cond_data = gene_df[gene_df['condition'] == cond]['expression']
            means.append(cond_data.mean())
            sems.append(cond_data.sem())

        # Create bars
        x_pos = np.arange(len(conditions))
        bars = ax.bar(x_pos, means, yerr=sems, capsize=4, alpha=0.8,
                      color=[colors[c] for c in conditions], edgecolor='black', linewidth=1)

        ax.set_xticks(x_pos)
        ax.set_xticklabels(conditions, fontsize=9, rotation=15, ha='right')
        ax.set_title(gene, fontsize=12, fontweight='bold')
        ax.set_xlabel('')
        ax.set_ylabel('Mean Expression ± SEM', fontsize=10)
        ax.grid(axis='y', alpha=0.3, linestyle='--')

    # Remove extra axes
    for k in range(i + 1, len(axes)):
        fig.delaxes(axes[k])

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


def plot_multiple_violins(genes: List[str], expr_data: Dict) -> Optional[plt.Figure]:
    """
    Create facetted violin plots (one facet per gene) for multiple genes.

    Parameters
    ----------
    genes : list[str]
        List of gene names
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
        positions = [1, 2, 3]
        data_by_condition = [
            gene_df[gene_df['condition'] == cond]['expression'].values
            for cond in conditions
        ]

        # Create violin plots
        parts = ax.violinplot(data_by_condition, positions=positions, showmeans=True, showmedians=True, widths=0.7)

        # Color the violins
        for j, pc in enumerate(parts['bodies']):
            pc.set_facecolor(colors[conditions[j]])
            pc.set_alpha(0.7)

        # Add individual points
        for j, cond in enumerate(conditions, 1):
            cond_data = gene_df[gene_df['condition'] == cond]['expression']
            x = [j] * len(cond_data)
            ax.scatter(x, cond_data, alpha=0.4, s=20, color=colors[cond], zorder=3)

        ax.set_xticks(positions)
        ax.set_xticklabels(conditions, fontsize=9)
        ax.set_title(gene, fontsize=12, fontweight='bold')
        ax.set_xlabel('')
        ax.set_ylabel('Expression', fontsize=10)
        ax.grid(axis='y', alpha=0.3)

    # Remove extra axes
    for k in range(i + 1, len(axes)):
        fig.delaxes(axes[k])

    plt.tight_layout()
    return fig


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
