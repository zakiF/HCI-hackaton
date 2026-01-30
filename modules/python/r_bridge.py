"""
R plotting bridge that prefers rpy2 but falls back to using Rscript CLI.

Provides:
- plot_multiple_genes(genes, expr_matrix_df, metadata_df, width=10, height=6, dpi=150)

If rpy2 is installed the function calls into R directly. Otherwise it writes
temporary CSVs and invokes `Rscript modules/R/cli_plot_wrapper.R` to create the
PNG. This avoids requiring rpy2 on the Python side.
"""
import os
import tempfile
import shutil
import subprocess
from pathlib import Path

_HAS_RPY2 = False
try:
    from rpy2 import robjects
    from rpy2.robjects import pandas2ri
    pandas2ri.activate()
    _HAS_RPY2 = True
except Exception:
    _HAS_RPY2 = False

_MODULE_DIR = Path(__file__).resolve().parent
_R_SCRIPT = (_MODULE_DIR / '..' / 'R' / 'multi_gene_viz.R').resolve()
_CLI_WRAPPER = (_MODULE_DIR / '..' / 'R' / 'cli_plot_wrapper.R').resolve()


def _source_r():
    """Source the R script using rpy2 if available."""
    if not _R_SCRIPT.exists():
        raise FileNotFoundError(f"R script not found at {_R_SCRIPT}")
    robjects.r['source'](str(_R_SCRIPT))


def _call_rscript(genes, expr_csv, meta_csv, out_png, width, height, dpi, group_col):
    rscript_exec = shutil.which('Rscript')
    if not rscript_exec:
        raise RuntimeError("Rscript executable not found on PATH. Install R and ensure Rscript is available.")

    if not _CLI_WRAPPER.exists():
        raise FileNotFoundError(f"CLI wrapper not found at {_CLI_WRAPPER}")

    genes_arg = ",".join(genes)

    cmd = [
        rscript_exec,
        str(_CLI_WRAPPER),
        genes_arg,
        expr_csv,
        meta_csv,
        out_png,
        str(width),
        str(height),
        str(dpi),
        group_col,
    ]

    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"Rscript failed: {proc.returncode}\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}")

    # On success the wrapper prints OK
    return out_png


def plot_multiple_genes(genes, expr_matrix_df, metadata_df, width=10, height=6, dpi=150, group_col='group'):
    """Create a multi-gene plot (PNG) using R.

    Tries rpy2 first; if not available uses Rscript CLI.

    Returns the path to the saved PNG file.
    """
    # Prepare temporary output file
    tmp = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
    tmpname = tmp.name
    tmp.close()

    if _HAS_RPY2:
        # Use rpy2 to call the R wrapper directly
        _source_r()
        r_expr = pandas2ri.py2rpy(expr_matrix_df)
        r_meta = pandas2ri.py2rpy(metadata_df)

        robjects.globalenv['expr_matrix'] = r_expr
        robjects.globalenv['metadata'] = r_meta
        robjects.globalenv['gene_names'] = robjects.StrVector(genes)

        r_plot_fn = robjects.r['plot_genes_save_png']
        r_plot_fn(robjects.globalenv['gene_names'],
                  robjects.globalenv['expr_matrix'],
                  robjects.globalenv['metadata'],
                  tmpname,
                  width, height, dpi,
                  group_col)

        return tmpname
    else:
        # Fallback: write CSVs and call Rscript
        expr_tmp = tempfile.NamedTemporaryFile(suffix='.csv', delete=False)
        meta_tmp = tempfile.NamedTemporaryFile(suffix='.csv', delete=False)
        expr_tmp.close()
        meta_tmp.close()

        expr_matrix_df.to_csv(expr_tmp.name, index=True)
        metadata_df.to_csv(meta_tmp.name, index=False)

        return _call_rscript(genes, expr_tmp.name, meta_tmp.name, tmpname, width, height, dpi, group_col)

