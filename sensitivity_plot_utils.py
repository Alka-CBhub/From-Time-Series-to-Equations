## sensitivity_plot_utils.py (Total = 2)
#---------------------------------------

# Import Necessary Library 
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from plot_utils import set_plot_style, set_spines_black
#----------------------------------------------------------------------------
## Utility 1: Convert parameter names into LaTeX-style labels
#----------------------------------------------------------------------------
def format_param(param):
    """
    Converts a plain parameter name (e.g., 'alpha1', 'beta', 'r2') 
    into a LaTeX-formatted string for pretty axis labels in plots.

    Supports Greek letter substitution for common parameter prefixes 
    such as 'alpha', 'beta', 'gamma', etc.

    Parameters
    ----------
    param : str
        A parameter name like 'alpha1', 'k3', or 'r'

    Returns
    -------
    str
        LaTeX-formatted string like '$\\alpha_{1}$', '$k_3$', or '$r$'
    """
    # Match the name (letters) and optional index (digits) from the string
    m = re.match(r"([A-Za-z]+)(\d*)$", param)
    # If the string does not match the pattern, return it unchanged
    if not m:
        return param
    # Extract the variable name and index
    name, idx = m.groups()
    # Convert to LaTeX-style Greek symbols if applicable (can be extended for more letters)
    greek = {
        "alpha": r"\alpha", "beta": r"\beta", "gamma": r"\gamma",
        "delta": r"\delta", "k": "k", "r": "r"
    }.get(name.lower(), name)
    # Format with or without index
    return f"${greek}_{{{idx}}}$" if idx else f"${greek}$"

#----------------------------------------------------------------------------
## Utility 2:  Plot one bar chart per DataFrame 
#----------------------------------------------------------------------------
def plot_sensitivity_bars(dfs, group_labels):
    """
    Plots a separate bar chart for each variable showing parameter sensitivities.

    Parameters
    ----------
    dfs : list of pd.DataFrame
        Each DataFrame must have:
            - Index: parameter names (e.g., alpha1, beta1)
            - Column: The sensitivity values

    group_labels : list of str
        One label per DataFrame

    Returns
    -------
    None. Displays one plot per DataFrame.
    """
    if len(dfs) != len(group_labels):
        raise ValueError("Mismatch between number of DataFrames and labels")

    # Choose one color per group
    palette = sns.color_palette("tab10", n_colors=len(dfs))
    
    for df, label, color in zip(dfs, group_labels, palette):
        # Format parameter names for x-axis
        labels_latex = [format_param(p) for p in df.index]

        # Plot styling
        # set_plot_style()
        fig, ax = plt.subplots(figsize=(7, 5))

        # Bar plot
        ax.bar(
            x=df.index.astype(str),
            height=df['Scaled'],
            tick_label=labels_latex,
            color=color,
            label=label
        )

        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_xticklabels(labels_latex, rotation=30, ha="right")
        ax.set_xlabel("Parameter")
        ax.set_ylabel("Sensitivity")
        ax.legend(title="Variable")
        ax.grid(axis="y", linestyle="--")
        set_spines_black(ax)
        plt.tight_layout()
        plt.show()
#---------------------------------------------END-----------------------------------------------------