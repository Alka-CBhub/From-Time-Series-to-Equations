## plot_utils.py (Total = 2)
#------------------------------

# Import Necessary Library 
import matplotlib as mpl

#-----------------------------------------------------------------------------------------
##  Utility 1: Global rcParams for plotting
#----------------------------------------------------------------------------
def set_plot_style(dpi=500, grid=True, autolayout=True):
    """
    Call once at the top of any script to apply global rcParams for plotting.
    
    Parameters:
    - dpi (int): Resolution of figures.
    - grid (bool): Whether to show grid lines on all plots.
    - autolayout (bool): Whether to automatically adjust layout to avoid clipping.
    """
    mpl.rcParams.update({
        # Font settings
        "text.usetex": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,

        # Axes labels and titles
        "axes.labelsize": 16,
        "axes.labelweight": "bold",
        "axes.titlesize": 16,
        "axes.titleweight": "bold",
        "axes.grid": grid,

        # Tick labels
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "xtick.color": "black",
        "ytick.color": "black",

        # Tick width/thickness
        "xtick.major.width": 1.5,
        "ytick.major.width": 1.5,
        "xtick.minor.width": 1.2,
        "ytick.minor.width": 1.2,
        "xtick.direction": "in",
        "ytick.direction": "in",

        # Legend
        "legend.fontsize": 13,

        # Line and marker settings
        "lines.linewidth": 2.5,
        "lines.markersize": 8,

        # Grid appearance
        "grid.linestyle": "--",
        "grid.alpha": 0.6,

        # DPI and layout
        "figure.dpi": dpi,
        "figure.autolayout": autolayout,

         # Make *all* text bold
        "font.weight": "bold",
    })
#--------------------------------------------------------------------------------------------------
## Format all spines of a given Matplotlib Axes object
def set_spines_black(ax):
    """
    Set all spines of a Matplotlib Axes instance to black with linewidth=2.
    
    Parameters:
    - ax: A matplotlib Axes object.
    
    Returns:
    - ax: The same Axes object (to allow for chaining).
    """
    for spine in ax.spines.values():
        spine.set_linewidth(2)  # Bold
        spine.set_edgecolor('k') # Black
    return ax
#---------------------------------------------END-----------------------------------------------------