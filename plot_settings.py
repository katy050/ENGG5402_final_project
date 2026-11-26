"""Shared Matplotlib defaults and overrides for each simulation plot."""

from matplotlib import rc_context


PLOT_DEFAULTS = {
    "font.size": 16,
    "axes.labelsize": 16,
    "figure.constrained_layout.use": False,
    "lines.linewidth": 1.5,
    "lines.markersize": 10,
    "savefig.dpi": 300,
    "savefig.facecolor": "white",
    "savefig.transparent": True,
    "savefig.bbox": "tight",
    "savefig.pad_inches": .03,
}

# The path figure and replay share the same workspace axes.
PATH_AXES = {
    "axes.grid": True,
    "grid.linewidth": .6,
    "xtick.labelsize":  16,
    "ytick.labelsize":  16,
    "xtick.major.size": 3,
    "ytick.major.size": 3,
    "legend.loc": "lower center",
    "legend.frameon": False,
    "legend.fontsize":  16,
}

PLOT_STYLES = {
    "map": {
        "figure.figsize": (13.5, 5.6),
        "figure.titlesize": 24,
        "axes.titlesize": 16,
        "axes.titlepad":  16,
        "axes.grid": True,
        "grid.linewidth": .6,
        "grid.alpha": .6,
        "xtick.labelsize":  16,
        "ytick.labelsize":  16,
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "legend.loc": "lower center",
        "legend.frameon": False,
        "legend.fontsize":  16,
        "legend.columnspacing": .8,
        "legend.handlelength": 1.25,
        "legend.handletextpad": .5,
        "savefig.dpi": 300,
    },
    "occupancy": {
        "figure.figsize": (11.5, 4.4),
        "figure.constrained_layout.use": True,
        "figure.titlesize": 24,
        "image.cmap": "viridis",
    },
    "times": {
        "figure.figsize": (9.2, 5.7),
        "figure.constrained_layout.use": True,
        "axes.titlesize": 24,
        "axes.titlepad": 16,
        "axes.grid": True,
        "axes.grid.axis": "x",
        "grid.alpha": .8,
    },
    "paths": {
        **PATH_AXES,
        "figure.figsize": (8.5, 5.5),
        "axes.titlesize": 16,
        "legend.loc": "center",
    },
    "path_animation": {
        **PATH_AXES,
        "figure.figsize": (8.5, 5.5),
        "figure.dpi": 100,
        "figure.subplot.top": .84,
        "figure.titlesize": 16,
    },
    "animation": {
        **PATH_AXES,
        "figure.figsize": (8.4, 10.8),
        "figure.dpi": 300,
        "figure.subplot.left": .09,
        "figure.subplot.right": .98,
        "figure.subplot.top": .876,
        "figure.subplot.bottom": .10,
        "figure.subplot.hspace": .33,
        "figure.subplot.wspace": .08,
        "figure.titlesize": 24,
        "axes.titlesize": 16,
        "axes.labelsize": 16,
        "legend.columnspacing": 1.3,
    },
}


def apply_plot_style(plot):
    """Apply the shared defaults and the selected plot's overrides while rendering."""
    return rc_context(PLOT_DEFAULTS | PLOT_STYLES[plot])
