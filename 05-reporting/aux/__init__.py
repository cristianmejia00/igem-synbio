"""Shared helpers for the joint-UMAP / density / precedence reporting notebooks.

The notebook ``reporting_enhanced.ipynb`` was split into focused units that all
share **one** joint UMAP projection:

    compute_coords.ipynb     -> projects papers+teams together once, persists x,y
    joint_umap.ipynb         -> topic scatter maps (micro/meso/macro × side/overlay)
    density_comparison.ipynb -> KDE density-ratio map + precedence computation
    precedence_charts.ipynb  -> violin / dumbbell / diverging precedence charts

Modules
-------
paths       : filesystem paths, palette, seed, persisted-coordinate filenames
coords      : compute / persist / load the joint projection; build plot frames
labels      : label placement strategies (side, overlay, density)
scatter     : topic scatter maps and the papers+teams overlay
density      : KDE density-ratio grid, heatmap rendering, zone classification
precedence  : temporal-precedence computation and the comparison charts
"""
