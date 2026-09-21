"""Shared helpers for the joint-UMAP / density / precedence reporting notebooks.

These notebooks all share **one** joint UMAP projection:

    03-compute_coords.ipynb     -> projects papers+teams together once, persists x,y
    04-joint_umap.ipynb         -> topic scatter maps (micro/meso/macro × side/overlay)
    06-density_comparison.ipynb -> KDE density-ratio map + precedence computation
    07-precedence_charts.ipynb  -> violin / dumbbell / diverging / pixel-grid charts
    09-awards.ipynb             -> village awards vs precedence clusters and outsiderness

The cluster-summary notebooks (01, 02) and the scripts run by 08-appendix.ipynb
do not use this package.

Modules
-------
paths       : filesystem paths, palette, seed, persisted-coordinate filenames
coords      : compute / persist / load the joint projection; build plot frames
labels      : label placement strategies (side, overlay, density)
scatter     : topic scatter maps and the papers+teams overlay
density      : KDE density-ratio grid, heatmap rendering, zone classification
precedence  : temporal-precedence computation and the comparison charts
"""
