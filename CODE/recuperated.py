#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# PARAMETERS (use same numbers as the original image)
N = 3000        # seeds in central cell
L = 1.0         # box length
rng = np.random.default_rng(0)

# 1) central seed points inside [0, L)^3
seed_coords = rng.random((N, 3)) * L

# 2) build 3x3x3 tiled cloud (central + 26 neighbours)
tiles = [(sx, sy, sz) for sx in (-1, 0, 1) for sy in (-1, 0, 1) for sz in (-1, 0, 1)]
tiled_coords = []
tile_shifts = []
for sx, sy, sz in tiles:
    shift = np.array([sx, sy, sz], dtype=float) * L
    for p in seed_coords:
        tiled_coords.append(p + shift)
        tile_shifts.append((sx, sy, sz))
tiled_coords = np.asarray(tiled_coords)
tile_shifts = np.asarray(tile_shifts)

# Separate central cell and ghosts
is_central = np.all(tile_shifts == (0, 0, 0), axis=1)
central = tiled_coords[is_central]
ghosts = tiled_coords[~is_central]

# Color by x coordinate in the central cell
xvals = central[:, 0]
norm = plt.Normalize(vmin=xvals.min(), vmax=xvals.max())
cmap = plt.get_cmap('viridis')
colors = cmap(norm(xvals))

# PLOTTING — reproduce layout and styling from the screenshot
fig = plt.figure(figsize=(8.5, 7.5))
ax = fig.add_subplot(111, projection='3d')

# Plot ghost tiles (faint grey points)
ax.scatter(ghosts[:, 0], ghosts[:, 1], ghosts[:, 2],
           c='lightgrey', s=3, alpha=0.20, depthshade=False)

# Plot central cell colored by x
sc = ax.scatter(central[:, 0], central[:, 1], central[:, 2],
                c=xvals, cmap='viridis', s=8, depthshade=False)

# Axis labels and title (UK English spelling)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Periodic Voronoi Network - central cell + ghost neighbours')

# Equal aspect ratio (use peak-to-peak ranges)
ax.set_box_aspect([np.ptp(tiled_coords[:, 0]),
                   np.ptp(tiled_coords[:, 1]),
                   np.ptp(tiled_coords[:, 2])])

# Colorbar with label
cbar = fig.colorbar(sc, ax=ax, shrink=0.62, pad=0.12)
cbar.set_label('x coordinate')

# Grid, ticks similar to screenshot
ax.grid(True)
plt.tight_layout()
plt.show()
