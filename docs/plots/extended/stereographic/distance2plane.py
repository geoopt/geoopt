from geoopt import Stereographic
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
from matplotlib import rcParams

rcParams["text.latex.preamble"] = r"\usepackage{amsmath}"
rcParams["text.usetex"] = True

sns.set_style("white")


def add_geodesic_grid(ax: plt.Axes, manifold: Stereographic, line_width=0.1):

    # define geodesic grid parameters
    N_EVALS_PER_GEODESIC = 10000
    STYLE = "--"
    COLOR = "gray"
    LINE_WIDTH = line_width

    # get manifold properties
    K = manifold.k.item()
    R = manifold.radius.item()

    # get maximal numerical distance to origin on manifold
    if K < 0:
        # create point on R
        r = torch.tensor((R, 0.0), dtype=manifold.dtype)
        # project point on R into valid range (epsilon border)
        r = manifold.projx(r)
        # determine distance from origin
        max_dist_0 = manifold.dist0(r).item()
    else:
        max_dist_0 = math.pi * R
    # adjust line interval for spherical geometry
    circumference = 2*math.pi*R

    # determine reasonable number of geodesics
    # choose the grid interval size always as if we'd be in spherical
    # geometry, such that the grid interpolates smoothly and evenly
    # divides the sphere circumference
    n_geodesics_per_circumference = 4 * 6  # multiple of 4!
    n_geodesics_per_quadrant = n_geodesics_per_circumference // 2
    grid_interval_size = circumference / n_geodesics_per_circumference
    if K < 0:
        n_geodesics_per_quadrant = int(max_dist_0 / grid_interval_size)

    # create time evaluation array for geodesics
    if K < 0:
        min_t = -1.2*max_dist_0
    else:
        min_t = -circumference/2.0
    t = torch.linspace(min_t, -min_t, N_EVALS_PER_GEODESIC)[:, None]

    # define a function to plot the geodesics
    def plot_geodesic(gv):
        ax.plot(*gv.t().numpy(), STYLE, color=COLOR, linewidth=LINE_WIDTH)

    # define geodesic directions
    u_x = torch.tensor((0.0, 1.0))
    u_y = torch.tensor((1.0, 0.0))

    # add origin x/y-crosshair
    o = torch.tensor((0.0, 0.0))
    if K < 0:
        x_geodesic = manifold.geodesic_unit(t, o, u_x)
        y_geodesic = manifold.geodesic_unit(t, o, u_y)
        plot_geodesic(x_geodesic)
        plot_geodesic(y_geodesic)
    else:
        # add the crosshair manually for the sproj of sphere
        # because the lines tend to get thicker if plotted
        # as done for K<0
        ax.axvline(0, linestyle=STYLE, color=COLOR, linewidth=LINE_WIDTH)
        ax.axhline(0, linestyle=STYLE, color=COLOR, linewidth=LINE_WIDTH)

    # add geodesics per quadrant
    for i in range(1, n_geodesics_per_quadrant):

        # determine start of geodesic on x/y-crosshair
        x = manifold.geodesic_unit(i*grid_interval_size, o, u_y)
        y = manifold.geodesic_unit(i*grid_interval_size, o, u_x)

        # compute point on geodesics
        x_geodesic = manifold.geodesic_unit(t, x, u_x)
        y_geodesic = manifold.geodesic_unit(t, y, u_y)

        # plot geodesics
        plot_geodesic(x_geodesic)
        plot_geodesic(y_geodesic)
        if K < 0:
            plot_geodesic(-x_geodesic)
            plot_geodesic(-y_geodesic)


lim = 1.1
coords = np.linspace(-lim, lim, 100)
x = torch.tensor([-0.75, 0])
v = torch.tensor([0.1 / 3, 0.0])
xx, yy = np.meshgrid(coords, coords)
dist2 = xx ** 2 + yy ** 2
mask = dist2 <= 1
grid = np.stack([xx, yy], axis=-1)
fig, ax = plt.subplots(1, 2, figsize=(9, 4))

manifold = Stereographic(-1)

dists = manifold.dist2plane(torch.from_numpy(grid).float(), x, v)
dists[(~mask).nonzero()] = np.nan
circle = plt.Circle((0, 0), 1, fill=False, color="b")


ax[0].add_artist(circle)
ax[0].set_xlim(-lim, lim)
ax[0].set_ylim(-lim, lim)
ax[0].set_aspect("equal")
ax[0].contourf(
    grid[..., 0], grid[..., 1], dists.log().numpy(), levels=100, cmap="inferno"
)
add_geodesic_grid(ax[0], manifold, 0.5)
ax[0].set_title(r"$\kappa = -1$")

manifold = Stereographic(1)

dists = manifold.dist2plane(torch.from_numpy(grid).float(), x, v)

ax[1].set_xlim(-lim, lim)
ax[1].set_ylim(-lim, lim)
ax[1].set_aspect("equal")
ax[1].contourf(
    grid[..., 0], grid[..., 1], dists.log().numpy(), levels=100, cmap="inferno"
)
add_geodesic_grid(ax[1], manifold, 0.5)
ax[1].set_title(r"$\kappa = 1$")
fig.suptitle(r"log distance to $\tilde{H}_{a, p}$")
plt.show()
