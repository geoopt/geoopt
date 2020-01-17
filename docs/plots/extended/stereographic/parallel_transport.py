from geoopt import Stereographic
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rcParams


def add_geodesic_grid(ax: plt.Axes, manifold: Stereographic, line_width=0.1):
    import math
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

rcParams["text.latex.preamble"] = r"\usepackage{amsmath}"
rcParams["text.usetex"] = True

sns.set_style("white")

x = torch.tensor((-0.25, -0.75))
xv1 = torch.tensor((np.sin(np.pi / 3), np.cos(np.pi / 3))) / 5
xv2 = torch.tensor((np.sin(-np.pi / 3), np.cos(np.pi / 3))) / 5
t = torch.linspace(0, 1, 10)[:, None]


def plot_gv(a, gv, **kwargs):
    a.plot(*gv.t().numpy(), **kwargs)
    a.arrow(*gv[-2], *(gv[-1] - gv[-2]), width=0.01, **kwargs)


fig, ax = plt.subplots(1, 2, figsize=(9, 4))
fig.suptitle(r"parallel transport $P_{x\to y}$")

manifold = Stereographic(-1)

y = torch.tensor((0.65, -0.55))
xy = manifold.logmap(x, y)
path = manifold.geodesic(t, x, y)
yv1 = manifold.transp(x, y, xv1)
yv2 = manifold.transp(x, y, xv2)


circle = plt.Circle((0, 0), 1, fill=False, color="b")


ax[0].add_artist(circle)
ax[0].set_xlim(-lim, lim)
ax[0].set_ylim(-lim, lim)
ax[0].set_aspect("equal")
add_geodesic_grid(ax[0], manifold, 0.5)

ax[0].annotate("x", x - 0.09, fontsize=15)
ax[0].annotate("y", y - 0.09, fontsize=15)
ax[0].annotate(r"$\vec{v}$", x + torch.tensor([0.3, 0.5]), fontsize=15)
ax[0].arrow(*x, *xv1, width=0.01, color="r")
ax[0].arrow(*x, *xy, width=0.01, color="g")
ax[0].arrow(*x, *xv2, width=0.01, color="b")
ax[0].arrow(*y, *yv1, width=0.01, color="r")
ax[0].arrow(*y, *yv2, width=0.01, color="b")

ax[0].plot(*path.t().numpy(), color="g")
ax[0].set_title(r"$\kappa = -1$")


manifold = Stereographic(1)

y = torch.tensor((0.65, -0.55))
xy = manifold.logmap(x, y)
path = manifold.geodesic(t, x, y)
yv1 = manifold.transp(x, y, xv1)
yv2 = manifold.transp(x, y, xv2)


ax[1].set_xlim(-lim, lim)
ax[1].set_ylim(-lim, lim)
ax[1].set_aspect("equal")
add_geodesic_grid(ax[1], manifold, 0.5)

ax[1].annotate("x", x - 0.09, fontsize=15)
ax[1].annotate("y", y - 0.09, fontsize=15)
ax[1].annotate(r"$\vec{v}$", torch.tensor([0.5, -0.9]), fontsize=15)
ax[1].arrow(*x, *xv1, width=0.01, color="r")
ax[1].arrow(*x, *xy, width=0.01, color="g")
ax[1].arrow(*x, *xv2, width=0.01, color="b")
ax[1].arrow(*y, *yv1, width=0.01, color="r")
ax[1].arrow(*y, *yv2, width=0.01, color="b")

ax[1].plot(*path.t().numpy(), color="g")
ax[1].set_title(r"$\kappa = 1$")

plt.show()
