import torch
import numpy as np
import matplotlib.pyplot as plt
import math
import seaborn as sns
from matplotlib import rcParams
import imageio
from pygifsicle import optimize


class COLORS:
    SHINY_GREEN = "#bfffbf"
    SHINY_BLUE = "#a9e7ff"
    MAT_RED = "#ffc7c7"
    MAT_YELLOW = "#ffffbd"
    NEON_PINK = "#ff3ea0"
    BACKGROUND_BLUE = "#1e0c45"
    TEXT_COLOR = "#ffffff"


def setup_plot(manifold, lo=None, width=7, height=7, grid_line_width=0.3, with_background=True):

    # define figure parameters
    rcParams["text.latex.preamble"] = r"\usepackage{amsmath}"
    rcParams["text.usetex"] = True
    rcParams['figure.figsize'] = width, height
    sns.set_style("white")

    # create figure
    fig = plt.figure()

    # determine manifold properties
    K = manifold.get_K()
    R = manifold.get_R()

    # add circle
    circle = plt.Circle((0, 0), R, fill=with_background, color=COLORS.BACKGROUND_BLUE)
    plt.gca().add_artist(circle)
    if K > 0:
        circle_border = plt.Circle((0, 0), R, fill=False, color="gray",
                                   linewidth=2.0)
        plt.gca().add_artist(circle_border)


    # add background color
    if K > 0 and with_background:
        plt.gca().set_facecolor(COLORS.BACKGROUND_BLUE)

    # set up plot axes and aspect ratio
    if lo==None:
        if K < 0:
            lo = -R - 0.1
        else:
            lo = -2 * R - 0.1
    hi = -lo
    plt.xlim(lo, hi)
    plt.ylim(lo, hi)
    plt.gca().set_aspect("equal")

    # add grid of geodesics
    add_geodesic_grid(plt, manifold, lo, hi, line_width=grid_line_width)

    return fig, plt, (lo, hi)


def get_maximal_numerical_distance(manifold):
    # get manifold properties
    K = manifold.get_K()
    R = manifold.get_R()
    # determine maximal distance from origin
    dist0 = None
    if K < 0:
        # create point on R
        r = torch.tensor((R, 0.0), dtype=torch.float64)
        # project point on R into valid range (epsilon border)
        r = manifold.projx(r)
        # determine distance from origin
        dist0 = manifold.dist0(r).item()
    else:
        dist0 = math.pi * R
    return dist0


def add_geodesic_grid(plt, manifold, lo, hi, line_width = 0.1):

    # define geodesic grid parameters
    N_EVALS_PER_GEODESIC = 10000
    STYLE = "--"
    COLOR = "gray"
    LINE_WIDTH = line_width

    # get manifold properties
    K = manifold.get_K().item()
    R = manifold.get_R().item()

    # get maximal numerical distance to origin on manifold
    max_dist_0 = get_maximal_numerical_distance(manifold)

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
    min_t = 0.0
    if K < 0:
        min_t = -1.2*max_dist_0
    else:
        min_t = -circumference/2.0
    t = torch.linspace(min_t, -min_t, N_EVALS_PER_GEODESIC)[:, None]

    # define a function to plot the geodesics
    def plot_geodesic(gv, **kwargs):
        plt.plot(*gv.t().numpy(), STYLE, color=COLOR, linewidth=LINE_WIDTH)

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
        plt.plot([lo, hi], [0, 0], STYLE, color=COLOR, linewidth=LINE_WIDTH)
        plt.plot([0, 0], [lo, hi], STYLE, color=COLOR, linewidth=LINE_WIDTH)

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


def add_K_box(plt, K):
    props = dict(pad=10.0, facecolor='white', edgecolor='black', linewidth=0.5)
    plt.gca().text(0.05, 0.95, f"$\kappa={K:1.3f}$",
                   transform=plt.gca().transAxes,
                   verticalalignment='top', bbox=props)


def get_interpolation_Ks(num=200):
    # S-curve going through zero
    x = np.linspace(-1.0, 1.7, num=num)
    Ks = (x**3).tolist()
    Ks = [K for K in Ks if abs(K) > 0.001]
    return Ks


# define a function which returns an image as numpy array from figure
def get_img_from_fig(fig, tmp_file, dpi=90):
    fig.savefig(tmp_file, dpi=dpi)
    img = imageio.imread(tmp_file)
    return img


def save_img_sequence_as_boomerang_gif(imgs, out_filename, fps=24):

    # invert image sequence
    l2 = list(imgs)
    l2.reverse()
    boomerang = imgs + l2

    # save gif
    imageio.mimsave(out_filename, boomerang, fps=fps)

    # optimize gif file size
    optimize(out_filename)


def save_img_sequence_as_gif(imgs, out_filename, fps=24):

    # save gif
    imageio.mimsave(out_filename, imgs, fps=fps)

    # optimize gif file size
    optimize(out_filename)