from geoopt.manifolds.stereographic import StereographicExact
from geoopt.manifolds.stereographic import math as stereomath
import torch
import numpy as np
from geoopt.manifolds.stereographic.utils import \
    setup_plot, get_interpolation_Ks, get_img_from_fig, \
    save_img_sequence_as_boomerang_gif, add_K_box, COLORS
from tqdm import tqdm

imgs = []

# for every K of the interpolation sequence
for K in tqdm(get_interpolation_Ks()):

    # create manifold for K
    manifold = StereographicExact(K=K,
                                  float_precision=torch.float64,
                                  keep_sign_fixed=False,
                                  min_abs_K=0.001)

    # set up plot
    fig, plt, (lo, hi) = setup_plot(manifold, lo=-3.0)

    # get manifold properties
    K = manifold.get_K().item()
    R = manifold.get_R().item()

    # create 4 points
    x1 = torch.tensor([ 0.25,  0.75]).to(torch.float64)
    x2 = torch.tensor([ 0.75,  0.05]).to(torch.float64)
    x3 = torch.tensor([-0.15, -0.55]).to(torch.float64)
    x4 = torch.tensor([-0.20,  0.50]).to(torch.float64)

    # create point weights
    w = torch.tensor([1.0, 1.0, 1.0, 1.0]).to(torch.float64)

    # combine points into tensor
    l = [x1,x2,x3,x4]
    l = [x.unsqueeze(dim=0) for x in l]
    X = torch.cat(l, dim=0)

    # compute midpoint
    m = manifold.weighted_midpoint(X, w)

    ZORDER = 2

    # plot geodesics connecting to midpoint
    n_geodesic_eval = 200
    t = torch.linspace(0, 1, n_geodesic_eval)[:, None]
    def plot_geodesic(a, b):
        path = manifold.geodesic(t, a, b)
        plt.plot(*path.t().numpy(), color=COLORS.NEON_PINK, zorder=ZORDER)
    plot_geodesic(x1, m)
    plot_geodesic(x2, m)
    plot_geodesic(x3, m)
    plot_geodesic(x4, m)

    ZORDER = 3

    # plot 4 points
    plt.scatter(*x1, s=3.0*w[0], color=COLORS.TEXT_COLOR, zorder=ZORDER)
    plt.scatter(*x2, s=3.0*w[1], color=COLORS.TEXT_COLOR, zorder=ZORDER)
    plt.scatter(*x3, s=3.0*w[2], color=COLORS.TEXT_COLOR, zorder=ZORDER)
    plt.scatter(*x4, s=3.0*w[3], color=COLORS.TEXT_COLOR, zorder=ZORDER)
    plt.annotate("$x_1$", x1 + torch.tensor([0.02, 0.02]), fontsize=15, color=COLORS.TEXT_COLOR, zorder=ZORDER)
    plt.annotate("$x_2$", x2 + torch.tensor([0.02, 0.02]), fontsize=15, color=COLORS.TEXT_COLOR, zorder=ZORDER)
    plt.annotate("$x_3$", x3 + torch.tensor([-0.15, -0.15]), fontsize=15, color=COLORS.TEXT_COLOR, zorder=ZORDER)
    plt.annotate("$x_4$", x4 + torch.tensor([-0.15, 0.05]), fontsize=15, color=COLORS.TEXT_COLOR, zorder=ZORDER)

    # plot midpoint
    plt.scatter(*m, s=40.0, marker='*', color=COLORS.SHINY_GREEN, zorder=ZORDER)
    plt.annotate("$m_{\kappa}$", m + torch.tensor([0.05, 0.05]), fontsize=15, color=COLORS.SHINY_GREEN, zorder=ZORDER)

    # plot antipode
    if K > 0:
        m_a = stereomath.antipode(m, torch.tensor(K).to(torch.float64))
        if m_a[0].abs() < hi and m_a[1].abs() < hi:
            plt.scatter(*m_a, s=40.0, marker='*', color=COLORS.SHINY_GREEN,
                        zorder=ZORDER)
            plt.annotate("$m^a_{\kappa}$", m_a + torch.tensor([0.05, 0.05]),
                         fontsize=15, color=COLORS.SHINY_GREEN, zorder=ZORDER)

    # add plot title
    #plt.title(r"Midpoint $m_{\kappa}(x_1,x_2,x_3,x_4,1,1,1,1)$")

    # add curvature box
    add_K_box(plt, K)

    # use tight layout
    plt.tight_layout()

    # convert plot to image array
    img = get_img_from_fig(fig, 'tmp/midpoint.png')
    imgs.append(img)

    # close plot to avoid warnings
    plt.close()

# save img sequence as infinite boomerang gif
save_img_sequence_as_boomerang_gif(imgs, 'out/midpoint.gif')
