from geoopt.manifolds.stereographic import StereographicExact
import torch
import numpy as np
from geoopt.manifolds.stereographic.utils import \
    setup_plot, get_interpolation_Ks, get_img_from_fig, \
    save_img_sequence_as_boomerang_gif
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

    x = torch.tensor((-0.25, -0.75))
    xv1 = torch.tensor((np.sin(np.pi / 3), np.cos(np.pi / 3))) / 5
    xv2 = torch.tensor((np.sin(-np.pi / 3), np.cos(np.pi / 3))) / 5
    t = torch.linspace(0, 1, 200)[:, None]

    y = torch.tensor((0.65, -0.55))
    xy = manifold.logmap(x, y)
    path = manifold.geodesic(t, x, y)
    yv1 = manifold.transp(x, y, xv1)
    yv2 = manifold.transp(x, y, xv2)

    xgv1 = manifold.geodesic_unit(t, x, xv1)
    xgv2 = manifold.geodesic_unit(t, x, xv2)

    ygv1 = manifold.geodesic_unit(t, y, yv1)
    ygv2 = manifold.geodesic_unit(t, y, yv2)


    def plot_gv(gv, **kwargs):
        plt.plot(*gv.t().numpy(), **kwargs)
        plt.arrow(*gv[-2], *(gv[-1] - gv[-2]), width=0.01, **kwargs)

    plt.annotate("$x$", x - 0.09, fontsize=15)
    plt.annotate("$y$", y - 0.09, fontsize=15)
    plt.annotate(r"$\vec{v}$", x + xy, fontsize=15)
    plot_gv(xgv1, color="r")
    plot_gv(xgv2, color="b")
    plt.arrow(*x, *xy, width=0.01, color="g")
    plot_gv(ygv1, color="r")
    plot_gv(ygv2, color="b")
    plt.plot(*path.t().numpy(), color="g")

    plt.title(r"Gyrovector Parallel Transport $P^\kappa_{x\to y}$")

    # convert plot to image array
    img = get_img_from_fig(fig, 'tmp/gyrovector-parallel-transport.png')
    imgs.append(img)

# save img sequence as infinite boomerang gif
save_img_sequence_as_boomerang_gif(imgs, 'out/gyrovector-parallel-transport.png')
