from geoopt.manifolds.stereographic import StereographicExact
import torch
import numpy as np
from geoopt.manifolds.stereographic.utils import \
    setup_plot, get_interpolation_Ks, get_img_from_fig, \
    save_img_sequence_as_boomerang_gif
from tqdm import tqdm

n_grid_evals = 1000
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
    v1 = torch.tensor((np.sin(np.pi / 3), np.cos(np.pi / 3))) / 5
    v2 = torch.tensor((np.sin(-np.pi / 3), np.cos(np.pi / 3))) / 5
    y = torch.tensor((0.65, -0.55))
    t = torch.linspace(0, 1)

    xy = manifold.logmap(x, y)
    path = manifold.geodesic(t[:, None], x, y)
    yv1 = manifold.transp(x, y, v1)
    yv2 = manifold.transp(x, y, v2)

    plt.annotate("$x$", x - 0.12, fontsize=15)
    plt.annotate("$y$", y - 0.12, fontsize=15)
    plt.annotate(r"$\vec{v}$", x + xy + 0.07, fontsize=15)

    plt.arrow(*x, *v1, width=0.01, color="r")
    plt.arrow(*x, *xy, width=0.01, color="g")
    plt.arrow(*x, *v2, width=0.01, color="b")
    plt.arrow(*y, *yv1, width=0.01, color="r")
    plt.arrow(*y, *yv2, width=0.01, color="b")
    plt.plot(*path.t().numpy(), color="g")

    plt.title(r"Parallel Transport $P^\kappa_{x\to y}$")

    # convert plot to image array
    img = get_img_from_fig(fig, 'tmp/parallel-transport.png')
    imgs.append(img)

# save img sequence as infinite boomerang gif
save_img_sequence_as_boomerang_gif(imgs, 'out/parallel-transport.gif')
