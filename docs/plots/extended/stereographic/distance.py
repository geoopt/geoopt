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

    # create mesh-grid
    coords = None
    if K < 0:
        coords = np.linspace(lo, hi, n_grid_evals)
    else:
        coords = np.linspace(lo, hi, n_grid_evals)
    xx, yy = np.meshgrid(coords, coords)
    grid = np.stack([xx, yy], axis=-1)

    # create point on manifold
    x = torch.tensor([-0.75, -0.2])

    # compute distances to point
    dists = manifold.dist(torch.from_numpy(grid).float(), x)

    # zero-out points outside of Poincaré ball
    if K < 0:
        dist2 = xx ** 2 + yy ** 2
        mask = dist2 <= R ** 2
        dists[(~mask).nonzero()] = np.nan

    # add contour plot
    plt.contourf(
        grid[..., 0],
        grid[..., 1],
        dists.log().numpy(),
        levels=np.linspace(-14, 3, 100),
        cmap="inferno"
    )
    plt.colorbar()

    # convert plot to image array
    img = get_img_from_fig(fig, 'tmp/distance.png')
    imgs.append(img)

# save img sequence as infinite boomerang gif
save_img_sequence_as_boomerang_gif(imgs, 'out/distance.gif')
