from geoopt.manifolds.stereographic import StereographicExact
import torch
from geoopt.manifolds.stereographic.utils import setup_plot

for K in [-1.0, 1.0]:

    manifold = StereographicExact(K=K,
                                  float_precision=torch.float64,
                                  keep_sign_fixed=False,
                                  min_abs_K=0.001)

    fig, plt, (lo, hi) = setup_plot(manifold, lo=-2.0)

    #plt.title(f"Grid of Geodesics at Equidistant Intervals ($\kappa={K:1.3f}$)")

    # use tight layout
    plt.tight_layout()

    plt.savefig(f"out/grid-of-geodesics-K-{K:1.1f}.svg")
