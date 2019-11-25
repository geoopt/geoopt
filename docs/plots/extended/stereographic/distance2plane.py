from geoopt.manifolds.stereographic import StereographicExact
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rcParams


rcParams["text.latex.preamble"] = r"\usepackage{amsmath}"
rcParams["text.usetex"] = True


# POINCARE BALL DISTANCE PLOT ##################################################


poincare_ball = StereographicExact(K=-1.0,
                                   float_precision=torch.float64,
                                   keep_sign_fixed=False,
                                   min_abs_K=0.001)

sns.set_style("white")
radius = 1
coords = np.linspace(-radius, radius, 100)
x = torch.tensor([-0.75, 0])
v = torch.tensor([0.5, -1 / 3])
xx, yy = np.meshgrid(coords, coords)
dist2 = xx ** 2 + yy ** 2
mask = dist2 <= radius ** 2
grid = np.stack([xx, yy], axis=-1)
dists = poincare_ball.dist2plane(torch.from_numpy(grid).float(), x, v)
dists[(~mask).nonzero()] = np.nan
circle = plt.Circle((0, 0), 1, fill=False, color="b")
plt.gca().add_artist(circle)
plt.xlim(-1.1, 1.1)
plt.ylim(-1.1, 1.1)
plt.gca().set_aspect("equal")
plt.contourf(
    grid[..., 0], grid[..., 1], dists.log().numpy(), levels=100, cmap="inferno"
)
plt.colorbar()
plt.scatter(*x, color="g")
plt.arrow(*x, *v, color="g", width=0.01)
plt.title(r"log distance to $\tilde{H}_{a, p}$")
plt.show()


# SPROJ OF SPHERE DISTANCE PLOT ################################################


sproj_of_sphere = StereographicExact(K=1.0,
                                     float_precision=torch.float64,
                                     keep_sign_fixed=False,
                                     min_abs_K=0.001)

sns.set_style("white")
radius = 1
coords = np.linspace(-2*radius, 2*radius, 100)
x = torch.tensor([-0.75, 0])
v = torch.tensor([0.5, -1 / 3])
xx, yy = np.meshgrid(coords, coords)
dist2 = xx ** 2 + yy ** 2
grid = np.stack([xx, yy], axis=-1)
dists = sproj_of_sphere.dist2plane(torch.from_numpy(grid).float(), x, v)
circle = plt.Circle((0, 0), 1, fill=False, color="b")
plt.gca().add_artist(circle)
lo = -2*radius-0.1
hi = -lo
plt.xlim(lo, hi)
plt.ylim(lo, hi)
plt.gca().set_aspect("equal")
plt.contourf(
    grid[..., 0], grid[..., 1], dists.numpy(), levels=100, cmap="inferno"
)
plt.colorbar()
plt.scatter(*x, color="g")
plt.arrow(*x, *v, color="g", width=0.01)
plt.title(r"distance to $\tilde{H}_{a, p}$")
plt.show()