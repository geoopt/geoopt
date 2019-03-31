import geoopt.manifolds.poincare.math as pmath
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rcParams

rcParams["text.latex.preamble"] = r"\usepackage{amsmath}"
rcParams["text.usetex"] = True

sns.set_style("white")
radius = 1
coords = np.linspace(-radius, radius, 100)
x = torch.tensor([-0.75, 0])
v = torch.tensor([0.1 / 3, -1 / 3])
xx, yy = np.meshgrid(coords, coords)
dist2 = xx ** 2 + yy ** 2
mask = dist2 <= radius ** 2
grid = np.stack([xx, yy], axis=-1)
dists = pmath.dist2plane(torch.from_numpy(grid).float(), x, v)
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
