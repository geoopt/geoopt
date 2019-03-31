import geoopt.manifolds.poincare.math as pmath
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rcParams

rcParams["text.latex.preamble"] = r"\usepackage{amsmath}"
rcParams["text.usetex"] = True


sns.set_style("white")

x = torch.tensor((-0.25, -0.75))
v1 = torch.tensor((np.sin(np.pi / 3), np.cos(np.pi / 3))) / 5
v2 = torch.tensor((np.sin(-np.pi / 3), np.cos(np.pi / 3))) / 5
y = torch.tensor((0.65, -0.55))
t = torch.linspace(0, 1)
xy = pmath.logmap(x, y)
path = pmath.geodesic(t[:, None], x, y)
yv1 = pmath.parallel_transport(x, y, v1)
yv2 = pmath.parallel_transport(x, y, v2)


circle = plt.Circle((0, 0), 1, fill=False, color="b")
plt.gca().add_artist(circle)
plt.xlim(-1.1, 1.1)
plt.ylim(-1.1, 1.1)
plt.gca().set_aspect("equal")
plt.annotate("x", x - 0.07, fontsize=15)
plt.annotate("y", y - 0.07, fontsize=15)
plt.annotate(r"$\vec{v}$", x + torch.tensor([0.3, 0.5]), fontsize=15)
plt.arrow(*x, *v1, width=0.01, color="r")
plt.arrow(*x, *xy, width=0.01, color="g")
plt.arrow(*x, *v2, width=0.01, color="b")
plt.arrow(*y, *yv1, width=0.01, color="r")
plt.arrow(*y, *yv2, width=0.01, color="b")
plt.plot(*path.t().numpy(), color="g")
plt.title(r"parallel transport $P^c_{x\to y}$")
plt.show()
