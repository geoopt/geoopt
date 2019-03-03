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
xv1 = torch.tensor((np.sin(np.pi / 3), np.cos(np.pi / 3))) / 5
xv2 = torch.tensor((np.sin(-np.pi / 3), np.cos(np.pi / 3))) / 5
t = torch.linspace(0, 1, 10)[:, None]

y = torch.tensor((0.65, -0.55))
xy = pmath.logmap(x, y)
path = pmath.geodesic(t, x, y)
yv1 = pmath.parallel_transport(x, y, xv1)
yv2 = pmath.parallel_transport(x, y, xv2)

xgv1 = pmath.geodesic_unit(t, x, xv1)
xgv2 = pmath.geodesic_unit(t, x, xv2)

ygv1 = pmath.geodesic_unit(t, y, yv1)
ygv2 = pmath.geodesic_unit(t, y, yv2)


def plot_gv(gv, **kwargs):
    plt.plot(*gv.t().numpy(), **kwargs)
    plt.arrow(*gv[-2], *(gv[-1] - gv[-2]), width=0.01, **kwargs)


circle = plt.Circle((0, 0), 1, fill=False, color="b")
plt.gca().add_artist(circle)
plt.xlim(-1.1, 1.1)
plt.ylim(-1.1, 1.1)
plt.gca().set_aspect("equal")
plt.annotate("x", x - 0.09, fontsize=15)
plt.annotate("y", y - 0.09, fontsize=15)
plt.annotate(r"$\vec{v}$", x + torch.tensor([0.3, 0.5]), fontsize=15)
plot_gv(xgv1, color="r")
plot_gv(xgv2, color="b")
plt.arrow(*x, *xy, width=0.01, color="g")
plot_gv(ygv1, color="r")
plot_gv(ygv2, color="b")

plt.plot(*path.t().numpy(), color="g")
plt.title(r"gyrovector parallel transport $P_{x\to y}$")
plt.show()
