import geoopt.manifolds.poincare.math as pmath
from matplotlib import rcParams
import torch
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("white")
rcParams["text.latex.preamble"] = r"\usepackage{amsmath}"
rcParams["text.usetex"] = True
x = torch.tensor((-0.25, -0.75)) / 3
f_x = pmath.mobius_fn_apply(torch.sigmoid, x)


circle = plt.Circle((0, 0), 1, fill=False, color="b")
plt.gca().add_artist(circle)
plt.xlim(-1.1, 1.1)
plt.ylim(-1.1, 1.1)
plt.gca().set_aspect("equal")
plt.annotate("x", x - 0.09, fontsize=15)
plt.annotate(
    r"$\sigma(x)=\frac{1}{1+e^{-x}}$", x + torch.tensor([-0.7, 0.5]), fontsize=15
)
plt.annotate(r"$\sigma^\otimes(x)$", f_x - torch.tensor([0.1, 0.15]), fontsize=15)
plt.arrow(0, 0, *x, width=0.01, color="r")
plt.arrow(0, 0, *f_x, width=0.01, color="b")
plt.title(r"Mobius function (sigmoid) apply $\sigma^\otimes(x)$")
plt.show()
