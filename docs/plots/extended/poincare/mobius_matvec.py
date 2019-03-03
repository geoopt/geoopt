import geoopt.manifolds.poincare.math as pmath
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rcParams

rcParams["text.latex.preamble"] = r"\usepackage{amsmath}"
rcParams["text.usetex"] = True
sns.set_style("white")
x = torch.tensor((-0.25, -0.75)) / 3
M = torch.tensor([[-1, -1.5], [0.2, 0.5]])
M_x = pmath.mobius_matvec(M, x)


circle = plt.Circle((0, 0), 1, fill=False, color="b")
plt.gca().add_artist(circle)
plt.xlim(-1.1, 1.1)
plt.ylim(-1.1, 1.1)
plt.gca().set_aspect("equal")
plt.annotate("x", x - 0.09, fontsize=15)
plt.annotate(
    r"$M=\begin{bmatrix}-1 &-1.5\\.2 &.5\end{bmatrix}$",
    x + torch.tensor([-0.5, 0.5]),
    fontsize=15,
)
plt.annotate(r"$M^\otimes x$", M_x - torch.tensor([0.1, 0.15]), fontsize=15)
plt.arrow(0, 0, *x, width=0.01, color="r")
plt.arrow(0, 0, *M_x, width=0.01, color="b")
plt.title(r"Matrix multiplication $M\otimes x$")
plt.show()
