from geoopt.manifolds.stereographic import StereographicExact
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rcParams

rcParams["text.latex.preamble"] = r"\usepackage{amsmath}"
rcParams["text.usetex"] = True


# POINCARE BALL MOBIUS ADD PLOT ################################################


poincare_ball = StereographicExact(K=-1.0,
                                   float_precision=torch.float64,
                                   keep_sign_fixed=False,
                                   min_abs_K=0.001)

sns.set_style("white")
x = torch.tensor((-0.25, -0.75)) / 2
y = torch.tensor((0.65, -0.55)) / 2
x_plus_y = poincare_ball.mobius_add(x, y)
circle = plt.Circle((0, 0), 1, fill=False, color="b")
plt.gca().add_artist(circle)
plt.xlim(-1.1, 1.1)
plt.ylim(-1.1, 1.1)
plt.gca().set_aspect("equal")
plt.annotate("x", x - 0.09, fontsize=15)
plt.annotate("y", y - 0.09, fontsize=15)
plt.annotate(r"$x\oplus y$", x_plus_y - torch.tensor([0.1, 0.15]), fontsize=15)
plt.arrow(0, 0, *x, width=0.01, color="r")
plt.arrow(0, 0, *y, width=0.01, color="g")
plt.arrow(0, 0, *x_plus_y, width=0.01, color="b")
plt.title(r"Addition $x\oplus y$")
plt.show()


# SPROJ OF SPHERE MOBIUS ADD PLOT ##############################################


sproj_of_sphere = StereographicExact(K=1.0,
                                     float_precision=torch.float64,
                                     keep_sign_fixed=False,
                                     min_abs_K=0.001)

sns.set_style("white")
x = torch.tensor((-0.25, -0.75)) / 2
y = torch.tensor((0.65, -0.55)) / 2
x_plus_y = sproj_of_sphere.mobius_add(x, y)
radius = 1.0
circle = plt.Circle((0, 0), 1, fill=False, color="b")
plt.gca().add_artist(circle)
lo = -2*radius-0.1
hi = -lo
plt.xlim(lo, hi)
plt.ylim(lo, hi)
plt.gca().set_aspect("equal")
plt.annotate("x", x - 0.09, fontsize=15)
plt.annotate("y", y - 0.09, fontsize=15)
plt.annotate(r"$x\oplus y$", x_plus_y - torch.tensor([0.1, 0.15]), fontsize=15)
plt.arrow(0, 0, *x, width=0.01, color="r")
plt.arrow(0, 0, *y, width=0.01, color="g")
plt.arrow(0, 0, *x_plus_y, width=0.01, color="b")
plt.title(r"Addition $x\oplus y$")
plt.show()
