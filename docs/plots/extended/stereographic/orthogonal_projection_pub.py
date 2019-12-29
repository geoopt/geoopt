import torch
import geoopt.manifolds.poincare.math as pmath
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import rcParams

# ORTHOGONAL PROJECTION ########################################################

# define Poincaré ball parameters
DTYPE = torch.float64
o = torch.tensor((0.0, 0.0), dtype=DTYPE)
c = torch.tensor(0.5, dtype=DTYPE)
R = 1.0/torch.sqrt(torch.abs(c))

# create gyrovector a which geodesic goes through
a = torch.tensor((0.6, -0.3), dtype=DTYPE)
a = a/a.norm(p=2)*0.8*R

# create gyrovector x to orthogonally project onto geodesic going through a
x = torch.tensor((-0.3, 0.7), dtype=DTYPE)
x = x/x.norm(p=2)*0.6*R

# define Lorentz factor (not in pmath!)
def gamma_x(x, c=1.0):
    return 1.0/torch.sqrt(1.0-c*(x.norm(p=2).pow(2)))

# define orthogonal projection of x onto a
def orthogonal_proj(x,a):
    cos_theta = (x*a).sum() / (x.norm(p=2)*a.norm(p=2))
    sin_theta = torch.sqrt(1.0-cos_theta.pow(2))
    h_M = gamma_x(x, c=c).pow(2) * x.norm(p=2) * sin_theta
    numerator = 2.0 * h_M
    denominator = 1.0 + torch.sqrt(1.0 + 4.0*c*h_M.pow(2))
    h_norm_sq = (numerator/denominator).pow(2)
    x_p_norm = torch.sqrt((1.0/torch.sqrt(c)) *
                          pmath.mobius_sub(torch.sqrt(c)*x.norm(p=2).pow(2),
                                           torch.sqrt(c)*h_norm_sq, c=c))
    s = -1.0 if (a * x).sum() < 0 else 1.0
    x_p = s*(a / a.norm(p=2)) * x_p_norm
    return x_p

# perform orthogonal projection
x_p = orthogonal_proj(x, a)

# ORTHOGONALITY CHECK ##########################################################

# check orthogonality in tangent space
u = pmath.logmap(x_p, x, c=c)
v = pmath.logmap(x_p, a, c=c)
print('Inner prodct should be zero:')
print((u*v).sum().item())

# PLOT #########################################################################

rcParams["text.latex.preamble"] = r"\usepackage{amsmath}"
rcParams["text.usetex"] = True
sns.set_style("white")

circle = plt.Circle((0, 0), R, fill=False, color="b")
plt.gca().add_artist(circle)
B = (R+0.1)
plt.xlim(-B, B)
plt.ylim(-B, B)
plt.gca().set_aspect("equal")

# plot geodesics
def plot_geodesic(a, b):
    n_approx = 1000
    l = torch.linspace(-100.0, 100.0, n_approx)
    geodesic_points = torch.zeros((2,n_approx))
    for i in range(n_approx):
        geodesic_points[:,i] = pmath.geodesic(l[i], a, b, c=c)
    plt.plot(geodesic_points[0,:], geodesic_points[1,:], ':', color='black')
plot_geodesic(x, x_p)
plot_geodesic(o, a)

plt.annotate("$\mathbf{0}$", o - 0.09, fontsize=15)
plt.annotate("$\mathbf{x}$", x - 0.09, fontsize=15)
plt.arrow(0, 0, *x, width=0.01, color="r")
plt.annotate("$\mathbf{a}$", a - 0.09, fontsize=15)
plt.arrow(0, 0, *a, width=0.01, color="b")
plt.annotate("$\mathbf{x}_p$", x_p - 0.09, fontsize=15)
plt.arrow(0, 0, *x_p, width=0.01, color="g")

plt.title(r"Orthogonal Projection of $\mathbf{x}$ onto Geodesic through Origin "
          r"$\mathbf{0}$ and $\mathbf{a}$")
plt.show()
