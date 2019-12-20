from geoopt.manifolds.stereographic.utils import COLORS
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rcParams

# define figure parameters
rcParams["text.latex.preamble"] = r"\usepackage{amsmath}"
rcParams["text.usetex"] = True
rcParams['figure.figsize'] = 7, 7
sns.set_style("white")
lo = -10.0
hi = -lo

# create figure
fig = plt.figure()

# set background color
plt.gca().set_facecolor(COLORS.BACKGROUND_BLUE)

# add horizontal and vertical lines
intervals = np.linspace(lo, hi, 21).tolist()
intervals = [i for i in intervals if abs(i) > 0.3]
plt.hlines(intervals, lo, hi, linestyles="--", colors="gray", linewidths=0.3)
plt.vlines(intervals, lo, hi, linestyles="--", colors="gray", linewidths=0.3)
plt.hlines([0.0], lo, hi, linestyles="--", colors="gray", linewidths=1.0)
plt.vlines([0.0], lo, hi, linestyles="--", colors="gray", linewidths=1.0)

# set up axes
plt.xlim(lo, hi)
plt.ylim(lo, hi)
plt.gca().set_aspect("equal")

# use tight layout
plt.tight_layout()

plt.savefig(f"out/grid-of-geodesics-K-0.0.svg")