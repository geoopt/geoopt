import torch
from geoopt.manifolds.stereographic import StereographicExact
from geoopt.optim import RiemannianAdam
from geoopt import ManifoldTensor
from geoopt import ManifoldParameter


# MANIFOLD INSTANTIATION, COMPUTATION OF MANIFOLD QUANTITIES ###################


# create manifold with initial K=-1.0 (Poincaré Ball)
manifold = StereographicExact(K=-1.0,
                              float_precision=torch.float64,
                              keep_sign_fixed=False,
                              min_abs_K=0.001)

# get manifold properties
K = manifold.get_K().item()
R = manifold.get_R().item()
print(f"K={K}")
print(f"R={R}")

# define dimensionality of space
n = 10

def create_random_point(manifold, n):
    x = torch.randn(n)
    x_norm = x.norm(p=2)
    x = x/x_norm * manifold.get_R() * 0.9 * torch.rand(1)
    return x

# create two random points on manifold
x = create_random_point(manifold, n)
y = create_random_point(manifold, n)

# compute their initial distances
initial_dist = manifold.dist(x, y)
print(f"initial_dist={initial_dist.item():.3f}")

# compute the log map of y at x
v = manifold.logmap(x, y)

# compute tangent space norm of v at x (should be equal to initial distance)
v_norm = manifold.norm(x, v)
print(f"v_norm={v_norm.item():.3f}")

# compute the exponential map of v at x (=y)
y2 = manifold.expmap(x, v)
diff = (y-y2).abs().sum()
print(f"diff={diff.item():.3f}")


# CURVATURE OPTIMIZATION #######################################################


# define embedding_optimizer for curvature
curvature_optimizer = torch.optim.SGD([manifold.get_trainable_K()], lr=1e-2)

# set curvature to trainable
manifold.set_K_trainable(True)

# define training loop to optimize curvature until the points have a
# certain target distance
def train_curvature(target_dist):
    for t in range(100):
        curvature_optimizer.zero_grad()
        dist_now = manifold.dist(x,y)
        loss = (dist_now - target_dist).pow(2)
        loss.backward()
        curvature_optimizer.step()

# keep the points x and y fixed,
# train the curvature until the distance is 0.1 more than the initial distance
# --> curvature smaller than initial curvature
train_curvature(initial_dist + 1.0)
print(f"K_smaller={manifold.get_K().item():.3f}")

# keep the points x and y fixed,
# train the curvature until the distance is 0.1 less than the initial distance
# --> curvature greater than initial curvature
train_curvature(initial_dist - 1.0)
print(f"K_larger={manifold.get_K().item():.3f}")


# EMBEDDING OPTIMIZATION #######################################################


# redefine x and y as manifold parameters and assign them to manifold such
# that the embedding_optimizer knows according to which manifold the gradient
# steps have to be performed
x = ManifoldTensor(x, manifold=manifold)
x = ManifoldParameter(x)
y = ManifoldTensor(y, manifold=manifold)
y = ManifoldParameter(y)

# define embedding optimizer and pass embedding parameters
embedding_optimizer = RiemannianAdam([x, y], lr=1e-1)

# define a training loop to optimize the embeddings of x and y
# until they have a certain distance
def train_embeddings(target_dist, x, y):
    for t in range(1000):
        embedding_optimizer.zero_grad()
        dist_now = manifold.dist(x,y)
        loss = (dist_now - target_dist).pow(2)
        loss.backward()
        embedding_optimizer.step()

# print current distance
print(f"dist(x,y)={manifold.dist(x,y).item():.3f}")

# optimize until points have target distance of 4.0
train_embeddings(4.0, x, y)
print(f"dist(x,y)={manifold.dist(x,y).item():.3f}  target:4.0")

# optimize until points have target distance of 2.0
train_embeddings(2.0, x, y)
print(f"dist(x,y)={manifold.dist(x,y).item():.3f}  target:2.0")
