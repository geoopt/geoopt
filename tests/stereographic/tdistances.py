import torch
from geoopt.manifolds import StereographicExact

dtype = torch.float64

K = torch.tensor(1.0).to(dtype) * torch.randn(1, dtype=dtype)+0.0001

manifold = StereographicExact(K=K)

R = 1/torch.sqrt(torch.abs(K))


x = torch.randn(2, dtype=dtype) * (0.999*R)
y = torch.randn(2, dtype=dtype) * (0.999*R)

print(manifold.dist(x,y))


