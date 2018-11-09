# geoopt

[![Build Status](https://travis-ci.com/ferrine/geoopt.svg?branch=master)](https://travis-ci.com/ferrine/geoopt)
<a href="https://github.com/ambv/black"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>

Manifold aware `pytorch.optim`

Unofficial implementation for ["Riemannian Adaptive Optimization Methods"](https://openreview.net/forum?id=r1eiqi09K7) ICLR2019 (they are likely to be accepted).

Here gonna be all easy to implement optimizers starting from RSGD and ending up with adaptive methods

## What is done so far
Work is in progress but you can already use this. Note that API might change in future releases.

### Tensors

* `geoopt.ManifoldTensor` -- just as torch.Tensor with additional `manifold` keyword argument.
* `geoopt.ManifoldParameter` -- same as above, recognized in `torch.nn.Module.parameters` as correctly subclassed.

All above containers have special methods to work with them as with points on certain manifold

* `.proj_()` -- inplace projection on the manifold.
* `.inner(u, v=None)` -- inner product at this point for two **tangent** vectors at this point. The passed vectors are not projected, they are assumed to be already projected.
* `.retr(u, t)` -- retraction map following vector `u` and time `t`
* `.transp(u, v, t)` -- transport vector `v` with direction `u` for time `t`
* `.proju(u)` -- project vector `u` on the tangent space

### Manifolds

* $R^n$ -- unconstrained manifold in $R^n$ with Euclidean metric
* [Stiefel](https://en.wikipedia.org/wiki/Stiefel_manifold) -- $A \in \mathbb{R}^{n\times p}\;:\; A^\top A=I$

### Optimizers

* `geoopt.RiemannianSGD` -- a subclass of `torch.optim.SGD` with the same API
