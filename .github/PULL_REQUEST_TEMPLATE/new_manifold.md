# Adding <NewManifold>

## Description

* Reference math in the paper (link)
* Reference implementation (some github repo maybe)

## Check list

- [ ] New manifold is self-contained in `geoopt.manifolds.<new_manifold>` file or package. It should have inplemented all the methods that are implemented in Sphere case (as an example implementation).
    - [ ] `_check_shape` -- to check shape consistency for the created tensors
    - [ ] `_check_point_on_manifold` -- to check validity of the created tensors
    - [ ] `_check_vector_on_tangent` -- to check validity of the tangent vectors
    - [ ] `projx` -- projecting a point from ambient space to the manifold 
    - [ ] `proju` -- projecting a tangent vector to the ambient space
    - [ ] `random` -- sampling some random point on the manifold
    - [ ] `origin` -- creating a reference point on the manifold
    - [ ] `inner` product between tangent vectors
    - [ ] `retr` -- retraction (approx. `expmap` for the manifold)
    - [ ] `expmap` -- exact expmap (if possible or fall back to `retr`)
    - [ ] `transp` -- vector transport for tangent vectors
    - [ ] `logmap` (if possible)
    - [ ] `dist` (if possible)
    - [ ] `egrad2rgrad` -- convert `tensor.grad` to a riemannian gradient (could be just projection)
    - [ ] `class NewManifoldExact(NewManifold)` if you have expmap implemented differently that retr.

- [ ] Imports into the package namespace are consistent with other manifolds, see [this file](https://github.com/geoopt/geoopt/blob/master/geoopt/manifolds/__init__.py). Rule of thumb is to import a class unless you have a reason to import a package.
- [ ] There is a test case in [tests/test_manifold_basic](https://github.com/geoopt/geoopt/blob/master/tests/test_manifold_basic.py). It will require to create a **shape_case** (`manifold_shapes` variable). And a test case `UnaryCase` tuple containing. You can see how it is done for `canonical_stiefel_case()` as an example (it is a generator yielding `UnaryCases`).
    1. initial point
    2. projection out of this point
    3. same for the tangent vector
    
    Add this test case to `unary_case_base` in that file.

- [ ] Create a simple convergence test to fogure out if there are any potential numerical issues in implementation as done in [stiefel case](https://github.com/geoopt/geoopt/blob/master/tests/test_rsgd.py)

