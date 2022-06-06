This file tracks important changes in PRs

geoopt (0.5.0)
==============

New Features
------------
* Add data generator for hyperbolic manifold (#192)
* add geoopt.layers.Distance2StereographicHyperplanes

geoopt (0.4.1)
==============

Maintainance
------------
* Upgrade to pytorch 1.10.2 and 1.11.0
* fix deepcopy bug (#189)

geoopt (0.4.0)
==============

New Features
------------
* new Symmetric Positive Definite manifold (#153)
* new Siegel manifolds: Upper half model and Bounded domain model, with support for Riemannian and Finsler metrics (#179)

Maintainance
------------
* create pull request templates (#154)
* update tests for pytorch 1.9.0

Bug Fixes
---------
* fix step increments in optimizers (#165)

geoopt v0.3.0
=============

New Features
------------
* Riemannian Line Search (#140)
* Per group stabilization (#149)

Maintenance
-----------
* Fix API warnings (raised in #148)

geoopt v0.2.0
=============

New Features
------------
* BirkhoffPolytope (#125)
* Lorenz Manifold (#121)
* :math:`kappa`-Stereographic model (#126)
* Sparse optimizers (#130)

Maintenance
-----------
* Tests for pytorch>=1.4, cpuonly (#133)

geoopt v0.1.2
==============

Bug Fixes
---------
* Fix scaling issues with random methods
* Fix poincare methods ``cosub`` and ``norm`` that were working not properly
* Fix Sphere distance for small values


geoopt v0.1.1
==============

New Features
------------
* Add ``geoopt.ismanifold`` utility

Bug Fixes
---------
* Fix typing compatibility with python 3.7+


geoopt v0.1.0
=============

Breaking Changes
----------------
* Better public api, refactored developer api a lot (#40). See the corresponding PR for more details
* Refactored internal design, caused another api change (#77)
* Removed ``t`` argument from everywhere (#76). The argument just scaled tangent vectors but
appeared to be very problematic in maintenance


New Features
------------
* Added ``Sphere`` manifold (#25)
* Added ``SphereSubspaceIntersection``, ``SphereSubspaceComplementIntersection`` manifolds (#29)
* Added expmap implementation (#43)
* Added dist, logmap implementation (#44)
* Added Poincare Ball model (#45)
* Poincare Ball manifold has now new methods (#78)
* Added ``ndim`` argument to ``Euclidean`` manifold
* Added ``Product`` manifold (#109)
* Added ``Scaled`` manifold (#109)
* Unified ``random`` for manifolds (#109) so it can be used in product manifold
* Added ``origin`` for manifolds (#109), it is useful for embeddings

Maintenance
-----------
* Add gitter chat (#31)
* Maintain torch>=1.0.0 only (#39)
* Manifolds are Modules (#49)
* Replace deprecated functions in torch>=1.1.0 (#67)
* Remove PyManOpt from test requirements (#75)
* Add pylint, pydoctest, pydocstyle to travis

Bug Fixes
---------
* Make pickle work with ManifoldTensors (#47)
* Resolve inconsistency with tensor strides and optimizer updates (#71)
