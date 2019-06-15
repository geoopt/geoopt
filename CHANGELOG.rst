This file tracks important changes in PRs

geoopt v0.1.0 (unreleased)
==========================

Breaking Changes
----------------
* better public api, refactored developer api a lot (#40). See the corresponding PR for more details

New Features
------------
* Added ``Sphere`` manifold (#25)
* Added ``SphereSubspaceIntersection``, ``SphereSubspaceComplementIntersection`` manifolds (#29)
* Added expmap implementation (#43)
* Added dist, logmap implementation (#44)
* Added Poincare Ball model (#45)

Maintenance
-----------
* Add gitter chat (#31)
* Maintain torch>=1.0.0 only (#39)
* Manifolds are Modules (#49)
* Replace deprecated functions in torch>=1.1.0 (#67)
* Remove PyManOpt from test requirements (#75)

Deprecations
------------

Bug Fixes
---------
* Make pickle work with ManifoldTensors (#47)
* Resolve inconsistency with tensor strides and optimizer updates (#71)
