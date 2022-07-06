geoopt
======

|Python Package Index| |Read The Docs| |Build Status| |Coverage Status| |Codestyle Black| |Gitter|

Manifold aware ``pytorch.optim``.

Unofficial implementation for `“Riemannian Adaptive Optimization
Methods”`_ ICLR2019 and more.

Installation
------------
Make sure you have pytorch>=1.11.0 installed

There are two ways to install geoopt:

1. GitHub (preferred so far) due to active development

.. code-block:: bash

    pip install git+https://github.com/geoopt/geoopt.git


2. pypi (this might be significantly behind master branch but kept as fresh as possible)

.. code-block:: bash

    pip install geoopt

The preferred way to install geoopt will change once stable project stage is achieved.
Now, pypi is behind master as we actively develop and implement new features.


PyTorch Support
~~~~~~~~~~~~~~~
Geoopt officially supports **2 latest stable versions** of pytorch upstream or the latest major release.

What is done so far
-------------------

Work is in progress but you can already use this. Note that API might
change in future releases.

Tensors
~~~~~~~

-  ``geoopt.ManifoldTensor`` - just as torch.Tensor with additional
   ``manifold`` keyword argument.
-  ``geoopt.ManifoldParameter`` - same as above, recognized in
   ``torch.nn.Module.parameters`` as correctly subclassed.

All above containers have special methods to work with them as with
points on a certain manifold

-  ``.proj_()`` - inplace projection on the manifold.
-  ``.proju(u)`` - project vector ``u`` on the tangent space. You need
   to project all vectors for all methods below.
-  ``.egrad2rgrad(u)`` - project gradient ``u`` on Riemannian manifold
-  ``.inner(u, v=None)`` - inner product at this point for two
   **tangent** vectors at this point. The passed vectors are not
   projected, they are assumed to be already projected.
-  ``.retr(u)`` - retraction map following vector ``u``
-  ``.expmap(u)`` - exponential map following vector ``u`` (if expmap is not available in closed form, best approximation is used)
-  ``.transp(v, u)`` - transport vector ``v``  with direction ``u``
-  ``.retr_transp(v, u)`` - transport ``self``, vector ``v``
   (and possibly more vectors) with direction ``u``
   (returns are plain tensors)

Manifolds
~~~~~~~~~

-  ``geoopt.Euclidean`` - unconstrained manifold in ``R`` with
   Euclidean metric
-  ``geoopt.Stiefel`` - Stiefel manifold on matrices
   ``A in R^{n x p} : A^t A=I``, ``n >= p``
-  ``geoopt.Sphere`` - Sphere manifold ``||x||=1``
-  ``geoopt.BirkhoffPolytope`` - manifold of Doubly Stochastic matrices
-  ``geoopt.Stereographic`` - Constant curvature stereographic projection model
-  ``geoopt.SphereProjection`` - Sphere stereographic projection model
-  ``geoopt.PoincareBall`` -  `Poincare ball model <https://en.wikipedia.org/wiki/Poincar%C3%A9_disk_model>`_
-  ``geoopt.Lorentz`` - `Hyperboloid model <https://en.wikipedia.org/wiki/Hyperboloid_model>`_
-  ``geoopt.ProductManifold`` - Product manifold constructor
-  ``geoopt.Scaled`` - Scaled version of the manifold. Similar to `Learning Mixed-Curvature Representations in Product Spaces <https://openreview.net/forum?id=HJxeWnCcF7>`_ if combined with ``ProductManifold``
-  ``geoopt.SymmetricPositiveDefinite`` - SPD matrix manifold
-  ``geoopt.UpperHalf`` - Siegel Upper half manifold. Supports Riemannian and Finsler metrics, as in `Symmetric Spaces for Graph Embeddings: A Finsler-Riemannian Approach <https://arxiv.org/abs/2106.04941>`_.
-  ``geoopt.BoundedDomain`` - Siegel Bounded domain manifold. Supports Riemannian and Finsler metrics.

All manifolds implement methods necessary to manipulate tensors on manifolds and
tangent vectors to be used in general purpose. See more in `documentation`_.

Optimizers
~~~~~~~~~~

-  ``geoopt.optim.RiemannianSGD`` - a subclass of ``torch.optim.SGD``
   with the same API
-  ``geoopt.optim.RiemannianAdam`` - a subclass of ``torch.optim.Adam``

Samplers
~~~~~~~~

-  ``geoopt.samplers.RSGLD`` - Riemannian Stochastic Gradient Langevin
   Dynamics
-  ``geoopt.samplers.RHMC`` - Riemannian Hamiltonian Monte-Carlo
-  ``geoopt.samplers.SGRHMC`` - Stochastic Gradient Riemannian
   Hamiltonian Monte-Carlo


Layers
~~~~~~
Experimental ``geoopt.layers`` module allows to embed geoopt into deep learning

Citing Geoopt
~~~~~~~~~~~~~
If you find this project useful in your research, please kindly add this bibtex entry in references and cite.

.. code:: bibtex

    @misc{geoopt2020kochurov,
        title={Geoopt: Riemannian Optimization in PyTorch},
        author={Max Kochurov and Rasul Karimov and Serge Kozlukov},
        year={2020},
        eprint={2005.02819},
        archivePrefix={arXiv},
        primaryClass={cs.CG}
    }

Donations
~~~~~~~~~
ETH: 0x008319973D4017414FdF5B3beF1369bA78275C6A
   

.. _“Riemannian Adaptive Optimization Methods”: https://openreview.net/forum?id=r1eiqi09K7
.. _documentation: https://geoopt.readthedocs.io/en/latest/manifolds.html


.. |Python Package Index| image:: https://img.shields.io/pypi/v/geoopt.svg
   :target: https://pypi.python.org/pypi/geoopt
.. |Read The Docs| image:: https://readthedocs.org/projects/geoopt/badge/?version=latest
   :target: https://geoopt.readthedocs.io/en/latest/?badge=latest
   :alt: Documentation Status
.. |Build Status| image:: https://github.com/geoopt/geoopt/actions/workflows/testing.yml/badge.svg?branch=master
   :target: https://github.com/geoopt/geoopt/actions/workflows/testing.yml
.. |Coverage Status| image:: https://codecov.io/gh/geoopt/geoopt/branch/master/graph/badge.svg?token=HOI5LD0VWF
   :target: https://codecov.io/gh/geoopt/geoopt
.. |Codestyle Black| image:: https://img.shields.io/badge/code%20style-black-000000.svg
   :target: https://github.com/ambv/black
.. |Gitter| image:: https://badges.gitter.im/geoopt/community.png
   :target: https://gitter.im/geoopt/community
   
   
