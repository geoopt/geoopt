from functools import partial
from typing import Optional, Tuple, Union
import torch
from .base import Manifold
from ..linalg import batch_linalg

__all__ = ["SymmetricPositiveDefinite"]


EPS = {torch.float32: 1e-4, torch.float64: 1e-7}


class SymmetricPositiveDefinite(Manifold):
    r"""Manifold of symmetric positive definite matrices.

    .. math::

        A = A^T\\
        \langle x, A x \rangle > 0 \quad , \forall x \in \mathrm{R}^{n}, x \neq 0 \\
        A \in \mathrm{R}^{n\times m}


    The tangent space of the manifold contains all symmetric matrices.

    References
    ----------
    - https://github.com/pymanopt/pymanopt/blob/master/pymanopt/manifolds/psd.py
    - https://github.com/dalab/matrix-manifolds/blob/master/graphembed/graphembed/manifolds/spd.py

    Parameters
    ----------
    ndim : int
        number of trailing dimensions treated as matrix dimensions.
        All the operations acting on such as inner products, etc
        will respect the :attr:`ndim`.
    """

    __scaling__ = Manifold.__scaling__.copy()
    name = "SymmetricPositiveDefinite"
    ndim = 0
    reversible = True
    defaulf_metric: str = "AIM"

    def __init__(self, ndim=2, defaulf_metric: str = defaulf_metric):
        super().__init__()
        self.ndim = ndim
        self.defaulf_metric = defaulf_metric

    _dist_doc = """
        Parameters
        ----------
        x : torch.Tensor
            point on the manifold
        y : torch.Tensor
            point on the manifold
        keepdim : bool
            keep the last dim?

        Returns
        -------
        torch.Tensor
            distance between two points
        """

    def _affine_invariant_metric(
        self, x: torch.Tensor, y: torch.Tensor, keepdim=False
    ) -> torch.Tensor:
        r"""Affine Invariant Metric distance.

        {}

        References
        ----------
        A Riemannian framework for tensor computing. 2006.
        """.format(
            self._dist_doc
        )
        inv_sqrt_x = batch_linalg.sym_inv_sqrtm1(x)
        return 0.5 * torch.norm(
            batch_linalg.sym_logm(inv_sqrt_x @ y @ inv_sqrt_x),
            dim=[-1, -2],
            keepdim=keepdim,
        )

    def _stein_metric(
        self, x: torch.Tensor, y: torch.Tensor, keepdim=False
    ) -> torch.Tensor:
        r"""Stein Metric distance.

        {}

        References
        ----------
        A new metric on the manifold of kernel matrices with application to matrix geometric means. 2012.
        """.format(
            self._dist_doc
        )

        def log_det(tensor: torch.Tensor) -> torch.Tensor:
            return torch.log(torch.det(tensor))

        ret = log_det((x + y) * 0.5) - 0.5 * log_det(x @ y)
        if keepdim:
            return torch.unsqueeze(torch.unsqueeze(ret, -1), -1)
        return ret

    def _log_eucliden_metric(
        self, x: torch.Tensor, y: torch.Tensor, keepdim=False
    ) -> torch.Tensor:
        r"""Log-Eucliden Metric distance.

        {}

        References
        ----------
        Log‐Euclidean metrics for fast and simple calculus on diffusion tensors. 2006.
        """.format(
            self._dist_doc
        )
        return torch.norm(
            batch_linalg.sym_logm(x) - batch_linalg.sym_logm(y),
            dim=[-1, -2],
            keepdim=keepdim,
        )

    def _check_point_on_manifold(
        self, x: torch.Tensor, *, atol=1e-5, rtol=1e-5
    ) -> Union[Tuple[bool, Optional[str]], bool]:
        ok = torch.allclose(x, x.transpose(-1, -2), atol=atol, rtol=rtol)
        if not ok:
            return False, "`x != x.transpose` with atol={}, rtol={}".format(atol, rtol)
        e, _ = torch.symeig(x)
        ok = (e > -atol).min()
        if not ok:
            return False, "eigenvalues of x are not all greater than 0."
        return True, None

    def _check_vector_on_tangent(
        self, x: torch.Tensor, u: torch.Tensor, *, atol=1e-5, rtol=1e-5
    ) -> Union[Tuple[bool, Optional[str]], bool]:
        ok = torch.allclose(u, u.transpose(-1, -2), atol=atol, rtol=rtol)
        if not ok:
            return False, "`u != u.transpose` with atol={}, rtol={}".format(atol, rtol)
        return True, None

    def projx(self, x: torch.Tensor) -> torch.Tensor:
        symx = batch_linalg.sym(x)
        return batch_linalg.sym_funcm(symx, partial(torch.clamp, min=EPS[x.dtype]))

    def proju(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        return batch_linalg.sym(u)

    def egrad2rgrad(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        return x @ self.proju(x, u) @ x.transpose(-1, -2)

    _dist_metric = {
        "AIM": _affine_invariant_metric,
        "SM": _stein_metric,
        "LEM": _log_eucliden_metric,
    }

    def dist(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        mode: str = defaulf_metric,
        keepdim=False,
    ) -> torch.Tensor:
        """Compute distance between 2 points on the manifold that is the shortest path along geodesics.

        Parameters
        ----------
        x : torch.Tensor
            point on the manifold
        y : torch.Tensor
            point on the manifold
        mode : str, optional
            choose metric to compute distance, by default defaulf_metric
        keepdim : bool, optional
            keep the last dim?, by default False

        Returns
        -------
        torch.Tensor
            distance between two points

        Raises
        ------
        ValueError
            if `mode` isn't in `_dist_metric`
        """
        if mode in self._dist_metric:
            return self._dist_metric[mode](self, x, y, keepdim=keepdim)
        else:
            raise ValueError(
                "Unsopported metric:'"
                + mode
                + "'. Please choose one from "
                + str(tuple(self._dist_metric.keys()))
            )

    def inner(
        self,
        x: torch.Tensor,
        u: torch.Tensor,
        v: Optional[torch.Tensor] = None,
        keepdim=False,
    ) -> torch.Tensor:
        """
        Inner product for tangent vectors at point :math:`x`.

        Parameters
        ----------
        x : torch.Tensor
            point on the manifold
        u : torch.Tensor
            tangent vector at point :math:`x`
        v : Optional[torch.Tensor]
            tangent vector at point :math:`x`
        keepdim : bool
            keep the last dim?

        Returns
        -------
        torch.Tensor
            inner product (broadcasted)

        Raises
        ------
        ValueError
            if `keepdim` sine `torch.trace` doesn't support keepdim
        """
        if v is None:
            v = u
        inv_x = batch_linalg.sym_invm(x)
        ret = batch_linalg.trace(inv_x @ u @ inv_x @ v)
        if keepdim:
            return torch.unsqueeze(torch.unsqueeze(ret, -1), -1)
        return ret

    def retr(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        inv_x = batch_linalg.sym_invm(x)
        return batch_linalg.sym(x + u + u @ inv_x @ u / 2)

    def expmap(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        inv_sqrt_x, sqrt_x = batch_linalg.sym_sqrtm2(x)
        return sqrt_x @ batch_linalg.sym_expm(inv_sqrt_x @ u @ inv_sqrt_x) @ sqrt_x

    def logmap(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        inv_sqrt_x, sqrt_x = batch_linalg.sym_inv_sqrtm2(x)
        return sqrt_x @ batch_linalg.sym_logm(inv_sqrt_x @ u @ inv_sqrt_x) @ sqrt_x

    def extra_repr(self) -> str:
        return "ndim={}".format(self.ndim)
