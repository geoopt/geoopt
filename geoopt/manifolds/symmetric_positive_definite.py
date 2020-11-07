from functools import partial
from typing import Callable, Literal, Optional, Tuple, Union, get_args
import torch
from .base import Manifold

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
    _metric_literal = Literal["AIM", "SM", "LEM"]
    defaulf_metric: _metric_literal = "AIM"

    def __init__(self, ndim=2, defaulf_metric: _metric_literal = defaulf_metric):
        super().__init__()
        self.ndim = ndim
        self.defaulf_metric = defaulf_metric

    def _funcm(
        self, x: torch.Tensor, func: Callable[[torch.Tensor], torch.Tensor]
    ) -> torch.Tensor:
        """Apply function to symmetric matrix.

        Parameters
        ----------
        x : torch.Tensor
            symmetric matrix
        func : Callable[[torch.Tensor], torch.Tensor]
            function to apply

        Returns
        -------
        torch.Tensor
            symmetric matrix with function applied to
        """
        e, v = torch.symeig(x, eigenvectors=True)
        return v @ torch.diag_embed(func(e)) @ v.transpose(-1, -2)

    def _expm(self, x: torch.Tensor, using_native=False) -> torch.Tensor:
        """Symmetric matrix exponent.

        Parameters
        ----------
        x : torch.Tensor
            symmetric matrix
        using_native : bool, optional
            if using native matrix exponent `torch.matrix_exp`, by default False

        Returns
        -------
        torch.Tensor
            exp(x)
        """
        if using_native:
            return torch.matrix_exp(x)
        else:
            return self._funcm(x, torch.exp)

    def _logm(self, x: torch.Tensor) -> torch.Tensor:
        """Symmetric matrix logarithm.

        Parameters
        ----------
        x : torch.Tensor
            symmetric matrix

        Returns
        -------
        torch.Tensor
            log(x)
        """
        return self._funcm(x, torch.log)

    def _sqrtm(self, x: torch.Tensor) -> torch.Tensor:
        """Symmetric matrix square root .

        Parameters
        ----------
        x : torch.Tensor
            symmetric matrix

        Returns
        -------
        torch.Tensor
            sqrt(x)
        """
        return self._funcm(x, torch.sqrt)

    def _invm(self, x: torch.Tensor) -> torch.Tensor:
        """Symmetric matrix inverse.

        Parameters
        ----------
        x : torch.Tensor
            symmetric matrix

        Returns
        -------
        torch.Tensor
            x^{-1}
        """
        return self._funcm(x, torch.reciprocal)

    def _inv_sqrtm1(self, x: torch.Tensor) -> torch.Tensor:
        """Symmetric matrix inverse square root.

        Parameters
        ----------
        x : torch.Tensor
            symmetric matrix

        Returns
        -------
        torch.Tensor
            x^{-1/2}
        """
        return self._funcm(x, lambda tensor: torch.reciprocal(torch.sqrt(tensor)))

    def _inv_sqrtm2(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Symmetric matrix inverse square root, with square root return also.

        Parameters
        ----------
        x : torch.Tensor
            symmetric matrix

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            x^{-1/2}, sqrt(x)
        """
        e, v = torch.symeig(x, eigenvectors=True)
        sqrt_e = torch.sqrt(e)
        inv_sqrt_e = 1 / sqrt_e
        return (
            v @ torch.diag_embed(inv_sqrt_e) @ v.transpose(-1, -2),
            v @ torch.diag_embed(sqrt_e) @ v.transpose(-1, -2),
        )

    def _sym(self, x: torch.Tensor) -> torch.Tensor:
        r"""Make matrix symmetric.

        .. math::

            \frac{A + A^T}{2}

        Parameters
        ----------
        x : torch.Tensor
            matrix to symmetrize

        Returns
        -------
        torch.Tensor
            symmetric matrix
        """
        return (x + x.transpose(-1, -2)) / 2

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
        inv_sqrt_x = self._inv_sqrtm1(x)
        return 0.5 * torch.norm(
            self._logm(inv_sqrt_x @ y @ inv_sqrt_x), dim=[-1, -2], keepdim=keepdim
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

        if keepdim:
            raise ValueError("`torch.det` doesn't support keepdim.")
        return log_det((x + y) * 0.5) - 0.5 * log_det(x @ y)

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
        return torch.norm(self._logm(x) - self._logm(y), dim=[-1, -2], keepdim=keepdim)

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
        symx = self._sym(x)
        return self._funcm(symx, partial(torch.clamp, min=EPS[x.dtype]))

    def proju(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        return self._sym(u)

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
        mode: _metric_literal = defaulf_metric,
        keepdim=False,
    ) -> torch.Tensor:
        """Compute distance between 2 points on the manifold that is the shortest path along geodesics.

        Parameters
        ----------
        x : torch.Tensor
            point on the manifold
        y : torch.Tensor
            point on the manifold
        mode : _metric_literal, optional
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
            if `mode` isn't in `_metric_literal`
        """
        if mode in self._dist_metric:
            return self._dist_metric[mode](self, x, y, keepdim=keepdim)
        else:
            raise ValueError(
                "Unsopported metric:'"
                + mode
                + "'. Please choose one from "
                + str(get_args(self._metric_literal))
            )

    _inner_metric = {
        "AIM": _affine_invariant_inner,
    }

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
        if keepdim:
            raise ValueError("`torch.trace` doesn't support keepdim.")
        inv_x = self._invm(x)
        return torch.trace(inv_x @ u @ inv_x @ v)
        

    def retr(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        inv_x = self._invm(x)
        return self._sym(x + u + u @ inv_x @ u / 2)

    def expmap(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        inv_sqrt_x, sqrt_x = self._inv_sqrtm2(x)
        return sqrt_x @ self._expm(inv_sqrt_x @ u @ inv_sqrt_x) @ sqrt_x

    def logmap(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        inv_sqrt_x, sqrt_x = self._inv_sqrtm2(x)
        return sqrt_x @ self._logm(inv_sqrt_x @ u @ inv_sqrt_x) @ sqrt_x

    def extra_repr(self) -> str:
        return "ndim={}".format(self.ndim)
