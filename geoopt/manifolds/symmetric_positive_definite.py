from functools import partial
from typing import Callable, Literal, Optional, Tuple, Union, get_args
import torch
from .base import Manifold

__all__ = ["SymmetricPositiveDefinite"]


EPS = {torch.float32: 1e-4, torch.float64: 1e-7}


class SymmetricPositiveDefinite(Manifold):
    r"""
    Manifold of symmetric positive definite matrices.

    .. math::

        A = A^T\\
        \langle x, A x \rangle > 0 \quad , \forall x \in \mathrm{R}^{n}, x \neq 0 \\
        A \in \mathrm{R}^{n\times m}


    The tangent space of the manifold contains all symmetric matrices.
    
    Reference implementations:
    - https://github.com/pymanopt/pymanopt/blob/master/pymanopt/manifolds/psd.py
    - https://github.com/dalab/matrix-manifolds/blob/master/graphembed/graphembed/manifolds/spd.py

    Parameters
    ----------
    ndim : int
        number of trailing dimensions treated as matrix dimensions. All the operations acting on cuch
        as inner products, etc will respect the :attr:`ndim`.
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

    def _t(self, x: torch.Tensor) -> torch.Tensor:
        return x.transpose(-1, -2)

    def _funcm(
        self, x: torch.Tensor, func: Callable[[torch.Tensor], torch.Tensor]
    ) -> torch.Tensor:
        e, v = torch.symeig(x, eigenvectors=True)
        return v @ torch.diag_embed(func(e)) @ v.transpose(-1, -2)

    def _expm(self, x: torch.Tensor, using_native=False) -> torch.Tensor:
        if using_native:
            return torch.matrix_exp(x)
        else:
            return self._funcm(x, torch.exp)

    def _logm(self, x: torch.Tensor) -> torch.Tensor:
        return self._funcm(x, torch.log)

    def _sqrtm(self, x: torch.Tensor) -> torch.Tensor:
        return self._funcm(x, torch.sqrt)

    def _invm(self, x: torch.Tensor) -> torch.Tensor:
        return self._funcm(x, torch.reciprocal)

    def _inv_sqrtm1(self, x: torch.Tensor) -> torch.Tensor:
        return self._funcm(x, lambda tensor: torch.reciprocal(torch.sqrt(tensor)))

    def _inv_sqrtm2(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        e, v = torch.symeig(x, eigenvectors=True)
        sqrt_e = torch.sqrt(e)
        inv_sqrt_e = 1 / sqrt_e
        return (
            v @ torch.diag_embed(inv_sqrt_e) @ v.transpose(-1, -2),
            v @ torch.diag_embed(sqrt_e) @ v.transpose(-1, -2),
        )

    def _affine_invariant_metric(
        self, x: torch.Tensor, y: torch.Tensor, keepdim=False
    ) -> torch.Tensor:
        # Affine Invariant Metric
        # Ref:
        # Pennec, Xavier, Pierre Fillard, and Nicholas Ayache. "A Riemannian framework for tensor computing." International Journal of computer vision 66.1 (2006): 41-66.
        inv_sqrt_x = self._inv_sqrtm1(x)
        return 0.5 * torch.norm(
            self._logm(inv_sqrt_x @ y @ inv_sqrt_x), dim=[-1, -2], keepdim=keepdim
        )

    def _affine_invariant_inner(
        self,
        x: torch.Tensor,
        u: torch.Tensor,
        v: torch.Tensor = None,
        keepdim=False,
    ) -> torch.Tensor:
        assert not keepdim, "`torch.trace` doesn't support keepdim."
        inv_x = self._invm(x)
        return torch.trace(inv_x @ u @ inv_x @ v)

    def _stein_metric(
        self, x: torch.Tensor, y: torch.Tensor, keepdim=False
    ) -> torch.Tensor:
        # Stein Metric
        # Ref:
        # Sra, Suvrit. "A new metric on the manifold of kernel matrices with application to matrix geometric means." Advances in neural information processing systems. 2012.
        assert not keepdim, "`torch.det` doesn't support keepdim."
        log_det = lambda x: torch.log(torch.det(x))
        return log_det((x + y) * 0.5) - 0.5 * log_det(x @ y)

    def _log_eucliden_metric(
        self, x: torch.Tensor, y: torch.Tensor, keepdim=False
    ) -> torch.Tensor:
        # Log-Eucliden Metric
        # Ref:
        # Arsigny, Vincent, et al. "Log‐Euclidean metrics for fast and simple calculus on diffusion tensors." Magnetic Resonance in Medicine: An Official Journal of the International Society for Magnetic Resonance in Medicine 56.2 (2006): 411-421.
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
        symx = (x + x.transpose(-1, -2)) / 2
        return self._funcm(symx, partial(torch.clamp, min=EPS[x.dtype]))

    def proju(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        return (u + u.transpose(-1, -2)) / 2

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
        mode: _metric_literal = defaulf_metric,
        keepdim=False,
    ) -> torch.Tensor:
        if v is None:
            v = u
        if mode in self._dist_metric:
            return self._inner_metric[mode](self, x, u, v, keepdim=keepdim)
        else:
            raise ValueError(
                "Unsopported metric:'"
                + mode
                + "'. Please choose one from "
                + str(get_args(self._metric_literal))
            )

    def retr(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def expmap(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        inv_sqrt_x, sqrt_x = self._inv_sqrtm2(x)
        return sqrt_x @ self._expm(inv_sqrt_x @ u @ inv_sqrt_x) @ sqrt_x

    def logmap(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        inv_sqrt_x, sqrt_x = self._inv_sqrtm2(x)
        return sqrt_x @ self._logm(inv_sqrt_x @ u @ inv_sqrt_x) @ sqrt_x

    def extra_repr(self):
        return "ndim={}".format(self.ndim)
