from abc import ABC
from typing import Union, Tuple, Optional
import torch
from ..base import Manifold
from ...linalg import batch_linalg as lalg
from ..siegel import csym_math as sm
from .vvd_metrics import SiegelMetric, SiegelMetricType


class SiegelManifold(Manifold, ABC):
    """
    Manifold to work on Siegel spaces.
    The implementation is aimed to work with realization of the Siegel space as
    spaces of complex symmetric matrices.

    References
    ----------
    - Federico López, Beatrice Pozzetti, Steve Trettel, Michael Strube, Anna Wienhard.
      "Symmetric Spaces for Graph Embeddings: A Finsler-Riemannian Approach", 2021.

    Parameters
    ----------
    metric: str
        one of "riem" (Riemannian), "fone": Finsler One, "finf": Finsler Infinity,
        "fmin": Finsler metric of minimum entropy, "wsum": Weighted sum.
    rank: int
        Rank of the space. Only mandatory for "fmin" and "wsum" metrics.
    """

    __scaling__ = Manifold.__scaling__.copy()
    name = "Siegel Space"
    ndim = 2
    reversible = False

    def __init__(self, metric: str = SiegelMetricType.RIEMANNIAN.value, rank: int = None):
        super().__init__()
        self.metric = SiegelMetric.get(metric, rank)

    def dist(self, z1: torch.Tensor, z2: torch.Tensor, *, keepdim=False) -> torch.Tensor:
        """Compute distance between two points on the manifold according to the specified metric
        Calculates the distance for the Upper Half Space Manifold (UHSM)

        It is implemented here since the way to calculate distances in the Bounded Domain Manifold
        requires mapping the points to the UHSM, and then applying this formula.

        Parameters
        ----------
        z1 : torch.Tensor
             point on the manifold
        z2 : torch.Tensor
             point on the manifold
        keepdim : bool, optional
            keep the last dim?, by default False

        Returns
        -------
        torch.Tensor
            distance between two points
        """
        # with Z1 = X + iY, define Z3 = sqrt(Y)^-1 (Z2 - X) sqrt(Y)^-1
        x, y = z1.real, z1.imag
        inv_sqrt_y = lalg.sym_inv_sqrtm1(y)
        inv_sqrt_y = sm.to_complex(inv_sqrt_y, torch.zeros_like(inv_sqrt_y))
        z2_minus_x = z2 - x
        z3 = inv_sqrt_y @ z2_minus_x @ inv_sqrt_y

        w = sm.inverse_cayley_transform(z3)

        eigvalues = sm.takagi_eigvals(w)      # eigenvalues are in ascending order v1 < v2 < vn

        # assert 1 >= eigvalues >= 0
        eps = sm.EPS[eigvalues.dtype]
        assert torch.all(eigvalues >= 0 - eps), f"Eigenvalues: {eigvalues}"
        assert torch.all(eigvalues <= 1.01), f"Eigenvalues: {eigvalues}"

        # Vector-valued distance: vi = (1 + di) / (1 - di)
        vvd = (1 + eigvalues) / (1 - eigvalues).clamp(min=eps)
        vvd = torch.log(vvd)
        res = self.metric.compute_metric(vvd)
        return res

    def retr(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        # always assume u is scaled properly
        approx = x + u
        return self.projx(approx)

    def _check_shape(self, shape: Tuple[int], name: str) -> Union[Tuple[bool, Optional[str]], bool]:
        reason = None
        if self.dims is not None:
            ok = shape[-1] == self.dims and shape[-2] == self.dims
            if not ok:
                reason = "'{}' on the {} requires more than {} dim".format(name, self, self.dims)
        else:
            ok = shape[-1] == shape[-2]
            if not ok:
                reason = "'{}' on the {} should be a squared matrix".format(name, self)
        return ok, reason

    def _check_matrices_are_symmetric(self, x: torch.Tensor, *, atol: float = 1e-5, rtol: float = 1e-5):
        """
        Parameters
        ----------
        x : torch.Tensor
            point on the manifold
        atol : float
            absolute tolerance for allclose
        rtol : float
            relative tolerance for allclose

        Returns
        -------
        boolean
            whether the points in x are complex symmetric or not
        """
        return sm.is_complex_symmetric(x.unsqueeze(0), atol, rtol)

    def projx(self, x: torch.Tensor) -> torch.Tensor:
        return lalg.sym(x)

    def proju(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        return self.egrad2rgrad(x, u)

    def transp(self, x: torch.Tensor, y: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        return v

    def expmap(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        # We might not need it
        pass

    def logmap(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # We might not need it
        pass

    def _check_vector_on_tangent(
        self, x: torch.Tensor, u: torch.Tensor, *, atol=1e-5, rtol=1e-5
    ) -> Union[Tuple[bool, Optional[str]], bool]:
        pass

    def extra_repr(self) -> str:
        return f"metric={type(self.metric).__name__}"
