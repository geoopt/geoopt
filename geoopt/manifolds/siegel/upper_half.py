from typing import Optional, Tuple, Union
import torch
from geoopt import linalg as lalg
from .siegel import SiegelManifold
from .vvd_metrics import SiegelMetricType
from ..siegel import csym_math as sm

__all__ = ["UpperHalf"]


class UpperHalf(SiegelManifold):
    r"""
    Upper Half Space Manifold.
    Points in the space are complex symmetric matrices.

    .. math::

        \mathcal{S} = \{Z = X + iY \in \operatorname{Sym](n, \mathbb{C}) | Y >> 0 \}.


    Parameters
    ----------
    metric: str
        one of "riem" (Riemannian), "fone": Finsler One, "finf": Finsler Infinity,
        "fmin": Finsler metric of minimum entropy, "wsum": Weighted sum.
    rank: int
        Rank of the space. Only mandatory for "fmin" and "wsum" metrics.
    """

    name = "Upper Half Space"

    def __init__(self, metric: str = SiegelMetricType.RIEMANNIAN.value, rank: int = None):
        super().__init__(metric=metric, rank=rank)

    def egrad2rgrad(self, z: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        r"""
        Transform gradient computed using autodiff to the correct Riemannian gradient for the point :math:`x`.

        For a function :math:`f(Z)` on :math:`\mathcal{S}_n`, the gradient is:
        :math:`\operatorname{grad}(f(Z)) = Y \cdot \operatorname{grad}_E(f(Z)) \cdot Y`,
        where :math:`Y` is the imaginary part of :math:`Z`.

        Parameters
        ----------
        z : torch.Tensor
             point on the manifold
        u : torch.Tensor
             gradient to be projected

        Returns
        -------
        torch.Tensor
            Riemannian gradient
        """
        real_grad, imag_grad = u.real, u.imag
        y = z.imag
        real_grad = y @ real_grad @ y
        imag_grad = y @ imag_grad @ y
        return sm.to_complex(real_grad, imag_grad)

    def projx(self, z: torch.Tensor) -> torch.Tensor:
        """
        Project point :math:`z` on the manifold.

        In this space, we need to ensure that :math:`Y = Im(Z)` is positive definite.
        Since the matrix Y is symmetric, it is possible to diagonalize it.
        For a diagonal matrix the condition is just that all diagonal entries are positive,
        so we clamp the values that are <= 0 in the diagonal to an epsilon, and then restore
        the matrix back into non-diagonal form using the base change matrix that was obtained
        from the diagonalization.

        Parameters
        ----------
        z : torch.Tensor
             point on the manifold

        Returns
        -------
        torch.Tensor
            Projected points
        """
        z = super().projx(z)

        y = z.imag
        y_tilde = sm.positive_conjugate_projection(y)

        return sm.to_complex(z.real, y_tilde)

    def inner(self, z: torch.Tensor, u: torch.Tensor, v=None, *, keepdim=False) -> torch.Tensor:
        r"""
        Inner product for tangent vectors at point :math:`z`.
        The inner product at point :math:`Z = X + iY` of the vectors :math:`U, V` is:

        .. math::

            g_{Z}(U, V) = \operatorname{tr}[ Y^-1 U Y^-1 \overline{V} ]

        Parameters
        ----------
        z : torch.Tensor
             point on the manifold
        u : torch.Tensor
             tangent vector at point :math:`z`
        v : torch.Tensor
             tangent vector at point :math:`z`
        keepdim : bool
            keep the last dim?

        Returns
        -------
        torch.Tensor
            inner product (broadcasted)
        """
        if v is None:
            v = u
        inv_y = sm.inverse(z.imag).type_as(z)

        res = inv_y @ u @ inv_y @ v.conj()
        res = lalg.trace(res)
        if keepdim:
            return torch.unsqueeze(res, -1)
        return res

    def _check_point_on_manifold(self, z: torch.Tensor, *, atol=1e-5, rtol=1e-5):
        if not self._check_matrices_are_symmetric(z, atol=atol, rtol=rtol):
            return False, "Matrices are not symmetric"

        # Im(Z) should be positive definite.
        ok = torch.all(torch.det(z.imag) > 0)
        if not ok:
            reason = "Imaginary part of Z is not positive definite"
        else:
            reason = None
        return ok, reason

    def random(self, *size, dtype=None, device=None, **kwargs) -> torch.Tensor:
        if dtype and dtype not in {torch.complex32, torch.complex64, torch.complex128}:
            raise ValueError("dtype must be one of {torch.complex32, torch.complex64, torch.complex128}")
        if dtype is None:
            dtype = torch.complex128
        tens = 0.5 * torch.randn(*size, dtype=dtype, device=device)
        tens = lalg.sym(tens)
        tens.imag = lalg.expm(tens.imag)
        return tens

    def origin(
        self,
        *size: Union[int, Tuple[int]],
        dtype=None,
        device=None,
        seed: Optional[int] = 42
    ) -> torch.Tensor:
        imag = torch.eye(*size[:-1], dtype=dtype, device=device)
        if imag.dtype in {torch.complex32, torch.complex64, torch.complex128}:
            imag = imag.real
        return torch.complex(torch.zeros_like(imag), imag)

