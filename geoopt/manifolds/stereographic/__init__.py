import torch.nn
from typing import Tuple, Optional
from . import math
import geoopt
from ...utils import size2shape, broadcast_shapes
from ..base import Manifold, ScalingInfo

__all__ = ["Stereographic", "StereographicExact"]

_stereographic_doc = r"""
    Stereographic manifold model, see more in :doc:`/extended/stereographic`
    
    Parameters
    ----------
    K : float|tensor 
        sectional curvature of manifold
        - K<0: Poincaré ball
        - K>0: Stereographic projection of sphere
        - c-->0: approaches Euclidean geometry if points stay relatively close 
          to center
    
    Notes
    -----
    It is extremely recommended to work with this manifold in double precision.
    Do not use this manifold with c=0 to get Euclidean geometry!
"""


# noinspection PyMethodOverriding
class Stereographic(Manifold):
    __doc__ = r"""{}

    See Also
    --------
    :class:`StereographicExact`
    """.format(
        _stereographic_doc
    )

    ndim = 1
    reversible = False
    name = "Stereographic"
    __scaling__ = Manifold.__scaling__.copy()

    def __init__(self,
                 K=1.0,
                 float_precision=torch.float64,
                 keep_sign_fixed=False,
                 min_abs_K=0.001):

        # initialize Manifold superclass
        super().__init__()

        # keep track of floating point precision
        self.float_precision = float_precision

        # keep track of sign fixing
        self.keep_sign_fixed = keep_sign_fixed

        # define minimal curvature
        self.min_abs_K = torch.tensor(min_abs_K, dtype=self.float_precision)

        # convert supplied K to tensor
        K = torch.tensor(K, dtype=self.float_precision)

        # keep track of sign of provided K (enforce no gradients)
        self.initial_sign = torch.nn.Parameter(torch.sign(K),
                                               requires_grad=False)

        # define initial curvature from the provided K
        K_init = torch.zeros(1, dtype=self.float_precision)
        if self.keep_sign_fixed:
            # if we keep the sign fixed it will always be
            # sign * (min abs curv + positive value of K(via softplus))
            K_init[0] = self._inverse_softplus(K.abs() - self.min_abs_K)
        else:
            # if we don't keep the sign fixed it will be
            # K = sign * (min abs curv) + K
            K_init[0] = K - self.initial_sign * self.min_abs_K
        self.trainable_K = torch.nn.Parameter(K_init, requires_grad=False)

    def _softplus(self, x) -> torch.Tensor:
        return torch.log(1.0 + torch.exp(x))

    def _inverse_softplus(self, y) -> torch.Tensor:
        x = torch.log(torch.exp(y)-1.0)
        return x

    def get_K(self) -> torch.Tensor:
        """
        Returns the manifold's sectional curvature according to the specified
        rules (keep sign fixed or not).

        The radius R is derived from get_K().

        Softplus (and its inverse) are used to get better-behaved gradients
        for the trainable part of the curvature K (self.trainable_K).

        Note that get_K() != self.trainable_K !!! Use get_K() if you want to get
        the actual sectional curvature of the manifold.

        :return: The manifold's sectional curvature K
        """
        if self.keep_sign_fixed:
            return -self.initial_sign * \
                   (self.min_abs_K + self._softplus(self.trainable_K))
        else:
            # take sign twice to avoid == 0
            return -torch.sign(torch.sign(self.trainable_K)+1e-15) \
                   * self.min_abs_K + self.trainable_K

    def get_R(self) -> torch.Tensor:
        """
        Gets the manifold's radius R.
        :return:
        """
        return 1.0 / torch.sqrt(torch.abs(self.get_K()))

    def _check_point_on_manifold(
        self, x: torch.Tensor, *, atol=1e-5, rtol=1e-5
    ) -> Tuple[bool, Optional[str]]:
        if self.get_K() < 0:
            px = math.project(x, K=self.get_K())
            ok = torch.allclose(x, px, atol=atol, rtol=rtol)
            if not ok:
                reason = \
                    "'x' norm lies out of the bounds " + \
                    "[-1/sqrt(abs(K))+eps, 1/sqrt(abs(K))-eps]"
            else:
                reason = None
            return ok, reason
        else:
            return True, None

    def _check_vector_on_tangent(
        self, x: torch.Tensor, u: torch.Tensor, *, atol=1e-5, rtol=1e-5
    ) -> Tuple[bool, Optional[str]]:
        return True, None

    def dist(
        self, x: torch.Tensor, y: torch.Tensor, *, keepdim=False, dim=-1
    ) -> torch.Tensor:
        return math.dist(x, y, K=self.get_K(), keepdim=keepdim, dim=dim)

    def egrad2rgrad(
        self, x: torch.Tensor, u: torch.Tensor, *, dim=-1
    ) -> torch.Tensor:
        return math.egrad2rgrad(x, u, K=self.get_K(), dim=dim)

    def retr(self, x: torch.Tensor, u: torch.Tensor, *, dim=-1) -> torch.Tensor:
        # always assume u is scaled properly
        approx = x + u
        return math.project(approx, K=self.get_K(), dim=dim)

    def projx(self, x: torch.Tensor, dim=-1) -> torch.Tensor:
        return math.project(x, K=self.get_K(), dim=dim)

    def proju(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        target_shape = broadcast_shapes(x.shape, u.shape)
        return u.expand(target_shape)

    def inner(
        self,
        x: torch.Tensor,
        u: torch.Tensor,
        v: torch.Tensor = None,
        *,
        keepdim=False,
        dim=-1
    ) -> torch.Tensor:
        if v is None:
            v = u
        return math.inner(x, u, v, K=self.get_K(), keepdim=keepdim, dim=dim)

    def norm(
        self, x: torch.Tensor, u: torch.Tensor, *, keepdim=False, dim=-1
    ) -> torch.Tensor:
        return math.norm(x, u, K=self.get_K(), keepdim=keepdim, dim=dim)

    def expmap(
        self, x: torch.Tensor, u: torch.Tensor, *, project=True, dim=-1
    ) -> torch.Tensor:
        res = math.expmap(x, u, K=self.get_K(), dim=dim)
        if project:
            return math.project(res, K=self.get_K(), dim=dim)
        else:
            return res

    def logmap(
        self, x: torch.Tensor, y: torch.Tensor, *, dim=-1
    ) -> torch.Tensor:
        return math.logmap(x, y, K=self.get_K(), dim=dim)

    def transp(self, x: torch.Tensor, y: torch.Tensor, v: torch.Tensor, dim=-1):
        return math.parallel_transport(x, y, v, K=self.get_K(), dim=dim)


    def transp_follow_retr(
        self, x: torch.Tensor, u: torch.Tensor, v: torch.Tensor, dim=-1
    ) -> torch.Tensor:
        y = self.retr(x, u, dim=dim)
        return self.transp(x, y, v, dim=dim)

    def transp_follow_expmap(
        self, x: torch.Tensor, u: torch.Tensor, v: torch.Tensor, dim=-1, project=True
    ) -> torch.Tensor:
        y = self.expmap(x, u, dim=dim, project=project)
        return self.transp(x, y, v, dim=dim)

    def expmap_transp(
        self, x: torch.Tensor, u: torch.Tensor, v: torch.Tensor, dim=-1, project=True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        y = self.expmap(x, u, dim=dim, project=project)
        v_transp = self.transp(x, y, v, dim=dim)
        return y, v_transp

    def retr_transp(
        self, x: torch.Tensor, u: torch.Tensor, v: torch.Tensor, dim=-1
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        y = self.retr(x, u, dim=dim)
        v_transp = self.transp(x, y, v, dim=dim)
        return y, v_transp

    def mobius_add(
        self, x: torch.Tensor, y: torch.Tensor, *, dim=-1, project=True
    ) -> torch.Tensor:
        res = math.mobius_add(x, y, K=self.get_K(), dim=dim)
        if project:
            return math.project(res, K=self.get_K(), dim=dim)
        else:
            return res

    def mobius_sub(
        self, x: torch.Tensor, y: torch.Tensor, *, dim=-1, project=True
    ) -> torch.Tensor:
        res = math.mobius_sub(x, y, K=self.get_K(), dim=dim)
        if project:
            return math.project(res, K=self.get_K(), dim=dim)
        else:
            return res

    def mobius_coadd(
        self, x: torch.Tensor, y: torch.Tensor, *, dim=-1, project=True
    ) -> torch.Tensor:
        res = math.mobius_coadd(x, y, K=self.get_K(), dim=dim)
        if project:
            return math.project(res, K=self.get_K(), dim=dim)
        else:
            return res

    def mobius_cosub(
        self, x: torch.Tensor, y: torch.Tensor, *, dim=-1, project=True
    ) -> torch.Tensor:
        res = math.mobius_coadd(x, y, K=self.get_K(), dim=dim)
        if project:
            return math.project(res, K=self.get_K(), dim=dim)
        else:
            return res

    def mobius_scalar_mul(
        self, r: torch.Tensor, x: torch.Tensor, *, dim=-1, project=True
    ) -> torch.Tensor:
        res = math.mobius_scalar_mul(r, x, K=self.get_K(), dim=dim)
        if project:
            return math.project(res, K=self.get_K(), dim=dim)
        else:
            return res

    def mobius_pointwise_mul(
        self, w: torch.Tensor, x: torch.Tensor, *, dim=-1, project=True
    ) -> torch.Tensor:
        res = math.mobius_pointwise_mul(w, x, K=self.get_K(), dim=dim)
        if project:
            return math.project(res, K=self.get_K(), dim=dim)
        else:
            return res

    def mobius_matvec(
        self, m: torch.Tensor, x: torch.Tensor, *, dim=-1, project=True
    ) -> torch.Tensor:
        res = math.mobius_matvec(m, x, K=self.get_K(), dim=dim)
        if project:
            return math.project(res, K=self.get_K(), dim=dim)
        else:
            return res

    def geodesic(
        self, t: torch.Tensor, x: torch.Tensor, y: torch.Tensor, *, dim=-1
    ) -> torch.Tensor:
        return math.geodesic(t, x, y, K=self.get_K(), dim=dim)

    @__scaling__(ScalingInfo(t=-1))
    def geodesic_unit(
        self, t: torch.Tensor, x: torch.Tensor, u: torch.Tensor, *, dim=-1,
        project=True
    ) -> torch.Tensor:
        res = math.geodesic_unit(t, x, u, K=self.get_K(), dim=dim)
        if project:
            return math.project(res, K=self.get_K(), dim=dim)
        else:
            return res

    def lambda_x(
        self, x: torch.Tensor, *, dim=-1, keepdim=False
    ) -> torch.Tensor:
        return math.lambda_x(x, K=self.get_K(), dim=dim, keepdim=keepdim)

    @__scaling__(ScalingInfo(1))
    def dist0(self, x: torch.Tensor, *, dim=-1, keepdim=False) -> torch.Tensor:
        return math.dist0(x, K=self.get_K(), dim=dim, keepdim=keepdim)

    @__scaling__(ScalingInfo(u=-1))
    def expmap0(self, u: torch.Tensor, *, dim=-1, project=True) -> torch.Tensor:
        res = math.expmap0(u, K=self.get_K(), dim=dim)
        if project:
            return math.project(res, K=self.get_K(), dim=dim)
        else:
            return res

    @__scaling__(ScalingInfo(1))
    def logmap0(self, x: torch.Tensor, *, dim=-1) -> torch.Tensor:
        return math.logmap0(x, K=self.get_K(), dim=dim)

    def transp0(
        self, y: torch.Tensor, u: torch.Tensor, *, dim=-1
    ) -> torch.Tensor:
        return math.parallel_transport0(y, u, K=self.get_K(), dim=dim)

    def transp0back(
        self, y: torch.Tensor, u: torch.Tensor, *, dim=-1
    ) -> torch.Tensor:
        return math.parallel_transport0back(y, u, K=self.get_K(), dim=dim)

    def gyration(
        self, x: torch.Tensor, y: torch.Tensor, z: torch.Tensor, *, dim=-1
    ) -> torch.Tensor:
        return math.gyration(x, y, z, K=self.get_K(), dim=dim)

    @__scaling__(ScalingInfo(1))
    def dist2plane(
        self,
        x: torch.Tensor,
        p: torch.Tensor,
        a: torch.Tensor,
        *,
        dim=-1,
        keepdim=False,
        signed=False
    ) -> torch.Tensor:
        return math.dist2plane(
            x, p, a, dim=dim, K=self.get_K(), keepdim=keepdim, signed=signed
        )

    # this does not yet work with scaling
    @__scaling__(ScalingInfo.NotCompatible)
    def mobius_fn_apply(
        self, fn: callable, x: torch.Tensor, *args, dim=-1, project=True,
        **kwargs
    ) -> torch.Tensor:
        res = math.mobius_fn_apply(fn, x, *args, K=self.get_K(),
                                   dim=dim, **kwargs)
        if project:
            return math.project(res, K=self.get_K(), dim=dim)
        else:
            return res

    # this does not yet work with scaling
    @__scaling__(ScalingInfo.NotCompatible)
    def mobius_fn_apply_chain(
        self, x: torch.Tensor, *fns: callable, project=True, dim=-1
    ) -> torch.Tensor:
        res = math.mobius_fn_apply_chain(x, *fns, K=self.get_K(), dim=dim)
        if project:
            return math.project(res, K=self.get_K(), dim=dim)
        else:
            return res

    # TODO: this way of doing the random normal sampling is not yet right
    # TODO: one has to account for the different densities
    def random_normal(
        self, *size, mean=0, std=1, dtype=None, device=None
    ) -> "geoopt.ManifoldTensor":
        """
        Create a point on the manifold, measure is induced by Normal
        distribution on the tangent space of zero.

        Parameters
        ----------
        size : shape
            the desired shape
        mean : float|tensor
            mean value for the Normal distribution
        std : float|tensor
            std value for the Normal distribution
        dtype: torch.dtype
            target dtype for sample, if not None, should match Manifold dtype
        device: torch.device
            target device for sample, if not None, should match Manifold device

        Returns
        -------
        ManifoldTensor
            random point on the Stereographic manifold

        Notes
        -----
        The device and dtype will match the device and dtype of the Manifold
        """
        self._assert_check_shape(size2shape(*size), "x")
        if device is not None and device != self.c.device:
            raise ValueError(
                "`device` does not match the projector `device`, set the "
                "`device` argument to None"
            )
        if dtype is not None and dtype != self.c.dtype:
            raise ValueError(
                "`dtype` does not match the projector `dtype`, set the `dtype` "
                "arguement to None"
            )
        tens = torch.randn(*size,
                           device=self.c.device,
                           dtype=self.c.dtype) \
               * std + mean
        return geoopt.ManifoldTensor(self.expmap0(tens), manifold=self)

    random = random_normal

    def origin(
        self, *size, dtype=None, device=None, seed=42
    ) -> "geoopt.ManifoldTensor":
        """
        Zero point origin.

        Parameters
        ----------
        size : shape
            the desired shape
        device : torch.device
            the desired device
        dtype : torch.dtype
            the desired dtype
        seed : int
            ignored

        Returns
        -------
        ManifoldTensor
            random point on the manifold
        """
        return geoopt.ManifoldTensor(
            torch.zeros(*size, dtype=dtype, device=device), manifold=self
        )

class StereographicExact(Stereographic):
    __doc__ = r"""{}

    The implementation of retraction is an exact exponential map, this 
    retraction will be used in optimization.

    See Also
    --------
    :class:`Stereographic`
    """.format(
        _stereographic_doc
    )

    reversible = True
    retr_transp = Stereographic.expmap_transp
    transp_follow_retr = Stereographic.transp_follow_expmap
    retr = Stereographic.expmap

    def extra_repr(self):
        return "exact"
