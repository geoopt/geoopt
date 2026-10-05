import torch

import geoopt


class ProperVelocityMLR(torch.nn.Module):
    r"""
    Multinomial logistic regression on the Proper Velocity model.

    Each output is the signed distance to a PV hyperplane scaled by the norm of its normal
    (Chen et al., 2026, Thm. 5.2), parameterized by :math:`z_j \in \mathbb{R}^n` and
    :math:`r_j \in \mathbb{R}`:

    .. math::

        v_j(x) = \sqrt{k} \|z_j\| \operatorname{arsinh}\left(
        \frac{\cosh(r_j / \sqrt{k}) \langle x, z_j \rangle - \sinh(r_j / \sqrt{k}) \|z_j\| \sqrt{k + \|x\|^2}}
        {\sqrt{k} \|z_j\|}\right)

    The hyperplane passes through :math:`p_j = \operatorname{Exp}_0(r_j z_j / \|z_j\|)` with normal
    :math:`\operatorname{PT}_{0 \to p_j}(z_j)`.

    Parameters
    ----------
    in_features : int
        dimension of the input PV points
    out_features : int
        number of outputs (classes)
    manifold : geoopt.ProperVelocity
        input manifold, defaults to ``geoopt.ProperVelocity()``
    eps : float
        lower bound for :math:`\|z_j\|`
    init_std : float
        standard deviation of the normal initialization of :math:`z`
    """

    def __init__(self, in_features, out_features, *, manifold=None, eps=1e-9, init_std=0.02):
        super().__init__()
        if in_features < 1:
            raise ValueError("in_features must be positive")
        if out_features < 1:
            raise ValueError("out_features must be positive")

        self.in_features = in_features
        self.out_features = out_features
        self.manifold = manifold if manifold is not None else geoopt.ProperVelocity()
        self.eps = eps
        self.init_std = init_std

        self.z = torch.nn.Parameter(torch.empty(out_features, in_features))
        self.r = torch.nn.Parameter(torch.empty(out_features))
        self.reset_parameters()

    def forward(self, input):
        return self.signed_distance(input)

    def signed_distance(self, input):
        r"""Compute the PV MLR scores :math:`v_j(x)`."""
        if input.size(-1) != self.in_features:
            raise ValueError(
                f"expected input with last dimension {self.in_features}, got {input.size(-1)}"
            )
        k = self.manifold.k
        sqrt_k = torch.sqrt(k)

        z_norm = torch.linalg.norm(self.z, dim=-1).clamp_min(self.eps)
        cosh_term = torch.cosh(self.r / sqrt_k)
        sinh_term = torch.sinh(self.r / sqrt_k)

        inner = torch.einsum("...i,oi->...o", input, self.z)
        input_time = torch.sqrt(k + (input * input).sum(dim=-1, keepdim=True))
        alpha = cosh_term * inner - sinh_term * z_norm * input_time
        return sqrt_k * z_norm * torch.asinh(alpha / (sqrt_k * z_norm))

    def extra_repr(self):
        return f"in_features={self.in_features}, out_features={self.out_features}"

    @torch.no_grad()
    def reset_parameters(self):
        torch.nn.init.normal_(self.z, mean=0.0, std=self.init_std)
        torch.nn.init.zeros_(self.r)


class ProperVelocityFC(ProperVelocityMLR):
    r"""
    Fully connected layer on the Proper Velocity model (Chen et al., 2026, Thm. 5.3).

    Output coordinate :math:`j` is the PV point whose signed distance to the :math:`j`-th coordinate
    hyperplane equals the PV MLR score :math:`v_j(x)`:

    .. math::

        y_j = \sqrt{k} \sinh\left(\frac{v_j(x)}{\sqrt{k}}\right)

    followed by an optional gyro bias :math:`y \oplus \operatorname{Exp}_0(b)`.

    Parameters
    ----------
    in_features : int
        dimension of the input PV points
    out_features : int
        dimension of the output PV points
    manifold : geoopt.ProperVelocity
        manifold of inputs and outputs, defaults to ``geoopt.ProperVelocity()``
    bias : bool
        add a gyro bias
    eps : float
        lower bound for :math:`\|z_j\|`
    init_std : float
        standard deviation of the normal initialization of :math:`z`
    """

    def __init__(
        self,
        in_features,
        out_features,
        *,
        manifold=None,
        bias=True,
        eps=1e-9,
        init_std=0.02,
    ):
        super().__init__(in_features, out_features, manifold=manifold, eps=eps, init_std=init_std)
        if bias:
            self.bias = torch.nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias", None)

    def forward(self, input):
        distance = self.signed_distance(input)
        sqrt_k = torch.sqrt(self.manifold.k)
        output = sqrt_k * torch.sinh(distance / sqrt_k)
        if self.bias is not None:
            output = self.manifold.gyroadd(output, self.manifold.expmap0(self.bias))
        return output

    def extra_repr(self):
        return (
            f"in_features={self.in_features}, "
            f"out_features={self.out_features}, "
            f"bias={self.bias is not None}"
        )

    @torch.no_grad()
    def reset_parameters(self):
        super().reset_parameters()
        if getattr(self, "bias", None) is not None:
            torch.nn.init.zeros_(self.bias)


PVMLR = ProperVelocityMLR
PVFC = ProperVelocityFC
