import torch
import geoopt


class LorentzPLFC(torch.nn.Module):
    """Point-to-hyperplane fully connected layer on the Lorentz model."""

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
        super().__init__()
        if in_features < 1:
            raise ValueError("in_features must be positive")
        if out_features < 1:
            raise ValueError("out_features must be positive")

        self.in_features = in_features
        self.out_features = out_features
        self.manifold = manifold if manifold is not None else geoopt.Lorentz()
        self.eps = eps
        self.init_std = init_std

        self.z = torch.nn.Parameter(torch.empty(out_features, in_features))
        self.a = torch.nn.Parameter(torch.empty(out_features))
        if bias:
            self.bias = torch.nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def forward(self, input):
        distance = self.signed_distance(input)
        sqrt_k = torch.sqrt(self.manifold.k)
        space = sqrt_k * torch.sinh(distance / sqrt_k)
        time = torch.sqrt(self.manifold.k + (space * space).sum(dim=-1, keepdim=True))
        output = torch.cat((time, space), dim=-1)

        if self.bias is not None:
            zero = torch.zeros_like(self.bias[..., :1])
            bias_tangent = torch.cat((zero, self.bias), dim=-1)
            bias = self.manifold.expmap0(bias_tangent)
            output = self.manifold.gyroadd(output, bias)
        return output

    def signed_distance(self, input):
        r"""Compute signed Lorentz point-to-hyperplane distances."""
        input_time = input.narrow(-1, 0, 1)
        input_space = input.narrow(-1, 1, self.in_features)
        sqrt_k = torch.sqrt(self.manifold.k)

        z_norm = torch.linalg.norm(self.z, dim=-1).clamp_min(self.eps)
        cosh_term = torch.cosh(self.a / sqrt_k)
        sinh_term = torch.sinh(self.a / sqrt_k)

        inner = torch.einsum("...i,oi->...o", input_space, self.z)
        alpha = cosh_term * inner - sinh_term * z_norm * input_time
        beta = torch.sqrt(((cosh_term * z_norm) ** 2 - (sinh_term * z_norm) ** 2).clamp_min(self.eps))
        return sqrt_k * beta * torch.asinh(alpha / (sqrt_k * beta))

    def extra_repr(self):
        return (
            f"in_features={self.in_features}, "
            f"out_features={self.out_features}, "
            f"bias={self.bias is not None}"
        )

    @torch.no_grad()
    def reset_parameters(self):
        torch.nn.init.normal_(self.z, mean=0.0, std=self.init_std)
        torch.nn.init.zeros_(self.a)
        if self.bias is not None:
            torch.nn.init.zeros_(self.bias)


PLFC = LorentzPLFC
