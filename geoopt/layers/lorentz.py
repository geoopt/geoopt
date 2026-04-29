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


class GyroLorentzBatchNorm(torch.nn.Module):
    """Gyrogroup batch normalization on the Lorentz model."""

    def __init__(
        self,
        num_features,
        *,
        manifold=None,
        eps=1e-5,
        momentum=0.1,
        affine=True,
        track_running_stats=True,
    ):
        super().__init__()
        if num_features < 1:
            raise ValueError("num_features must be positive")

        self.num_features = num_features
        self.manifold = manifold if manifold is not None else geoopt.Lorentz()
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats

        if affine:
            self.bias = geoopt.ManifoldParameter(
                self.manifold.origin(num_features + 1), manifold=self.manifold
            )
            self.log_scale = torch.nn.Parameter(torch.zeros(()))
        else:
            self.register_parameter("bias", None)
            self.register_parameter("log_scale", None)

        if track_running_stats:
            self.register_buffer("running_mean", self.manifold.origin(num_features + 1).data)
            self.register_buffer("running_var", torch.ones(()))
        else:
            self.register_buffer("running_mean", None)
            self.register_buffer("running_var", None)

    def forward(self, input):
        if input.size(-1) != self.num_features + 1:
            raise ValueError(
                f"expected input with last dimension {self.num_features + 1}, "
                f"got {input.size(-1)}"
            )

        if self.training or not self.track_running_stats:
            mean, var = self._compute_batch_stats(input)
            if self.training and self.track_running_stats:
                self._update_running_stats(mean, var)
        else:
            mean = self.running_mean
            var = self.running_var

        centered = self.manifold.gyroadd(self.manifold.gyroinv(mean), input)
        factor = torch.rsqrt(var + self.eps)
        if self.affine:
            factor = self.log_scale.exp() * factor
        output = self.manifold.gyroscalar(factor, centered)
        if self.affine:
            output = self.manifold.gyroadd(self.bias, output)
        return output

    def _compute_batch_stats(self, input):
        reducedim = list(range(input.dim() - 1))
        mean = self.manifold.lorentz_centroid(input, reducedim=reducedim)
        var = self.manifold.lorentz_dispersion(input, mean, reducedim=reducedim)
        return mean, var

    @torch.no_grad()
    def _update_running_stats(self, mean, var):
        direction = self.manifold.logmap(self.running_mean, mean)
        new_mean = self.manifold.expmap(self.running_mean, self.momentum * direction)
        self.running_mean.copy_(new_mean)
        self.running_var.mul_(1 - self.momentum).add_(self.momentum * var.detach())

    def extra_repr(self):
        return (
            f"num_features={self.num_features}, "
            f"eps={self.eps}, "
            f"momentum={self.momentum}, "
            f"affine={self.affine}, "
            f"track_running_stats={self.track_running_stats}"
        )


PLFC = LorentzPLFC
GyroLBN = GyroLorentzBatchNorm
