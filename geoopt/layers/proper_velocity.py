"""
Layers of the Proper Velocity model of hyperbolic space.

References
----------
Ziheng Chen, Zihan Su, Bernhard Schölkopf, Nicu Sebe. Proper Velocity Neural Networks. ICLR 2026.

Xianglong Shi, Nicu Sebe, Bernhard Schölkopf, Ziheng Chen. PVFormer: Proper Velocity Transformer
for Stable and Scalable Hyperbolic Representation Learning. NeurIPS 2026.
"""

import math

import torch

import geoopt


def _rotation(weight):
    return torch.matrix_exp(weight - weight.transpose(-1, -2))


def _rotate(input, weight):
    # (..., N, n) and a stack of H generators (H, n, n) -> (..., H, N, n)
    return input.unsqueeze(-3) @ _rotation(weight).transpose(-1, -2)


def _lift(x, k):
    # time coordinate of the lifted point divided by sqrt(k)
    return torch.sqrt(1 + (x * x).sum(dim=-1, keepdim=True) / k)


def _centroid(b, a, k, eps):
    # PV coordinates of the time-like vector (sqrt(k) a, b) rescaled onto the hyperboloid
    return b / torch.sqrt((a * a - (b * b).sum(dim=-1, keepdim=True) / k).clamp_min(eps))


def _midpoint(values, weights, k, eps):
    # PV midpoint of the rows of values (..., M, n) for every row of weights (..., N, M)
    return _centroid(weights @ values, weights @ _lift(values, k), k, eps)


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


class GyroProperVelocityBatchNorm(torch.nn.Module):
    r"""
    Gyrogroup batch normalization on the Proper Velocity model (Chen et al., 2026, Sec. 5.4).

    .. math::

        \tilde{x}_i = \beta \oplus \left(\frac{s}{\sqrt{\sigma^2 + \epsilon}} \otimes
        (\ominus \mu \oplus x_i)\right)

    where :math:`\mu` and :math:`\sigma^2` are the Fréchet mean and variance of the batch, and
    :math:`\beta` and :math:`s` are the learnable bias and scale. All dimensions except the last
    one are treated as batch dimensions. The Fréchet mean is computed with Karcher flow. By default
    the batch statistics are treated as constants in the backward pass, as in the reference
    implementation of the paper; with ``differentiable_stats=True`` gradients also flow through
    the Karcher flow and the variance, like in standard batch normalization (at a higher cost).

    Parameters
    ----------
    num_features : int
        dimension of the PV points
    manifold : geoopt.ProperVelocity
        manifold of inputs and outputs, defaults to ``geoopt.ProperVelocity()``
    eps : float
        added to the variance for numerical stability
    momentum : float
        momentum of the running statistics
    affine : bool
        learn the bias :math:`\beta` and scale :math:`s`
    track_running_stats : bool
        keep running statistics for evaluation
    max_iter : int
        maximum number of Karcher flow iterations for the Fréchet mean
    tol : float
        tolerance of the Karcher flow
    differentiable_stats : bool
        backpropagate through the batch statistics
    """

    def __init__(
        self,
        num_features,
        *,
        manifold=None,
        eps=1e-5,
        momentum=0.1,
        affine=True,
        track_running_stats=True,
        max_iter=100,
        tol=1e-6,
        differentiable_stats=False,
    ):
        super().__init__()
        if num_features < 1:
            raise ValueError("num_features must be positive")

        self.num_features = num_features
        self.manifold = manifold if manifold is not None else geoopt.ProperVelocity()
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats
        self.max_iter = max_iter
        self.tol = tol
        self.differentiable_stats = differentiable_stats

        if affine:
            self.bias = geoopt.ManifoldParameter(
                self.manifold.origin(num_features), manifold=self.manifold
            )
            self.log_scale = torch.nn.Parameter(torch.zeros(()))
        else:
            self.register_parameter("bias", None)
            self.register_parameter("log_scale", None)

        if track_running_stats:
            self.register_buffer("running_mean", self.manifold.origin(num_features).detach())
            self.register_buffer("running_var", torch.ones(()))
        else:
            self.register_buffer("running_mean", None)
            self.register_buffer("running_var", None)

    def forward(self, input):
        if input.size(-1) != self.num_features:
            raise ValueError(
                f"expected input with last dimension {self.num_features}, got {input.size(-1)}"
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
        with torch.set_grad_enabled(self.differentiable_stats and torch.is_grad_enabled()):
            mean = self.manifold.frechet_mean(
                input, reducedim=reducedim, max_iter=self.max_iter, tol=self.tol
            )
            var = self.manifold.frechet_variance(input, mean, reducedim=reducedim)
        return mean, var

    @torch.no_grad()
    def _update_running_stats(self, mean, var):
        direction = self.manifold.logmap(self.running_mean, mean)
        new_mean = self.manifold.expmap(self.running_mean, self.momentum * direction)
        self.running_mean.copy_(new_mean)
        self.running_var.mul_(1 - self.momentum).add_(self.momentum * var)

    def extra_repr(self):
        return (
            f"num_features={self.num_features}, "
            f"eps={self.eps}, "
            f"momentum={self.momentum}, "
            f"affine={self.affine}, "
            f"track_running_stats={self.track_running_stats}, "
            f"differentiable_stats={self.differentiable_stats}"
        )


class ProperVelocityHomomorphism(torch.nn.Module):
    r"""
    Gyrovector homomorphism of the Proper Velocity model given by a learnable rotation
    (Shi et al., 2026, Thm. 1).

    .. math::

        \Phi_R(x) = R x, \qquad R = \exp(W - W^\top) \in SO(n)

    :math:`\Phi_R` preserves gyroaddition and scalar gyromultiplication,
    :math:`\Phi_R(x \oplus y) = \Phi_R(x) \oplus \Phi_R(y)` and
    :math:`\Phi_R(t \otimes x) = t \otimes \Phi_R(x)`, and it is an isometry of the PV metric for
    every curvature. It replaces the query and value projections of a Transformer in
    :class:`ProperVelocityBusemannAttention` and :class:`ProperVelocityLinearAttention`.
    :math:`W` is initialized to zero, i.e. :math:`R = I`.

    Parameters
    ----------
    features : int
        dimension of the PV points
    """

    def __init__(self, features):
        super().__init__()
        if features < 1:
            raise ValueError("features must be positive")
        self.features = features
        self.weight = torch.nn.Parameter(torch.zeros(features, features))

    @property
    def rotation(self):
        """The rotation matrix :math:`R`."""
        return _rotation(self.weight)

    def forward(self, input):
        if input.size(-1) != self.features:
            raise ValueError(
                f"expected input with last dimension {self.features}, got {input.size(-1)}"
            )
        return input @ self.rotation.transpose(-1, -2)

    def extra_repr(self):
        return f"features={self.features}"

    @torch.no_grad()
    def reset_parameters(self):
        torch.nn.init.zeros_(self.weight)


class ProperVelocityHyperplaneKey(ProperVelocityMLR):
    r"""
    Point-to-hyperplane key map of the Proper Velocity model (Shi et al., 2026, Thm. 4).

    The signed PV point-to-hyperplane margins :math:`v_j(x)` of :class:`ProperVelocityMLR` are
    stacked into a tangent vector at the origin,

    .. math::

        \Psi(x) = \left[v_1(x), \ldots, v_m(x)\right]^\top \in T_0\mathcal{PV}^m_k,

    which serves as a direction-like key for :class:`ProperVelocityBusemannAttention`. As
    :math:`k \to \infty` it reduces to the fully connected layer :math:`x \mapsto Zx + b` with rows
    :math:`z_j` of :math:`Z` and :math:`b_j = -r_j \|z_j\|`. The normals :math:`z_j` are initialized
    with :func:`torch.nn.init.xavier_uniform_` and the offsets :math:`r_j` with zeros.

    Parameters
    ----------
    in_features : int
        dimension of the input PV points
    out_features : int
        dimension of the output tangent keys
    manifold : geoopt.ProperVelocity
        input manifold, defaults to ``geoopt.ProperVelocity()``
    eps : float
        lower bound for :math:`\|z_j\|`
    """

    def __init__(self, in_features, out_features, *, manifold=None, eps=1e-9):
        super().__init__(in_features, out_features, manifold=manifold, eps=eps)

    @torch.no_grad()
    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.z)
        torch.nn.init.zeros_(self.r)


class ProperVelocityBusemannAttention(torch.nn.Module):
    r"""
    Busemann attention on the Proper Velocity model (Shi et al., 2026, Sec. 4.2).

    For every head, queries :math:`q_i` and values :math:`v_j` are rotated by
    :class:`ProperVelocityHomomorphism` maps and keys :math:`k_j` are produced by a
    :class:`ProperVelocityHyperplaneKey`. The logits are PV Busemann similarities of the queries
    to the key directions :math:`\hat{k}_j = k_j / \|k_j\|` (Thm. 2),

    .. math::

        S_{ij} = -\tau_h \sqrt{k} \log\left(\sqrt{1 + \|q_i\|^2 / k}
        - \langle q_i, \hat{k}_j \rangle / \sqrt{k}\right),

    with a learnable temperature :math:`\tau_h > 0` per head. The output is the PV midpoint
    (Thm. 3) of the values under :math:`\alpha_{ij} = \operatorname{softmax}_j(S_{ij} / \sigma)`
    with a learnable scale :math:`\sigma`,

    .. math::

        z_i = \frac{b_i}{\sqrt{a_i^2 - \|b_i\|^2 / k}}, \qquad
        b_i = \sum_j \alpha_{ij} v_j, \qquad
        a_i = \sum_j \alpha_{ij} \sqrt{1 + \|v_j\|^2 / k}.

    Every head works on all ``features`` dimensions and the heads are merged by their uniform PV
    midpoint. The cost is quadratic in the number of tokens, see
    :class:`ProperVelocityLinearAttention` for a linear one.

    Parameters
    ----------
    features : int
        dimension of the PV points
    num_heads : int
        number of attention heads
    manifold : geoopt.ProperVelocity
        manifold of inputs and outputs, defaults to ``geoopt.ProperVelocity()``
    temperature : float
        initial temperature :math:`\tau_h`
    score_clip : float
        if positive, clip the Busemann similarities to ``[-score_clip, score_clip]``
    eps : float
        numerical stability constant

    Examples
    --------
    >>> import torch, geoopt.layers
    >>> attention = geoopt.layers.ProperVelocityBusemannAttention(8, num_heads=2)
    >>> attention(torch.randn(3, 10, 8)).shape
    torch.Size([3, 10, 8])
    """

    def __init__(
        self,
        features,
        num_heads=1,
        *,
        manifold=None,
        temperature=1.0,
        score_clip=0.0,
        eps=1e-8,
    ):
        super().__init__()
        if features < 1:
            raise ValueError("features must be positive")
        if num_heads < 1:
            raise ValueError("num_heads must be positive")
        if temperature <= 0:
            raise ValueError("temperature must be positive")

        self.features = features
        self.num_heads = num_heads
        self.manifold = manifold if manifold is not None else geoopt.ProperVelocity()
        self.init_temperature = temperature
        self.score_clip = score_clip
        self.eps = eps

        self.query_weight = torch.nn.Parameter(torch.empty(num_heads, features, features))
        self.key_map = ProperVelocityHyperplaneKey(
            features, num_heads * features, manifold=self.manifold
        )
        self.value_weight = torch.nn.Parameter(torch.empty(num_heads, features, features))
        self.raw_temperature = torch.nn.Parameter(torch.empty(num_heads, 1, 1))
        self.scale = torch.nn.Parameter(torch.empty(()))
        self.reset_parameters()

    @property
    def temperature(self):
        r"""The per-head temperatures :math:`\tau_h`."""
        return torch.nn.functional.softplus(self.raw_temperature) + self.eps

    def forward(self, query, key=None, value=None, mask=None):
        r"""
        Attend from ``query`` to ``key`` and aggregate ``value``.

        Parameters
        ----------
        query : tensor
            PV points of shape ``(..., N, features)``
        key : tensor
            PV points of shape ``(..., M, features)``, defaults to ``query``
        value : tensor
            PV points of shape ``(..., M, features)``, defaults to ``key``
        mask : tensor
            boolean mask broadcastable to ``(..., num_heads, N, M)`` that is ``True`` where
            attention is allowed, or a float mask added to the logits

        Returns
        -------
        tensor
            PV points of shape ``(..., N, features)``
        """
        key = query if key is None else key
        value = key if value is None else value
        for input in (query, key, value):
            if input.size(-1) != self.features:
                raise ValueError(
                    f"expected input with last dimension {self.features}, got {input.size(-1)}"
                )
        k = self.manifold.k
        sqrt_k = torch.sqrt(k)

        q = _rotate(query, self.query_weight)
        v = _rotate(value, self.value_weight)
        keys = self.key_map(key)
        keys = keys.reshape(keys.shape[:-1] + (self.num_heads, self.features)).transpose(-2, -3)
        direction = keys / keys.norm(dim=-1, keepdim=True).clamp_min(self.eps)

        argument = _lift(q, k) - q @ direction.transpose(-1, -2) / sqrt_k
        logits = -self.temperature * sqrt_k * torch.log(argument.clamp_min(self.eps))
        if self.score_clip > 0:
            logits = logits.clamp(min=-self.score_clip, max=self.score_clip)
        logits = logits / self.scale.abs().clamp_min(self.eps)
        if mask is not None:
            if mask.dtype == torch.bool:
                logits = logits.masked_fill(~mask, float("-inf"))
            else:
                logits = logits + mask

        output = _midpoint(v, torch.softmax(logits, dim=-1), k, self.eps)
        if self.num_heads == 1:
            return output.squeeze(-3)
        heads = output.transpose(-2, -3)
        uniform = heads.new_full((1, self.num_heads), 1.0 / self.num_heads)
        return _midpoint(heads, uniform, k, self.eps).squeeze(-2)

    def extra_repr(self):
        return (
            f"features={self.features}, "
            f"num_heads={self.num_heads}, "
            f"temperature={self.init_temperature}, "
            f"score_clip={self.score_clip}"
        )

    @torch.no_grad()
    def reset_parameters(self):
        torch.nn.init.zeros_(self.query_weight)
        torch.nn.init.zeros_(self.value_weight)
        for head in range(self.num_heads):
            rows = slice(head * self.features, (head + 1) * self.features)
            torch.nn.init.xavier_uniform_(self.key_map.z[rows])
        torch.nn.init.zeros_(self.key_map.r)
        self.raw_temperature.fill_(math.log(math.expm1(self.init_temperature)))
        self.scale.fill_(math.sqrt(self.features))


class ProperVelocityLinearAttention(torch.nn.Module):
    r"""
    Linear attention on the Proper Velocity model (Shi et al., 2026, Sec. 4.4).

    For every head, queries, keys and values are rotated by :class:`ProperVelocityHomomorphism`
    maps and compared with the focused nonnegative feature map

    .. math::

        \varphi_p(e) = \frac{\|\tilde{e}\|}{\|\tilde{e}^{p}\|} \tilde{e}^{p}, \qquad
        \tilde{e} = \frac{\operatorname{ReLU}(e) + \epsilon}{t},

    with a learnable scale :math:`t`. The output is the PV midpoint of the values under weights
    proportional to :math:`\lambda_{ij} = \varphi_p(q_i)^\top \varphi_p(k_j)`, which can be
    evaluated without forming the :math:`N \times M` weight matrix (Thm. 5),

    .. math::

        o_i = \frac{B_i}{\sqrt{A_i^2 - \|B_i\|^2 / k}}, \qquad
        \left[B_i, A_i\right] = \varphi_p(q_i)^\top
        \sum_j \varphi_p(k_j) \left[v_j, \sqrt{1 + \|v_j\|^2 / k}\right].

    :math:`B_i` and :math:`A_i` are divided by :math:`\sum_j \lambda_{ij}`, which leaves
    :math:`o_i` unchanged and keeps it accurate when all kernel weights of a query are tiny.
    Every head works on all ``features`` dimensions. With several heads their outputs are mapped to
    the tangent space at the origin, concatenated, merged by a linear layer and mapped back.

    Parameters
    ----------
    features : int
        dimension of the PV points
    num_heads : int
        number of attention heads
    manifold : geoopt.ProperVelocity
        manifold of inputs and outputs, defaults to ``geoopt.ProperVelocity()``
    power : float
        focusing power :math:`p` of the feature map
    eps : float
        numerical stability constant :math:`\epsilon`

    Examples
    --------
    >>> import torch, geoopt.layers
    >>> attention = geoopt.layers.ProperVelocityLinearAttention(8, num_heads=2)
    >>> attention(torch.randn(3, 1000, 8)).shape
    torch.Size([3, 1000, 8])
    """

    def __init__(self, features, num_heads=1, *, manifold=None, power=2.0, eps=1e-8):
        super().__init__()
        if features < 1:
            raise ValueError("features must be positive")
        if num_heads < 1:
            raise ValueError("num_heads must be positive")
        if power <= 0:
            raise ValueError("power must be positive")

        self.features = features
        self.num_heads = num_heads
        self.manifold = manifold if manifold is not None else geoopt.ProperVelocity()
        self.power = power
        self.eps = eps

        self.query_weight = torch.nn.Parameter(torch.empty(num_heads, features, features))
        self.key_weight = torch.nn.Parameter(torch.empty(num_heads, features, features))
        self.value_weight = torch.nn.Parameter(torch.empty(num_heads, features, features))
        self.feature_scale = torch.nn.Parameter(torch.empty(()))
        if num_heads > 1:
            self.merge = torch.nn.Linear(num_heads * features, features)
        else:
            self.merge = None
        self.reset_parameters()

    def feature_map(self, input):
        r"""Compute the focused feature map :math:`\varphi_p`."""
        input = (torch.relu(input) + self.eps) / self.feature_scale.abs().clamp_min(self.eps)
        powered = input.pow(self.power)
        norm = powered.norm(dim=-1, keepdim=True).clamp_min(self.eps)
        return input.norm(dim=-1, keepdim=True) * powered / norm

    def forward(self, query, key=None, value=None, mask=None):
        r"""
        Attend from ``query`` to ``key`` and aggregate ``value``.

        Parameters
        ----------
        query : tensor
            PV points of shape ``(..., N, features)``
        key : tensor
            PV points of shape ``(..., M, features)``, defaults to ``query``
        value : tensor
            PV points of shape ``(..., M, features)``, defaults to ``key``
        mask : tensor
            boolean mask of shape ``(..., M)`` that is ``True`` for the keys to attend to

        Returns
        -------
        tensor
            PV points of shape ``(..., N, features)``
        """
        key = query if key is None else key
        value = key if value is None else value
        for input in (query, key, value):
            if input.size(-1) != self.features:
                raise ValueError(
                    f"expected input with last dimension {self.features}, got {input.size(-1)}"
                )
        k = self.manifold.k

        phi_q = self.feature_map(_rotate(query, self.query_weight))
        phi_k = self.feature_map(_rotate(key, self.key_weight))
        if mask is not None:
            phi_k = phi_k * mask[..., None, :, None].to(phi_k.dtype)
        v = _rotate(value, self.value_weight)

        ones = torch.ones_like(v[..., :1])
        summary = phi_k.transpose(-1, -2) @ torch.cat((v, _lift(v, k), ones), dim=-1)
        sums = phi_q @ summary
        total = sums[..., -1:].clamp_min(torch.finfo(sums.dtype).tiny)
        output = _centroid(sums[..., :-2] / total, sums[..., -2:-1] / total, k, self.eps)
        if self.num_heads == 1:
            return output.squeeze(-3)
        tangent = self.manifold.logmap0(output).transpose(-2, -3).flatten(-2)
        return self.manifold.expmap0(self.merge(tangent))

    def extra_repr(self):
        return f"features={self.features}, num_heads={self.num_heads}, power={self.power}"

    @torch.no_grad()
    def reset_parameters(self):
        torch.nn.init.zeros_(self.query_weight)
        torch.nn.init.zeros_(self.key_weight)
        torch.nn.init.zeros_(self.value_weight)
        torch.nn.init.ones_(self.feature_scale)
        if self.merge is not None:
            self.merge.reset_parameters()


PVMLR = ProperVelocityMLR
PVFC = ProperVelocityFC
GyroPVBN = GyroProperVelocityBatchNorm
PVHomomorphism = ProperVelocityHomomorphism
PVHyperplaneKey = ProperVelocityHyperplaneKey
PVBusemannAttention = ProperVelocityBusemannAttention
PVLinearAttention = ProperVelocityLinearAttention
