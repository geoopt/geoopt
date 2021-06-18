import torch
from typing import Tuple

EPS = {torch.float32: 4e-3, torch.float64: 1e-5}

to_complex = torch.complex
inverse = torch.linalg.inv


def takagi_eig(z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""
    Given a complex symmetric matrix A, the Takagi factorization is an algorithm that
    computes a real diagonal matrix D and a complex unitary matrix S such that:

    .. math::

        A = \overline{S} D S^*

    where:
        - :math:`D` is a real nonnegative diagonal matrix
        - :math:`S` is complex unitary
        - :math:`\overline{S}`: S conjugate
        - :math:`S^*`: S conjugate transpose

    References
    ----------
    - https://en.wikipedia.org/wiki/Matrix_decomposition#Takagi's_factorization
    - https://github.com/hajifkd/takagi_fact/blob/master/takagi_fact/__init__.py

    Parameters
    ----------
    z : torch.Tensor
         Complex symmetric matrix

    Returns
    -------
    torch.Tensor
        Real eigenvalues of z
    torch.Tensor
        Complex eigenvectors of z
    """
    compound_z = to_compound_symmetric(z)  # Z = A + iB, then compound_z = [(A, B),(B, -A)] of 2n x 2n

    evalues, q = torch.linalg.eigh(compound_z)  # b x n in ascending order, b x 2n x 2n

    # I can think of Q as 4 n x n matrices.
    # Q = [(X,  Re(U)),
    #      (Y, -Im(U))]     where X, Y are irrelevant and I need to build U
    real_u_on_top_of_minus_imag_u = torch.chunk(q, 2, dim=-1)[-1]
    real_u, minus_imag_u = torch.chunk(real_u_on_top_of_minus_imag_u, 2, dim=-2)
    u = to_complex(real_u, -minus_imag_u)

    sing_values = evalues[:, z.shape[-1]:]
    return sing_values, u


def takagi_eigvals(z: torch.Tensor) -> torch.Tensor:
    r"""
    Given a complex symmetric matrix A, the Takagi factorization is an algorithm that
    computes a real diagonal matrix D and a complex unitary matrix S such that:

    .. math::

        A = \overline{S} D S^*

    where:
        - :math:`D` is a real nonnegative diagonal matrix
        - :math:`S` is complex unitary
        - :math:`\overline{S}`: S conjugate
        - :math:`S^*`: S conjugate transpose

    References
    ----------
    - https://en.wikipedia.org/wiki/Matrix_decomposition#Takagi's_factorization
    - https://github.com/hajifkd/takagi_fact/blob/master/takagi_fact/__init__.py

    Parameters
    ----------
    z : torch.Tensor
         Complex symmetric matrix

    Returns
    -------
    torch.Tensor
        eigenvalues of z
    """
    compound_z = to_compound_symmetric(z)  # Z = A + iB, then compound_z = [(A, B),(B, -A)] of 2n x 2n
    evalues = torch.linalg.eigvalsh(compound_z)  # b x 2n in ascending order
    sing_values = evalues[:, z.shape[-1]:]
    return sing_values


def cayley_transform(z: torch.Tensor) -> torch.Tensor:
    r"""
    The Cayley transform is used to map elements from the Bounded Domain Model
    to the Upper Half Space Model.

    .. math::

        \operatorname{cayley}(Z): \mathcal{B}_n \to \mathcal{S}_n \\
        \operatorname{cayley}(Z) = i(Z + Id) (Z - Id)^{-1}

    References
    ----------
     - https://en.wikipedia.org/wiki/Cayley_transform#Matrix_map

    Parameters
    ----------
    z : torch.Tensor
         Point in the Upper Half Space model

    Returns
    -------
    torch.Tensor
        Point in the Bounded Domain model
    """
    identity = identity_like(z)

    i_z_plus_id = multiply_by_i(z + identity)
    inv_z_minus_id = inverse(z - identity)

    return i_z_plus_id @ inv_z_minus_id


def inverse_cayley_transform(z: torch.Tensor) -> torch.Tensor:
    r"""
    Cayley transform is used to map elements from the Upper Half Space Model to the Bounded Domain Model.

    .. math::

        \operatorname{cayley}^{-1}(X): \mathcal{S}_n \to \mathcal{B}_n \\
        \operatorname{cayley}^{-1}(X) = (X - i Id) (Z + i Id)^{-1}

    References
    ----------
     - https://en.wikipedia.org/wiki/Cayley_transform#Matrix_map

    Parameters
    ----------
    z : torch.Tensor
         Point in the Upper Half Space model

    Returns
    -------
    torch.Tensor
        Point in the Bounded Domain model
    """
    identity = identity_like(z)
    i_identity = to_complex(identity.imag, identity.real)

    z_minus_id = z - i_identity
    inv_z_plus_id = inverse(z + i_identity)
    return z_minus_id @ inv_z_plus_id


def is_complex_symmetric(x: torch.Tensor, atol=1e-05, rtol=1e-5):
    """
    Returns whether the complex symmetric matrices are symmetric or not

    Parameters
    ----------
    x : tensor
        Complex symmetric matrix
    atol : float
        absolute tolerance for allclose
    rtol : float
        relative tolerance for allclose

    Returns
    -------
    boolean
        whether the points in x are complex symmetric or not
    """
    real_x, imag_x = x.real, x.imag
    return torch.allclose(real_x, real_x.transpose(-1, -2), atol=atol, rtol=rtol) and \
           torch.allclose(imag_x, imag_x.transpose(-1, -2), atol=atol, rtol=rtol)


def to_compound_symmetric(z: torch.Tensor) -> torch.Tensor:
    """
    Let Z = A + iB be a matrix with complex entries, where A, B are n x n matrices with real entries.
    We build a 2n x 2n matrix in the following form:
        M = [(A, B),
             (B, -A)]
    Since Z is symmetric, then M is symmetric and with all real entries.

    Parameters
    ----------
    z : tensor
        Complex symmetric matrix of rank n

    Returns
    -------
    torch.Tensor
        Real symmetric matrix of rank 2n
    """
    a, b = z.real, z.imag
    a_and_b = torch.cat((a, b), dim=-1)
    b_and_minus_a = torch.cat((b, -a), dim=-1)
    m = torch.cat((a_and_b, b_and_minus_a), dim=-2)
    return m


def identity_like(z: torch.Tensor) -> torch.Tensor:
    """
    Return a complex identity of the shape of z, with the same type and device.

    Parameters
    ----------
    z : tensor
        Complex matrix of rank n

    Returns
    -------
    torch.Tensor
        Complex identity of rank n with zeros in the imaginary part
    """
    return torch.eye(z.shape[-1], dtype=z.dtype, device=z.device).expand_as(z)


def multiply_by_i(z: torch.Tensor):
    """
    For Z = X + iY, calculates the operation i Z = i (X + iY) = -Y + iX

    Parameters
    ----------
    z : tensor
        Complex matrix

    Returns
    -------
    torch.Tensor
        Complex matrix
    """
    return to_complex(-z.imag, z.real)
