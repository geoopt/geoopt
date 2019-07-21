# author: @nsde
# maintainer: @ferrine


import torch.jit


@torch.jit.script
def torch_pade13(A):  # pragma: no cover
    # avoid torch select operation and unpack coefs
    (b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13) = (
        64764752532480000.0,
        32382376266240000.0,
        7771770303897600.0,
        1187353796428800.0,
        129060195264000.0,
        10559470521600.0,
        670442572800.0,
        33522128640.0,
        1323241920.0,
        40840800.0,
        960960.0,
        16380.0,
        182.0,
        1.0,
    )

    ident = torch.eye(A.shape[1], dtype=A.dtype, device=A.device)
    A2 = torch.matmul(A, A)
    A4 = torch.matmul(A2, A2)
    A6 = torch.matmul(A4, A2)
    U = torch.matmul(
        A,
        torch.matmul(A6, b13 * A6 + b11 * A4 + b9 * A2)
        + b7 * A6
        + b5 * A4
        + b3 * A2
        + b1 * ident,
    )
    V = (
        torch.matmul(A6, b12 * A6 + b10 * A4 + b8 * A2)
        + b6 * A6
        + b4 * A4
        + b2 * A2
        + b0 * ident
    )
    return U, V


@torch.jit.script
def matrix_2_power(x, p):  # pragma: no cover
    for _ in range(int(p)):
        x = x @ x
    return x


@torch.jit.script
def expm_one(A):  # pragma: no cover
    # no checks, this is private implementation
    # but A should be a matrix
    A_fro = torch.norm(A)

    # Scaling step

    n_squarings = torch.clamp(
        torch.ceil(torch.log(A_fro / 5.371920351148152).div(0.6931471805599453)), min=0
    )
    scaling = 2.0 ** n_squarings
    Ascaled = A / scaling

    # Pade 13 approximation
    U, V = torch_pade13(Ascaled)
    P = U + V
    Q = -U + V

    R, _ = torch.solve(P, Q)  # solve P = Q*R
    expmA = matrix_2_power(R, n_squarings)
    return expmA
