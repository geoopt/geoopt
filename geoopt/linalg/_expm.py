"""
author: nsde
maintainer: ferrine
"""

import torch


@torch.jit.script
def torch_pade13(A):
    b = [
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
    ]

    ident = torch.eye(A.shape[1], dtype=A.dtype, device=A.device)
    A2 = torch.matmul(A, A)
    A4 = torch.matmul(A2, A2)
    A6 = torch.matmul(A4, A2)
    U = torch.matmul(
        A,
        torch.matmul(A6, b[13] * A6 + b[11] * A4 + b[9] * A2)
        + b[7] * A6
        + b[5] * A4
        + b[3] * A2
        + b[1] * ident,
    )
    V = (
        torch.matmul(A6, b[12] * A6 + b[10] * A4 + b[8] * A2)
        + b[6] * A6
        + b[4] * A4
        + b[2] * A2
        + b[0] * ident
    )
    return U, V


@torch.jit.script
def matrix_2_power(x, p):
    while bool(p > 0):
        x = x @ x
        p = p - 1
    return x


@torch.jit.script
def expm_one(A):
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
    R, _ = torch.gesv(P, Q)  # solve P = Q*R
    expmA = matrix_2_power(R, n_squarings)
    return expmA
