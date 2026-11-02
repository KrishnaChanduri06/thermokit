"""Laminar flow and heat transfer in a rectangular duct, solved numerically.

Finite-volume solution on the duct cross-section, lengths scaled by D_h:

* fully developed velocity:  ∇²u = -1, no slip on all four walls
* fully developed H1 Nusselt number: axially uniform heat input,
  peripherally uniform temperature on the heated walls, the rest adiabatic
* thermal entry (Graetz) problem for H1, hydrodynamically developed flow,
  marched in x* = x / (D_h Re Pr) with backward Euler

Walls are named left/right (y = 0, w) and bottom/top (z = 0, h). Axial
conduction is neglected, which is fine for Pe = Re Pr above about 100.
"""

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from scipy.sparse import coo_matrix, diags
from scipy.sparse.linalg import splu

WALLS = ("left", "right", "bottom", "top")


def _faces(n, stretch=1.8):
    """Cell faces on [0, 1], clustered towards both walls."""
    xi = np.linspace(0.0, 1.0, n + 1)
    return 0.5 * (1.0 + np.tanh(stretch * (2.0 * xi - 1.0)) / np.tanh(stretch))


class _Grid:
    def __init__(self, w, h, ny, nz):
        self.w, self.h = w, h
        self.yf = w * _faces(ny)
        self.zf = h * _faces(nz)
        self.dy = np.diff(self.yf)
        self.dz = np.diff(self.zf)
        self.yc = 0.5 * (self.yf[1:] + self.yf[:-1])
        self.zc = 0.5 * (self.zf[1:] + self.zf[:-1])
        self.ny, self.nz = ny, nz
        self.vol = np.outer(self.dy, self.dz).ravel()  # index i*nz + j

    def laplacian(self, dirichlet):
        """Integrated Laplacian. Returns (A, b) so that ∫∇²T dA = A T + b T_wall."""
        ny, nz = self.ny, self.nz
        idx = np.arange(ny * nz).reshape(ny, nz)
        rows, cols, vals = [], [], []
        diag = np.zeros(ny * nz)
        b = np.zeros(ny * nz)

        # y-direction faces
        gy = self.dz[None, :] / (self.yc[1:] - self.yc[:-1])[:, None]
        a, c = idx[:-1, :].ravel(), idx[1:, :].ravel()
        g = gy.ravel()
        rows += [a, c]
        cols += [c, a]
        vals += [g, g]
        np.subtract.at(diag, a, g)
        np.subtract.at(diag, c, g)

        # z-direction faces
        gz = self.dy[:, None] / (self.zc[1:] - self.zc[:-1])[None, :]
        a, c = idx[:, :-1].ravel(), idx[:, 1:].ravel()
        g = gz.ravel()
        rows += [a, c]
        cols += [c, a]
        vals += [g, g]
        np.subtract.at(diag, a, g)
        np.subtract.at(diag, c, g)

        # walls: Dirichlet through a half-cell distance, otherwise zero flux
        wall_cells = {
            "left": (idx[0, :], self.dz / (0.5 * self.dy[0])),
            "right": (idx[-1, :], self.dz / (0.5 * self.dy[-1])),
            "bottom": (idx[:, 0], self.dy / (0.5 * self.dz[0])),
            "top": (idx[:, -1], self.dy / (0.5 * self.dz[-1])),
        }
        for wall in dirichlet:
            cells, g = wall_cells[wall]
            np.subtract.at(diag, cells, g)
            np.add.at(b, cells, g)

        rows = np.concatenate(rows + [np.arange(ny * nz)])
        cols = np.concatenate(cols + [np.arange(ny * nz)])
        vals = np.concatenate(vals + [diag])
        A = coo_matrix((vals, (rows, cols)), shape=(ny * nz, ny * nz)).tocsc()
        return A, b

    def heated_perimeter(self, heated):
        return sum(self.h if wl in ("left", "right") else self.w for wl in heated)


@dataclass(frozen=True)
class DuctSolution:
    aspect: float        # w / h
    heated: tuple
    f_re: float          # Fanning f·Re, fully developed
    nu_fd: float         # fully developed Nu_H1, h on heated perimeter, Re and Nu on D_h
    x_star: np.ndarray   # x / (D_h Re Pr)
    nu_x: np.ndarray     # local Nu_H1 at x_star

    def nu_local(self, x_star):
        """Local Nu at x* (array ok). Fully developed value beyond the table."""
        x = np.clip(np.asarray(x_star, dtype=float), self.x_star[0], None)
        nu = np.interp(np.log(x), np.log(self.x_star), self.nu_x)
        return np.where(x > self.x_star[-1], self.nu_fd, nu)

    def nu_mean(self, x_star):
        """Length-averaged Nu from the inlet to x*, H1 (mean of local h)."""
        xs = np.geomspace(self.x_star[0], max(x_star, self.x_star[0] * 1.01), 400)
        return np.trapezoid(self.nu_local(xs), xs) / (xs[-1] - xs[0])


@lru_cache(maxsize=64)
def _solve(aspect, heated, n_short, x_max):
    w, h = aspect, 1.0
    dh = 2.0 * w * h / (w + h)
    w, h = w / dh, h / dh
    if w >= h:
        nz = n_short
        ny = int(min(4 * n_short, max(n_short, round(n_short * w / h))))
    else:
        ny = n_short
        nz = int(min(4 * n_short, max(n_short, round(n_short * h / w))))
    grid = _Grid(w, h, ny, nz)
    area = w * h
    vol = grid.vol

    a_all, _ = grid.laplacian(WALLS)
    u = splu(a_all).solve(-vol)
    u_m = vol @ u / area
    f_re = 1.0 / (2.0 * u_m)
    U = u / u_m

    a_h, b_h = grid.laplacian(heated)
    p_h = grid.heated_perimeter(heated)
    phi = splu(a_h).solve(vol * U)
    phi_b = vol @ (U * phi) / area
    nu_fd = area / (p_h * -phi_b)

    # thermal entry: blocks of constant step, step doubling between blocks
    steps = []
    x, dx = 0.0, 2e-6
    while x < x_max:
        steps += [dx] * 8
        x += 8 * dx
        dx *= 2.0
    vu = vol * U
    t = np.zeros_like(U)
    xs, nus = [], []
    x = 0.0
    lu, dx_lu = None, None
    for dx in steps:
        if dx != dx_lu:
            lu = splu((diags(vu / dx) - a_h).tocsc())
            t_c = lu.solve(b_h)
            dx_lu = dx
        x += dx
        t_a = lu.solve(vu * t / dx)
        t_w = (x * area - vu @ t_a) / (vu @ t_c)
        t = t_a + t_w * t_c
        t_b = vu @ t / area
        xs.append(x)
        nus.append(area / (p_h * (t_w - t_b)))

    return DuctSolution(aspect, heated, f_re, nu_fd, np.array(xs), np.array(nus))


def rectangular_duct(width, height, heated=WALLS, n_short=40, x_max=2.0):
    """Solve the duct once and cache it. width, height in any consistent unit."""
    heated = tuple(sorted(set(heated), key=WALLS.index))
    if not heated:
        raise ValueError("at least one wall must be heated")
    bad = set(heated) - set(WALLS)
    if bad:
        raise ValueError(f"unknown walls {bad}")
    return _solve(round(width / height, 6), heated, n_short, x_max)


def f_re_shah_london(alpha):
    """Fanning f·Re, rectangular duct, alpha = short/long side (Shah & London 1978)."""
    a = min(alpha, 1.0 / alpha)
    return 24 * (1 - 1.3553 * a + 1.9467 * a**2 - 1.7012 * a**3 + 0.9564 * a**4 - 0.2537 * a**5)


def hagenbach_k(alpha):
    """Incremental pressure drop K(∞) for developing laminar flow (Shah & London 1978)."""
    a = min(alpha, 1.0 / alpha)
    return 0.6796 + 1.2197 * a + 3.3089 * a**2 - 9.5921 * a**3 + 8.9089 * a**4 - 2.9959 * a**5
