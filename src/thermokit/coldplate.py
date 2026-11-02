"""Straight-channel liquid cold plate (micro or mini channels)."""

from dataclasses import dataclass, field

import fluids
import ht
import numpy as np

from .duct import hagenbach_k, rectangular_duct
from .props import fluid

LPM = 1e-3 / 60.0  # L/min -> m³/s


@dataclass
class ColdPlate:
    """Parallel rectangular channels machined into a base, closed by a lid.

    width        plate width across the channels (m)
    length       channel length in the flow direction (m), heated over its full length
    channel_width, channel_height, fin_width, base_thickness (m)
    k            solid conductivity (W/(m·K))
    lid          "adiabatic" (polymer or unbonded cover) or "bonded" (brazed metal lid)
    """

    width: float
    length: float
    channel_width: float
    channel_height: float
    fin_width: float
    base_thickness: float
    k: float = 390.0
    lid: str = "adiabatic"
    n_channels: int | None = None

    def __post_init__(self):
        if self.lid not in ("adiabatic", "bonded"):
            raise ValueError("lid must be 'adiabatic' or 'bonded'")
        if self.n_channels is None:
            self.n_channels = int(self.width // self.pitch)
        if self.n_channels < 1:
            raise ValueError("plate is narrower than one channel pitch")

    @property
    def pitch(self):
        return self.channel_width + self.fin_width

    @property
    def d_h(self):
        w, h = self.channel_width, self.channel_height
        return 2 * w * h / (w + h)

    @property
    def flow_area(self):
        return self.n_channels * self.channel_width * self.channel_height

    @property
    def heated_walls(self):
        walls = ("left", "right", "bottom")
        return walls + ("top",) if self.lid == "bonded" else walls

    def flow_for_re(self, re, t_c=25.0, coolant="water", fraction=None):
        """Flow rate in L/min that gives channel Re at t_c."""
        f = fluid(coolant, t_c, fraction=fraction)
        u = re * f.mu / (f.rho * self.d_h)
        return u * self.flow_area / LPM

    def solve(self, flow_lpm, power, t_in=25.0, coolant="water", fraction=None,
              k_minor=0.0, developing=True, wall_viscosity=True, regime="auto", n_x=200):
        """Operating point.

        k_minor adds manifold or plenum losses in velocity heads.
        developing=False uses the fully developed Nu everywhere (laminar only),
        which is what most hand calculations do.
        wall_viscosity applies the heated-liquid property correction (laminar).
        regime "auto" is laminar below Re = 2300, turbulent above 10⁴ and a
        linear blend in between (Gnielinski, IJHMT 63, 2013). "laminar" or
        "turbulent" forces one branch, e.g. to compare with laminar CFD.
        """
        if regime not in ("auto", "laminar", "turbulent"):
            raise ValueError("regime must be auto, laminar or turbulent")
        vdot = flow_lpm * LPM
        f_in = fluid(coolant, t_in, fraction=fraction)
        m_dot = f_in.rho * vdot
        t_out = t_in + power / (m_dot * f_in.cp)
        f = fluid(coolant, 0.5 * (t_in + t_out), fraction=fraction)

        w, h, L = self.channel_width, self.channel_height, self.length
        dh = self.d_h
        u = m_dot / (f.rho * self.flow_area)
        re = f.rho * u * dh / f.mu
        x = np.linspace(0.0, L, n_x)
        dyn = 0.5 * f.rho * u**2
        q_len = power / (self.n_channels * L)         # W/m per channel
        t_bulk = t_in + (t_out - t_in) * x / L
        notes = []

        def laminar_terms(re_l):
            duct = rectangular_duct(w, h, self.heated_walls)
            if developing:
                nu_l = duct.nu_local(np.maximum(x, L / n_x / 10) / (dh * re_l * f.pr))
            else:
                nu_l = np.full_like(x, duct.nu_fd)
            return nu_l, 4 * duct.f_re / re_l * L / dh, hagenbach_k(w / h)

        def turbulent_terms(re_t):
            fd = fluids.friction_factor(re_t)
            return np.full_like(x, ht.turbulent_Gnielinski(re_t, f.pr, fd)), fd * L / dh, 0.0

        if regime == "auto":
            regime = "laminar" if re < 2300 else "turbulent" if re > 1e4 else "transitional"

        if regime == "laminar":
            nu_cp, friction, k_dev = laminar_terms(re)
            if L / (dh * re * f.pr) < 0.05:
                notes.append("flow is thermally developing over the whole channel")
        elif regime == "turbulent":
            nu_cp, friction, k_dev = turbulent_terms(re)
            notes.append("turbulent Nu is fully developed, entry enhancement is not counted")
        else:
            gamma = (re - 2300) / (1e4 - 2300)
            nu_l, fr_l, k_l = laminar_terms(2300.0)
            nu_t, fr_t, _ = turbulent_terms(1e4)
            nu_cp = (1 - gamma) * nu_l + gamma * nu_t
            friction = (1 - gamma) * fr_l + gamma * fr_t
            k_dev = (1 - gamma) * k_l
            notes.append(f"Re = {re:.0f} is transitional, laminar and turbulent results are "
                         f"blended ({gamma:.0%} turbulent), treat the answer as rough")
        minor = k_dev + k_minor

        # Heated liquid, laminar: Nu ∝ (μ_b/μ_w)^0.14 and f ∝ (μ_w/μ_b)^0.58
        # (Deissler, in Kakaç, Shah & Aung, Handbook of Single-Phase Convective
        # Heat Transfer, 1987). Iterated on the mean channel-wall temperature.
        correct = regime == "laminar" and wall_viscosity and coolant.lower() != "air"
        ratio = 1.0
        for _ in range(5 if correct else 1):
            nu = nu_cp * ratio**-0.14
            hx = nu * f.k / dh
            mh = np.sqrt(2 * hx / (self.k * self.fin_width)) * h
            eta = np.tanh(mh) / mh
            wetted = w + 2 * eta * h
            if self.lid == "bonded":
                wetted = wetted + w / np.cosh(mh)
            t_wall = t_bulk + q_len / (hx * wetted)
            if correct:
                ratio = fluid(coolant, float(t_wall.mean()), fraction=fraction).mu / f.mu

        dp = (friction * ratio**0.58 + minor) * dyn
        t_base = t_wall + q_len * self.base_thickness / (self.k * self.pitch)

        return ColdPlateResult(
            plate=self, flow_lpm=flow_lpm, power=power, t_in=t_in, t_out=t_out,
            re=re, pr=f.pr, velocity=u, regime=regime,
            dp=dp, pumping_power=vdot * dp, viscosity_ratio=ratio,
            x=x, nu=nu, h=hx, fin_efficiency=eta, t_bulk=t_bulk, t_wall=t_wall,
            t_base=t_base, q_per_length=q_len, notes=notes,
        )


@dataclass
class ColdPlateResult:
    plate: ColdPlate
    flow_lpm: float
    power: float
    t_in: float
    t_out: float
    re: float
    pr: float
    velocity: float
    regime: str
    dp: float
    pumping_power: float
    viscosity_ratio: float   # μ_wall / μ_bulk used in the correction
    x: np.ndarray
    nu: np.ndarray
    h: np.ndarray
    fin_efficiency: np.ndarray
    t_bulk: np.ndarray
    t_wall: np.ndarray       # channel floor
    t_base: np.ndarray       # heated face of the base
    q_per_length: float
    notes: list = field(default_factory=list)

    @property
    def t_base_max(self):
        return self.t_base.max()

    @property
    def r_th(self):
        """(T_base,max - T_in) / Q, K/W."""
        return (self.t_base_max - self.t_in) / self.power

    @property
    def r_th_mean(self):
        return (np.trapezoid(self.t_base, self.x) / self.x[-1] - self.t_in) / self.power

    def t_at_depth(self, depth):
        """Temperature inside the base, depth below the channel floor (m)."""
        p = self.plate
        return self.t_wall + self.q_per_length * depth / (p.k * p.pitch)

    def summary(self):
        lines = [
            f"{self.plate.n_channels} channels, Re = {self.re:.0f} ({self.regime}), u = {self.velocity:.2f} m/s",
            f"T_out = {self.t_out:.1f} °C, T_base,max = {self.t_base_max:.1f} °C",
            f"R_th = {self.r_th * 1e3:.1f} mK/W (peak), {self.r_th_mean * 1e3:.1f} mK/W (mean)",
            f"Δp = {self.dp / 1e3:.2f} kPa, pumping power = {self.pumping_power:.3f} W",
        ]
        return "\n".join(lines + [f"note: {n}" for n in self.notes])
