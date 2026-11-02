"""Air-cooled plate-fin heat sink, forced and natural convection.

Forced convection, fully shrouded (no bypass):
    heat transfer   Teertstra, Yovanovich & Culham, Proc. 15th IEEE SEMI-THERM,
                    1999, pp. 34-41. Composite of developing and fully developed
                    parallel-plate flow, quoted in full by Zaghlol et al., ITherm
                    2002, who measured -3.5 % to +7.2 % against it for an all
                    copper sink. Stated range 0.26 < Re_b* < 175.
    pressure drop   Culham & Muzychka, IEEE Trans. CPT 24(2), 2001: sudden
                    contraction and expansion plus apparent friction for
                    developing laminar flow (Shah & London f·Re).

Natural convection, vertical fins, open top and bottom:
    Bar-Cohen & Rohsenow, J. Heat Transfer 106, 1984, symmetric isothermal
    plates (Incropera eq. 9.45). Radiation is not counted.

Geometry: width W across the fins, length L along the flow (vertical for
natural convection), N fins of height H and thickness t on a base of
thickness t_b. Outer fins sit on the base edges, so there are N - 1 channels.
The base is taken as isothermal here; put the sink in a stack behind a
Spreader to add spreading from a small source.
"""

from dataclasses import dataclass, field, replace

import numpy as np
from scipy.optimize import brentq

from .duct import f_re_shah_london
from .props import fluid
from .stack import Fixed, Spreader

G = 9.81


@dataclass
class PlateFinHeatSink:
    width: float
    length: float
    fin_height: float
    fin_thickness: float
    n_fins: int
    base_thickness: float
    k: float = 200.0
    density: float = 2700.0

    def __post_init__(self):
        if self.n_fins < 2:
            raise ValueError("need at least two fins")
        if self.spacing <= 0:
            raise ValueError("fins do not fit on the base")

    @property
    def spacing(self):
        return (self.width - self.n_fins * self.fin_thickness) / (self.n_fins - 1)

    @property
    def n_channels(self):
        return self.n_fins - 1

    @property
    def fin_area(self):
        """Fin faces bounding the channels (outer faces of the end fins excluded)."""
        return 2 * self.n_channels * self.fin_height * self.length

    @property
    def base_area(self):
        return self.width * self.length

    @property
    def channel_dh(self):
        s, H = self.spacing, self.fin_height
        return 2 * s * H / (s + H)

    @property
    def mass(self):
        vol = self.base_area * self.base_thickness + self.n_fins * self.fin_thickness * self.fin_height * self.length
        return self.density * vol

    # -- forced convection ---------------------------------------------------

    def forced(self, flow, t_amb=25.0, power=None):
        """Forced convection at volumetric flow (m³/s) through the fin array."""
        air = fluid("air", t_amb)
        s, H, L, t = self.spacing, self.fin_height, self.length, self.fin_thickness
        v = flow / (self.n_channels * s * H)
        re_b = v * s / air.nu
        re_star = re_b * s / L
        nu = teertstra_nu(re_star, air.pr)
        h = nu * air.k / s

        mh = np.sqrt(2 * nu * (air.k / self.k) * (H / s) * (H / t) * (t / L + 1))
        eta = np.tanh(mh) / mh
        r_th = 1.0 / (h * eta * self.fin_area)

        dp, re_dh = self._pressure_drop(v, air)
        notes = []
        if not 0.26 <= re_star <= 175:
            notes.append(f"Re_b* = {re_star:.2f} is outside the model range 0.26 to 175")
        if re_dh > 2300:
            notes.append(f"channel Re_Dh = {re_dh:.0f}, flow may be turbulent, laminar model used")

        t_rise = power / (air.rho * flow * air.cp) if power else None
        return HeatSinkResult(self, "forced", r_th, h, eta, flow=flow, velocity=v, re=re_b,
                              re_star=re_star, dp=dp, air_rise=t_rise, notes=notes)

    def _pressure_drop(self, v, air):
        s, H, L = self.spacing, self.fin_height, self.length
        dh = self.channel_dh
        sigma = 1 - self.n_fins * self.fin_thickness / self.width
        k_c = 0.42 * (1 - sigma**2)
        k_e = (1 - sigma**2) ** 2
        re_dh = v * dh / air.nu
        l_star = L / (dh * re_dh)
        f_app = np.sqrt((3.44 / np.sqrt(l_star)) ** 2 + f_re_shah_london(s / H) ** 2) / re_dh
        dp = (k_c + 4 * f_app * L / dh + k_e) * 0.5 * air.rho * v**2
        return dp, re_dh

    def pressure_drop(self, flow, t_amb=25.0):
        air = fluid("air", t_amb)
        v = flow / (self.n_channels * self.spacing * self.fin_height)
        return self._pressure_drop(v, air)[0]

    def operating_point(self, fan_flow, fan_dp, t_amb=25.0, power=None):
        """Intersect a fan curve (flow m³/s, static pressure Pa) with the sink."""
        q = np.asarray(fan_flow, dtype=float)
        p = np.asarray(fan_dp, dtype=float)
        order = np.argsort(q)
        q, p = q[order], p[order]

        def gap(x):
            return np.interp(x, q, p) - self.pressure_drop(x, t_amb)

        lo = max(q[0], 1e-9)
        if gap(lo) <= 0:
            raise ValueError("fan cannot push air through this heat sink")
        if gap(q[-1]) > 0:
            raise ValueError("fan curve ends before it meets the sink curve, extend it to free delivery")
        return self.forced(brentq(gap, lo, q[-1]), t_amb, power)

    # -- natural convection --------------------------------------------------

    def _natural_at(self, dt, t_amb):
        s, H, L, t = self.spacing, self.fin_height, self.length, self.fin_thickness
        air = fluid("air", t_amb + 0.5 * dt)
        ra = G * air.beta * dt * s**3 / (air.nu * air.alpha)
        el = ra * s / L
        nu = (576 / el**2 + 2.873 / np.sqrt(el)) ** -0.5
        h = nu * air.k / s
        mh = np.sqrt(2 * h / (self.k * t)) * (H + t / 2)
        eta = np.tanh(mh) / mh
        r_th = 1.0 / (h * (eta * self.fin_area + self.n_channels * s * L))
        return r_th, h, eta, ra, el

    def natural(self, power, t_amb=25.0, dt_max=400.0):
        """Natural convection with vertical fins.

        Solves ΔT = P · R(ΔT) for the base rise. R falls as ΔT grows, so the
        root is unique.
        """
        def gap(dt):
            return dt - power * self._natural_at(dt, t_amb)[0]

        if gap(dt_max) < 0:
            raise ValueError(f"base would run more than {dt_max:g} K above ambient, sink is far too small")
        dt = brentq(gap, 1e-4, dt_max, xtol=1e-6)
        r_th, h, eta, ra, el = self._natural_at(dt, t_amb)
        notes = ["radiation not counted, so R_th is on the safe side"]
        return HeatSinkResult(self, "natural", r_th, h, eta, rayleigh=ra, elenbaas=el,
                              base_rise=dt, notes=notes)

    def optimum_spacing_natural(self, dt, t_amb=25.0):
        """Bar-Cohen & Rohsenow optimum gap, symmetric isothermal plates.

        S_opt = 2.714 (Ra_S / (S³ L))^(-1/4), Incropera eq. 9.47.
        """
        air = fluid("air", t_amb + 0.5 * dt)
        p = G * air.beta * dt / (air.nu * air.alpha * self.length)
        return 2.714 * p**-0.25

    def layers(self, source_area, result):
        """Base spreading plus this result, ready for thermokit.solve."""
        return [
            Spreader("heat sink base", source_area, self.base_area, self.base_thickness, self.k),
            Fixed("fins to air", result.r_th),
        ]


def teertstra_nu(re_star, pr):
    """Channel Nusselt number on fin spacing, referenced to inlet air temperature."""
    fd = re_star * pr / 2
    dev = 0.664 * np.sqrt(re_star) * pr ** (1 / 3) * np.sqrt(1 + 3.65 / np.sqrt(re_star))
    return (fd**-3 + dev**-3) ** (-1 / 3)


@dataclass
class HeatSinkResult:
    sink: PlateFinHeatSink
    mode: str
    r_th: float          # base (isothermal) to inlet air, K/W
    h: float
    fin_efficiency: float
    flow: float = None
    velocity: float = None
    re: float = None
    re_star: float = None
    dp: float = None
    air_rise: float = None
    rayleigh: float = None
    elenbaas: float = None
    base_rise: float = None
    notes: list = field(default_factory=list)

    def summary(self):
        s = self.sink
        lines = [f"{s.n_fins} fins, gap {s.spacing * 1e3:.2f} mm, {self.mode} convection"]
        if self.mode == "forced":
            lines.append(f"flow {self.flow * 3600:.1f} m³/h ({self.flow * 2118.88:.1f} CFM), "
                         f"channel velocity {self.velocity:.2f} m/s, Re_b* = {self.re_star:.2f}")
            lines.append(f"Δp = {self.dp:.1f} Pa")
            if self.air_rise is not None:
                lines.append(f"air temperature rise {self.air_rise:.1f} K")
        else:
            lines.append(f"base rise {self.base_rise:.1f} K, El = {self.elenbaas:.1f}")
        lines.append(f"h = {self.h:.1f} W/(m²·K), fin efficiency {self.fin_efficiency:.3f}")
        lines.append(f"R_th = {self.r_th:.4f} K/W")
        return "\n".join(lines + [f"note: {n}" for n in self.notes])


def fin_count_sweep(sink, n_values, mode="forced", **kwargs):
    """Evaluate the sink for each fin count. kwargs go to forced, operating_point or natural."""
    out = []
    for n in n_values:
        try:
            trial = replace(sink, n_fins=int(n))
        except ValueError:
            continue
        if mode == "forced":
            r = trial.forced(**kwargs)
        elif mode == "fan":
            try:
                r = trial.operating_point(**kwargs)
            except ValueError:
                continue
        elif mode == "natural":
            try:
                r = trial.natural(**kwargs)
            except ValueError:
                continue
        else:
            raise ValueError("mode must be forced, fan or natural")
        out.append(r)
    return out
