"""Fluid properties from CoolProp."""

from dataclasses import dataclass

from CoolProp.CoolProp import PropsSI

COOLANTS = {
    "water": "Water",
    "air": "Air",
    "ethylene glycol": "INCOMP::MEG",
    "propylene glycol": "INCOMP::MPG",
}


@dataclass(frozen=True)
class Fluid:
    rho: float   # kg/m³
    cp: float    # J/(kg·K)
    k: float     # W/(m·K)
    mu: float    # Pa·s
    beta: float  # 1/K, only meaningful for air

    @property
    def nu(self):
        return self.mu / self.rho

    @property
    def alpha(self):
        return self.k / (self.rho * self.cp)

    @property
    def pr(self):
        return self.mu * self.cp / self.k


def fluid(name, t_c, p=101325.0, fraction=None):
    """Properties at t_c (°C).

    name is one of COOLANTS. Glycols need a mass fraction (0 to 0.6).
    Water is held above saturation pressure so it stays liquid.
    """
    key = name.lower()
    if key not in COOLANTS:
        raise ValueError(f"unknown fluid {name!r}, pick from {list(COOLANTS)}")
    t = t_c + 273.15
    ref = COOLANTS[key]
    if ref.startswith("INCOMP"):
        if fraction is None:
            raise ValueError("glycol needs a mass fraction")
        ref = f"{ref}[{fraction}]"
    if key == "water":
        p = max(p, 1.05 * PropsSI("P", "T", t, "Q", 0, "Water"))

    rho, cp, k, mu = (PropsSI(q, "T", t, "P", p, ref) for q in ("D", "C", "L", "V"))
    beta = 1.0 / t if key == "air" else 0.0
    return Fluid(rho, cp, k, mu, beta)
