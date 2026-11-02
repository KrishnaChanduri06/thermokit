"""1D series thermal resistance stack, junction to ambient.

All inputs are SI: m, m², W/(m·K), W/(m²·K), K·m²/W, K/W, W, °C.
Layers are listed from the heat source outwards.
"""

from dataclasses import dataclass

import numpy as np

from .spreading import spreading_resistance


@dataclass
class Fixed:
    """A resistance taken straight from a datasheet, e.g. θ_jc."""

    name: str
    r: float

    def resistance(self, downstream):
        return self.r


@dataclass
class Slab:
    """Plane conduction through a layer of uniform cross-section."""

    name: str
    thickness: float
    k: float
    area: float

    def resistance(self, downstream):
        return self.thickness / (self.k * self.area)


@dataclass
class Contact:
    """Interface given as an area-specific resistance R'' (K·m²/W)."""

    name: str
    r_area: float
    area: float

    def resistance(self, downstream):
        return self.r_area / self.area


@dataclass
class Convection:
    """Film resistance 1/(hA)."""

    name: str
    h: float
    area: float

    def resistance(self, downstream):
        return 1.0 / (self.h * self.area)


@dataclass
class Spreader:
    """Plate fed by a smaller source, e.g. a die on a lid or a heat sink base.

    The bottom-side film coefficient is taken from everything downstream of
    the plate, h = 1/(R_down · A_p), so the spreading term responds to how
    well the plate is cooled. Returns spreading plus 1D conduction.
    """

    name: str
    source_area: float
    plate_area: float
    thickness: float
    k: float

    def resistance(self, downstream):
        h = np.inf if downstream <= 0 else 1.0 / (downstream * self.plate_area)
        r_sp = spreading_resistance(
            self.source_area, self.plate_area, self.thickness, self.k, h
        )
        return r_sp + self.thickness / (self.k * self.plate_area)


@dataclass
class StackResult:
    names: list
    resistances: np.ndarray
    power: float
    ambient: float

    @property
    def total(self):
        return self.resistances.sum()

    @property
    def temperatures(self):
        """Temperature at the hot face of each layer, then ambient (°C)."""
        downstream = np.cumsum(self.resistances[::-1])[::-1]
        return np.append(self.ambient + self.power * downstream, self.ambient)

    @property
    def junction(self):
        return self.temperatures[0]

    @property
    def drops(self):
        return self.power * self.resistances

    @property
    def shares(self):
        return self.resistances / self.total

    def table(self):
        rows = [f"{'layer':<18}{'R (K/W)':>10}{'ΔT (K)':>10}{'share':>8}"]
        for name, r, dt, s in zip(self.names, self.resistances, self.drops, self.shares):
            rows.append(f"{name:<18}{r:>10.4f}{dt:>10.2f}{s:>8.1%}")
        rows.append(f"{'total':<18}{self.total:>10.4f}{self.total * self.power:>10.2f}")
        rows.append(f"T_j = {self.junction:.1f} °C at {self.power:g} W, T_amb = {self.ambient:g} °C")
        return "\n".join(rows)


def solve(layers, power, ambient=25.0):
    """Evaluate the stack from ambient back to the source."""
    rs = []
    downstream = 0.0
    for layer in reversed(layers):
        r = layer.resistance(downstream)
        rs.append(r)
        downstream += r
    rs.reverse()
    return StackResult([l.name for l in layers], np.array(rs), power, ambient)
