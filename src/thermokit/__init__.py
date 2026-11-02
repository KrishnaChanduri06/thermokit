from .coldplate import ColdPlate, ColdPlateResult
from .duct import rectangular_duct
from .heatsink import HeatSinkResult, PlateFinHeatSink, fin_count_sweep, teertstra_nu
from .props import fluid
from .spreading import equivalent_radius, spreading_resistance
from .stack import Contact, Convection, Fixed, Slab, Spreader, StackResult, solve

__version__ = "0.1.0"
