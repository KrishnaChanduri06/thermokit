"""Channel width vs flow rate for a 448 W copper microchannel cold plate.

Footprint 10 x 44.8 mm (100 W/cm²), 713 µm deep channels, 100 µm fins,
1 mm base, bonded lid, water at 30 °C.
"""

from thermokit import ColdPlate

um, mm = 1e-6, 1e-3

print("w_ch   flow     Re    R_th     Δp      pump   T_max")
for w_ch in (100, 150, 231, 350):
    plate = ColdPlate(width=10 * mm, length=44.8 * mm, channel_width=w_ch * um, channel_height=713 * um,
                      fin_width=100 * um, base_thickness=1 * mm, k=390, lid="bonded")
    for flow in (0.2, 0.5, 1.0):
        r = plate.solve(flow, power=448.0, t_in=30.0)
        print(f"{w_ch:>4}  {flow:4.1f} L/min {r.re:5.0f}  {r.r_th * 1e3:5.1f} mK/W "
              f"{r.dp / 1e3:6.1f} kPa {r.pumping_power:5.2f} W {r.t_base_max:5.1f} °C")

plate = ColdPlate(10 * mm, 44.8 * mm, 100 * um, 713 * um, 100 * um, 1 * mm, 390, "bonded")
print("\n" + plate.solve(0.5, 448.0, 30.0).summary())
fd = plate.solve(0.5, 448.0, 30.0, developing=False, wall_viscosity=False)
print(f"fully developed hand calc: {fd.r_th * 1e3:.1f} mK/W")
