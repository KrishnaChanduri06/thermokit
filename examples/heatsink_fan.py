"""Pick a fin count for a fan-cooled 50 x 50 mm sink, then check the junction.

Fan curve values are illustrative, use the real datasheet curve.
"""

import numpy as np

from thermokit import Contact, Fixed, PlateFinHeatSink, fin_count_sweep, solve

mm = 1e-3
CFM = 1 / 2118.88

sink = PlateFinHeatSink(width=50 * mm, length=50 * mm, fin_height=25 * mm, fin_thickness=1 * mm,
                        n_fins=12, base_thickness=4 * mm, k=200)
fan_flow = np.array([0, 4.2, 8.5, 12.7, 17.0]) * CFM
fan_dp = np.array([60, 48, 33, 16, 0.0])

print("fins   R_sink    Δp     flow")
for r in fin_count_sweep(sink, range(6, 31, 2), mode="fan", fan_flow=fan_flow, fan_dp=fan_dp):
    print(f"{r.sink.n_fins:>4}  {r.r_th:6.3f}  {r.dp:5.1f} Pa  {r.flow / CFM:5.2f} CFM")

best = min(fin_count_sweep(sink, range(6, 31), mode="fan", fan_flow=fan_flow, fan_dp=fan_dp),
           key=lambda r: r.r_th)
print("\n" + best.summary())

# 15 x 15 mm package, 30 W, 40 °C inside the box
die = 15 * mm * 15 * mm
layers = [Fixed("junction-case", 0.4), Contact("thermal pad", 0.5e-4, die)]
layers += best.sink.layers(die, best)
print("\n" + solve(layers, power=30.0, ambient=40.0).table())
