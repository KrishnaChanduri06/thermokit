"""Cold plate model against 40 conjugate CFD runs (Ansys Fluent).

The CFD is the design study in
github.com/KrishnaChanduri06/microchannel-cold-plate-cht: copper and
silicon plates, 1 cm x 4.48 cm, 713 µm deep channels, 1 mm base, 1 mm
bonded lid, water at 30 °C, uniform 100 or 200 W/cm² on the base.
Peak heater temperature and channel pressure drop are compared.

    python validation/fluent_design_sweep.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from thermokit import ColdPlate

HERE = Path(__file__).parent
UM = 1e-6
LENGTH = 44.764e-3
K = {"copper": 401.0, "silicon": 148.0}


def main():
    d = np.genfromtxt(HERE / "data" / "fluent_design_sweep.csv", delimiter=",", names=True,
                      dtype=None, encoding="utf-8")
    rows = []
    for c in d:
        pitch = (c["w_ch"] + c["w_fin"]) * UM
        plate = ColdPlate(width=c["n_ch"] * pitch, length=LENGTH, channel_width=c["w_ch"] * UM,
                          channel_height=713 * UM, fin_width=c["w_fin"] * UM, base_thickness=1e-3,
                          k=K[c["solid"]], lid="bonded", n_channels=c["n_ch"])
        power = c["q"] * 1e4 * plate.width * LENGTH
        r = plate.solve(c["flow"] / 1000, power, t_in=30.0, regime="laminar")  # CFD was run laminar
        rows.append((c["R_th"], r.r_th, c["dP_bar"] * 1e5, r.dp, c["oneD_R_th"], c["Re"]))
    a = np.array(rows)

    def report(label, model, ref):
        err = (model - ref) / ref * 100
        print(f"  {label:<28} mean |error| {np.mean(np.abs(err)):4.1f} %, range {err.min():+5.1f} to {err.max():+5.1f} %")

    print(f"{len(a)} Fluent runs, Re {a[:, 5].min():.0f} to {a[:, 5].max():.0f}")
    report("R_th,max  thermokit", a[:, 1], a[:, 0])
    report("R_th,max  1-D model in repo", a[:, 4], a[:, 0])
    report("Δp channel  thermokit", a[:, 3], a[:, 2])

    fig, axes = plt.subplots(1, 2, figsize=(9, 4), constrained_layout=True)
    for ax, (x, y, label, unit) in zip(axes, ((0, 1, "$R_{th,max}$", "mK/W"), (2, 3, r"$\Delta p$", "kPa"))):
        scale = 1e3 if unit == "mK/W" else 1e-3
        lim = [0, 1.05 * a[:, [x, y]].max() * scale]
        ax.plot(lim, lim, color="0.6", lw=1)
        ax.fill_between(lim, [0.9 * v for v in lim], [1.1 * v for v in lim], color="0.92", lw=0, label="±10 %")
        ax.scatter(a[:, x] * scale, a[:, y] * scale, s=16, color="#1f4e79", label="thermokit")
        ax.set(xlim=lim, ylim=lim, xlabel=f"Fluent {label} ({unit})", ylabel=f"thermokit ({unit})")
        ax.legend(frameon=False, loc="upper left")
    fig.suptitle("Cold plate design study, 40 conjugate CFD runs")
    out = HERE / "figures" / "fluent_design_sweep.png"
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=160)
    print("figure:", out)


if __name__ == "__main__":
    main()
