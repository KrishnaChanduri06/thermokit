"""Air heat sink model against Zaghlol et al. (2002), all-copper sink.

Zaghlol, A., Hermann, K., Butler, J., Teertstra, P. and Culham, R., 2002.
Forced convection heat transfer for swaged mixed metal heatsinks.
Proc. ITherm 2002. Geometry from their Table 1 (Cu base, Cu fins).

Data: the red Cu B - Cu F markers of Figs. 7 and 9, read from the vector
paths in the paper's PDF and calibrated on the axis tick labels. The Re_S
values agree with the paper's Table 2 to within ±4.
    data/zaghlol_fig7_cucu.csv   Re_S, R_KW    (R from the mean of ten base thermocouples)
    data/zaghlol_fig9_cucu.csv   Re_S, dp_Pa

Re_S = V_S · 2S / ν, the paper's Eq. 8.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from thermokit import PlateFinHeatSink, fluid

HERE = Path(__file__).parent
mm = 1e-3

sink = PlateFinHeatSink(width=153.7 * mm, length=253.6 * mm, fin_height=49 * mm, fin_thickness=1.33 * mm,
                        n_fins=45, base_thickness=12.7 * mm, k=390, density=8930)


def model(re_s, t_amb=25.0):
    air = fluid("air", t_amb)
    v = re_s * air.nu / (2 * sink.spacing)
    flow = v * sink.n_channels * sink.spacing * sink.fin_height
    r = sink.forced(flow, t_amb)
    # measured R uses the mean of ten base thermocouples: isothermal-base sink plus 1D base conduction
    return r.r_th + sink.base_thickness / (sink.k * sink.base_area), r.dp


def fully_developed_dp(re_s, t_amb=25.0):
    """Channel friction only, f·Re = 24 for parallel plates on D_h = 2S."""
    air = fluid("air", t_amb)
    v = re_s * air.nu / (2 * sink.spacing)
    return 4 * 24 / re_s * sink.length / (2 * sink.spacing) * 0.5 * air.rho * v**2


def main():
    print(f"gap {sink.spacing * 1e3:.2f} mm (paper: 2.1 mm)")
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8), constrained_layout=True)
    for ax, (name, col, idx, unit, label) in zip(axes, (
            ("zaghlol_fig7_cucu.csv", "R_KW", 0, "K/W", r"$R_\theta$"),
            ("zaghlol_fig9_cucu.csv", "dp_Pa", 1, "Pa", r"$\Delta p$"))):
        d = np.genfromtxt(HERE / "data" / name, delimiter=",", names=True)
        pred = np.array([model(re)[idx] for re in d["Re_S"]])
        err = (pred - d[col]) / d[col] * 100
        print(f"{col}: mean |error| {np.mean(np.abs(err)):.1f} %, range {err.min():+.1f} to {err.max():+.1f} %")
        for re, m, p in zip(d["Re_S"], d[col], pred):
            print(f"  Re_S {re:6.0f}  measured {m:8.4g} {unit}  model {p:8.4g}")

        res = np.linspace(d["Re_S"].min(), d["Re_S"].max(), 60)
        ax.plot(d["Re_S"], d[col], "s", mfc="none", color="0.2", label="experiment, Cu base / Cu fins")
        ax.plot(res, [model(r)[idx] for r in res], color="#1f4e79", label="thermokit")
        if idx == 1:
            fd = np.array([fully_developed_dp(r) for r in d["Re_S"]])
            e = (fd - d[col]) / d[col] * 100
            print(f"  channel friction only: mean |error| {np.mean(np.abs(e)):.1f} %, "
                  f"range {e.min():+.1f} to {e.max():+.1f} %")
            ax.plot(res, [fully_developed_dp(r) for r in res], "--", lw=1, color="#6a9fd4",
                    label="fully developed friction only")
        ax.set(xlabel="$Re_S$", ylabel=f"{label} ({unit})")
        ax.legend(frameon=False)
    fig.suptitle("Plate-fin heat sink, Zaghlol et al. (2002)")
    fig.savefig(HERE / "figures" / "zaghlol_2002.png", dpi=160)
    print("figure:", HERE / "figures" / "zaghlol_2002.png")


if __name__ == "__main__":
    main()
