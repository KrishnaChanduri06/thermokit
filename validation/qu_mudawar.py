"""Cold plate model against the Qu & Mudawar (2002) experiments.

Qu, W. and Mudawar, I., 2002. Experimental and numerical study of pressure
drop and heat transfer in a single-phase micro-channel heat sink.
Int. J. Heat Mass Transfer 45, 2549-2565.

Data are the paper's Figs. 4a and 6, digitised in
github.com/KrishnaChanduri06/microchannel-cold-plate-cht.
Geometry from the paper's Table 1. Run from the repo root:

    python validation/qu_mudawar.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from thermokit import ColdPlate

HERE = Path(__file__).parent
FIG = HERE / "figures"
UM = 1e-6

plate = ColdPlate(
    width=21 * 467 * UM,
    length=44.764e-3,
    channel_width=231 * UM,
    channel_height=713 * UM,
    fin_width=236 * UM,
    base_thickness=5637 * UM,
    k=401.0,
    lid="adiabatic",          # polycarbonate cover
    n_channels=21,
)
TC_DEPTH = (5637 - 3175) * UM  # thermocouple plane below the channel floor
AREA = 1.0e-2 * 4.48e-2        # planform area used for q''_eff
T_IN = 15.0
K_PLENUM = 2.0                 # two velocity heads, upper estimate (plenums not given)


def load(name):
    return np.genfromtxt(HERE / "data" / name, delimiter=",", names=True)


MODELS = {
    "full model": dict(developing=True, wall_viscosity=True),
    "entry, constant props": dict(developing=True, wall_viscosity=False),
    "fully developed Nu": dict(developing=False, wall_viscosity=False),
}


def thermocouples():
    """Columns: q, location, Re, measured ΔT, then one column per model."""
    d = load("fig6_tc_vs_re.csv")
    rows = []
    for q, loc, x_tc, re, dt_exp in zip(d["q_eff_Wcm2"], d["location"], d["x_tc_mm"], d["Re"], d["dT_th_in_C"]):
        flow = plate.flow_for_re(re, T_IN)
        power = q * 1e4 * AREA
        out = []
        for opts in MODELS.values():
            r = plate.solve(flow, power, T_IN, **opts)
            out.append(np.interp(x_tc * 1e-3, r.x, r.t_at_depth(TC_DEPTH)) - T_IN)
        rows.append((q, loc, re, dt_exp, *out))
    return np.array(rows)


def pressure_drop():
    """Columns: q, Re, measured Δp, channel + plenum, channel only, constant props."""
    d = load("fig4a_dp_vs_re.csv")
    rows = []
    for q, re, dp_exp in zip(d["q_eff_Wcm2"], d["Re"], d["dP_bar"]):
        flow = plate.flow_for_re(re, T_IN)
        power = q * 1e4 * AREA
        rows.append((
            q, re, dp_exp * 1e5,
            plate.solve(flow, power, T_IN, k_minor=K_PLENUM).dp,
            plate.solve(flow, power, T_IN, k_minor=0.0).dp,
            plate.solve(flow, power, T_IN, k_minor=K_PLENUM, wall_viscosity=False).dp,
        ))
    return np.array(rows)


def stats(model, exp):
    err = (model - exp) / exp * 100
    return np.mean(np.abs(err)), err.min(), err.max()


def main():
    FIG.mkdir(exist_ok=True)
    tc = thermocouples()
    dp = pressure_drop()

    print(f"Thermocouple temperature rise T_tc - T_in, {len(tc)} points")
    for col, label in enumerate(MODELS, start=4):
        mae, lo, hi = stats(tc[:, col], tc[:, 3])
        print(f"  {label:<24} mean |error| {mae:5.1f} %, range {lo:+6.1f} to {hi:+6.1f} %")
    for loc in (1, 2, 3, 4):
        sel = tc[:, 1] == loc
        mae, lo, hi = stats(tc[sel, 4], tc[sel, 3])
        print(f"    full model, TC{loc}: mean |error| {mae:4.1f} %, range {lo:+5.1f} to {hi:+5.1f} %")

    print(f"Pressure drop, {len(dp)} points")
    for col, label in ((3, f"channel + {K_PLENUM:g} vel. heads"), (4, "channel only"),
                       (5, "constant props + plenum")):
        mae, lo, hi = stats(dp[:, col], dp[:, 2])
        print(f"  {label:<24} mean |error| {mae:5.1f} %, range {lo:+6.1f} to {hi:+6.1f} %")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), constrained_layout=True)
    ax = axes[0]
    lim = [0, 1.05 * tc[:, 3:].max()]
    ax.plot(lim, lim, color="0.6", lw=1)
    ax.fill_between(lim, [0.9 * v for v in lim], [1.1 * v for v in lim], color="0.92", lw=0, label="±10 %")
    ax.scatter(tc[:, 3], tc[:, 6], s=14, marker="s", facecolors="none", edgecolors="#b5651d",
               label="fully developed Nu (typical hand calc)")
    ax.scatter(tc[:, 3], tc[:, 4], s=14, color="#1f4e79", label="thermokit")
    ax.set(xlim=lim, ylim=lim, xlabel="measured $T_{tc} - T_{in}$ (K)", ylabel="predicted (K)",
           title="Thermocouples, 4 locations, 100 and 200 W/cm²")
    ax.legend(frameon=False, loc="upper left")

    ax = axes[1]
    for q, mk, color in ((100, "o", "#1f4e79"), (200, "^", "#6a9fd4")):
        sel = dp[:, 0] == q
        ax.plot(dp[sel, 1], dp[sel, 2] / 1e3, mk, mfc="none", color="0.2", label=f"experiment, {q} W/cm²")
        ax.plot(dp[sel, 1], dp[sel, 4] / 1e3, "-", color=color, label=f"channel only, {q} W/cm²")
        ax.plot(dp[sel, 1], dp[sel, 3] / 1e3, "--", lw=1, color=color,
                label=f"+ {K_PLENUM:g} velocity heads, {q} W/cm²")
    ax.set(xlabel="Re (inlet)", ylabel="Δp (kPa)", title="Pressure drop")
    ax.legend(frameon=False, fontsize=8)
    fig.savefig(FIG / "qu_mudawar.png", dpi=160)
    print("figure:", FIG / "qu_mudawar.png")


if __name__ == "__main__":
    main()
