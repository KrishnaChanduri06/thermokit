"""thermokit web demo.  Run:  streamlit run app/streamlit_app.py"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from thermokit import (
    ColdPlate, Contact, Fixed, PlateFinHeatSink, Slab, Spreader, fin_count_sweep, solve,
)

mm, um = 1e-3, 1e-6
CFM = 1 / 2118.88  # m³/s
BLUE, LIGHT, GREY = "#1f4e79", "#6a9fd4", "0.55"

st.set_page_config(page_title="thermokit", layout="wide")
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})


def figure(w=5.0, h=3.0):
    return plt.subplots(figsize=(w, h), constrained_layout=True)


def show_notes(notes):
    for n in notes:
        st.caption(f"Note: {n}")


# ---------------------------------------------------------------------------

def page_stack():
    st.header("Chip thermal stack")
    st.write("Junction to ambient, one layer at a time. See which layer is eating your temperature budget.")
    c1, c2 = st.columns([1, 1.4])
    with c1:
        power = st.number_input("Power (W)", 1.0, 2000.0, 125.0)
        t_amb = st.number_input("Ambient / inlet (°C)", -40.0, 80.0, 35.0)
        t_max = st.number_input("Junction limit (°C)", 50.0, 150.0, 95.0)
        st.subheader("Layers")
        r_jc = st.number_input("θ_jc from datasheet (K/W)", 0.0, 5.0, 0.10, format="%.3f")
        die = st.number_input("Die / lid footprint, square side (mm)", 2.0, 100.0, 35.0)
        tim = st.number_input("TIM R″ (K·cm²/W)", 0.0, 2.0, 0.10, format="%.3f")
        base_side = st.number_input("Heat sink base, square side (mm)", 5.0, 300.0, 60.0)
        base_t = st.number_input("Base thickness (mm)", 0.5, 30.0, 6.0)
        base_k = st.selectbox("Base material", ["copper (390)", "aluminium (200)", "vapour chamber (~2000)"])
        r_sink = st.number_input("Fins + air, from the heat sink page (K/W)", 0.001, 20.0, 0.18, format="%.3f")
    k = {"copper (390)": 390.0, "aluminium (200)": 200.0, "vapour chamber (~2000)": 2000.0}[base_k]
    a_die = (die * mm) ** 2
    a_base = (base_side * mm) ** 2
    if a_die > a_base:
        st.error("Die footprint is larger than the base.")
        return
    layers = [
        Fixed("junction-case", r_jc),
        Contact("TIM", tim * 1e-4, a_die),
        Spreader("base", a_die, a_base, base_t * mm, k),
        Fixed("fins + air", r_sink),
    ]
    res = solve(layers, power, t_amb)
    with c2:
        a, b, c = st.columns(3)
        a.metric("T_j", f"{res.junction:.1f} °C")
        b.metric("R_ja", f"{res.total:.3f} K/W")
        c.metric("Margin", f"{t_max - res.junction:+.1f} K")
        fig, ax = figure(5.5, 3.2)
        bottoms = res.temperatures[1:]
        ax.bar(res.names, res.drops, bottom=bottoms, color=[BLUE, LIGHT, LIGHT, BLUE])
        ax.axhline(t_max, color="#b5651d", lw=1, ls="--")
        ax.text(len(layers) - 0.5, t_max, " limit", va="bottom", ha="right", color="#b5651d")
        ax.set_ylabel("temperature (°C)")
        ax.set_ylim(t_amb, max(t_max, res.junction) * 1.05)
        st.pyplot(fig)
        st.code(res.table(), language=None)
        st.caption("Spreading in the base uses the exact series solution for a centred source "
                   "(Yovanovich et al. 1998), with square footprints mapped to circles of equal area.")


# ---------------------------------------------------------------------------

def page_heatsink():
    st.header("Air-cooled plate-fin heat sink")
    c1, c2 = st.columns([1, 1.4])
    with c1:
        mode = st.radio("Cooling", ["Fixed airflow", "Fan curve", "Natural convection"], horizontal=True)
        W = st.number_input("Base width, across fins (mm)", 10.0, 500.0, 50.0)
        L = st.number_input("Length, along flow (mm)", 10.0, 500.0, 50.0)
        H = st.number_input("Fin height (mm)", 2.0, 150.0, 25.0)
        t = st.number_input("Fin thickness (mm)", 0.2, 10.0, 1.0)
        n = st.number_input("Number of fins", 2, 200, 12)
        tb = st.number_input("Base thickness (mm)", 0.5, 30.0, 4.0)
        mat = st.selectbox("Material", ["aluminium 6063 (200)", "copper (390)"])
        power = st.number_input("Heat load (W)", 0.1, 2000.0, 30.0)
        t_amb = st.number_input("Air inlet (°C)", -40.0, 80.0, 25.0)
        if mode == "Fixed airflow":
            cfm = st.number_input("Airflow through the fins (CFM)", 0.1, 500.0, 6.6)
        elif mode == "Fan curve":
            st.caption("Fan curve as flow (CFM) : static pressure (Pa), one point per line, ending at free delivery.")
            txt = st.text_area("Fan curve", "0 : 60\n4.2 : 48\n8.5 : 33\n12.7 : 16\n17 : 0", height=130)
    k, rho = (390.0, 8930.0) if mat.startswith("copper") else (200.0, 2700.0)
    try:
        sink = PlateFinHeatSink(W * mm, L * mm, H * mm, t * mm, int(n), tb * mm, k, rho)
    except ValueError as e:
        st.error(str(e))
        return

    try:
        if mode == "Fixed airflow":
            res = sink.forced(cfm * CFM, t_amb, power)
            sweep = fin_count_sweep(sink, range(3, 80), "forced", flow=cfm * CFM, t_amb=t_amb)
        elif mode == "Fan curve":
            pts = [tuple(float(v) for v in line.split(":")) for line in txt.strip().splitlines() if ":" in line]
            fq = [p[0] * CFM for p in pts]
            fp = [p[1] for p in pts]
            res = sink.operating_point(fq, fp, t_amb, power)
            sweep = fin_count_sweep(sink, range(3, 80), "fan", fan_flow=fq, fan_dp=fp, t_amb=t_amb)
        else:
            res = sink.natural(power, t_amb)
            sweep = fin_count_sweep(sink, range(3, 60), "natural", power=power, t_amb=t_amb)
    except ValueError as e:
        st.error(str(e))
        return

    with c2:
        a, b, c, d = st.columns(4)
        a.metric("R_sink", f"{res.r_th:.3f} K/W")
        b.metric("Base temp", f"{t_amb + power * res.r_th:.1f} °C")
        if res.mode == "forced":
            c.metric("Δp", f"{res.dp:.1f} Pa")
            d.metric("Air rise", f"{res.air_rise:.1f} K")
        else:
            c.metric("Gap", f"{sink.spacing * 1e3:.2f} mm")
            d.metric("Best gap", f"{sink.optimum_spacing_natural(res.base_rise, t_amb) * 1e3:.2f} mm")
        st.caption(f"Mass ≈ {sink.mass * 1e3:.0f} g, fin efficiency {res.fin_efficiency:.2f}, "
                   f"h = {res.h:.1f} W/(m²·K)")

        g1, g2 = st.columns(2)
        with g1:
            fig, ax = figure(3.6, 2.8)
            ns = [r.sink.n_fins for r in sweep]
            ax.plot(ns, [r.r_th for r in sweep], color=BLUE)
            ax.plot([sink.n_fins], [res.r_th], "o", color="#b5651d")
            ax.set(xlabel="number of fins", ylabel="R_sink (K/W)", title="Fin count sweep")
            best = min(sweep, key=lambda r: r.r_th)
            ax.axvline(best.sink.n_fins, color=GREY, lw=0.8, ls=":")
            st.pyplot(fig)
            st.caption(f"Lowest R at {best.sink.n_fins} fins: {best.r_th:.3f} K/W")
        with g2:
            if res.mode == "forced":
                fig, ax = figure(3.6, 2.8)
                q = np.linspace(0.05, 1.0, 40) * max(res.flow * 2, 1e-4)
                ax.plot(q / CFM, [sink.pressure_drop(x, t_amb) for x in q], color=BLUE, label="heat sink")
                if mode == "Fan curve":
                    ax.plot(np.array(fq) / CFM, fp, color=GREY, label="fan")
                ax.plot([res.flow / CFM], [res.dp], "o", color="#b5651d")
                ax.set(xlabel="airflow (CFM)", ylabel="Δp (Pa)", title="System curve")
                ax.legend(frameon=False)
                st.pyplot(fig)
            else:
                fig, ax = figure(3.6, 2.8)
                line = []
                for p in np.linspace(0.2, 2.0, 20) * power:
                    try:
                        line.append((p, t_amb + sink.natural(p, t_amb).base_rise))
                    except ValueError:
                        break
                ax.plot(*zip(*line), color=BLUE)
                ax.set(xlabel="heat load (W)", ylabel="base temperature (°C)", title="Load line")
                st.pyplot(fig)
        show_notes(res.notes)
        st.caption("Forced: Teertstra et al. (1999), fully shrouded, no bypass; R within 3.3 % mean of the "
                   "Zaghlol et al. (2002) experiments. Δp: Culham & Muzychka (2001), conservative, it read up to "
                   "2× high on that test. "
                   "Natural: Bar-Cohen & Rohsenow (1984), vertical fins. Base taken as isothermal; "
                   "use the stack page for spreading from a small source.")


# ---------------------------------------------------------------------------

def page_coldplate():
    st.header("Liquid cold plate, straight channels")
    c1, c2 = st.columns([1, 1.4])
    with c1:
        W = st.number_input("Plate width, across channels (mm)", 1.0, 300.0, 10.0)
        L = st.number_input("Channel length (mm)", 1.0, 500.0, 44.8)
        wc = st.number_input("Channel width (µm)", 50.0, 10000.0, 231.0)
        hc = st.number_input("Channel depth (µm)", 50.0, 20000.0, 713.0)
        wf = st.number_input("Fin width (µm)", 50.0, 10000.0, 236.0)
        tb = st.number_input("Base under channels (mm)", 0.1, 20.0, 1.0)
        mat = st.selectbox("Material", ["copper (390)", "aluminium (200)", "silicon (148)"])
        lid = st.radio("Lid", ["bonded", "adiabatic"], horizontal=True,
                       help="bonded: brazed or diffusion-bonded metal lid. adiabatic: polymer cover.")
        coolant = st.selectbox("Coolant", ["water", "ethylene glycol", "propylene glycol"])
        frac = st.slider("Glycol mass fraction", 0.1, 0.6, 0.3) if coolant != "water" else None
        flow = st.number_input("Flow rate (L/min)", 0.01, 50.0, 0.5)
        power = st.number_input("Heat load (W)", 1.0, 5000.0, 448.0)
        t_in = st.number_input("Inlet temperature (°C)", 0.0, 80.0, 30.0)
        k_minor = st.number_input("Manifold losses (velocity heads)", 0.0, 20.0, 1.5)
    k = {"copper (390)": 390.0, "aluminium (200)": 200.0, "silicon (148)": 148.0}[mat]
    try:
        plate = ColdPlate(W * mm, L * mm, wc * um, hc * um, wf * um, tb * mm, k, lid)
        res = plate.solve(flow, power, t_in, coolant, frac, k_minor)
        fd = plate.solve(flow, power, t_in, coolant, frac, k_minor, developing=False, wall_viscosity=False)
    except ValueError as e:
        st.error(str(e))
        return

    with c2:
        a, b, c, d = st.columns(4)
        a.metric("R_th (peak)", f"{res.r_th * 1e3:.1f} mK/W")
        b.metric("T_base,max", f"{res.t_base_max:.1f} °C")
        c.metric("Δp", f"{res.dp / 1e3:.2f} kPa")
        d.metric("Pumping", f"{res.pumping_power:.3f} W")
        st.caption(f"{plate.n_channels} channels, Re = {res.re:.0f} ({res.regime}), u = {res.velocity:.2f} m/s, "
                   f"T_out = {res.t_out:.1f} °C. Fully developed hand calc would say "
                   f"{fd.r_th * 1e3:.1f} mK/W ({(fd.r_th / res.r_th - 1) * 100:+.0f} %).")
        g1, g2 = st.columns(2)
        with g1:
            fig, ax = figure(3.6, 2.8)
            ax.plot(res.x * 1e3, res.t_base, color=BLUE, label="base (heated face)")
            ax.plot(res.x * 1e3, res.t_wall, color=LIGHT, label="channel wall")
            ax.plot(res.x * 1e3, res.t_bulk, color=GREY, label="coolant")
            ax.set(xlabel="x (mm)", ylabel="°C", title="Along the channel")
            ax.legend(frameon=False)
            st.pyplot(fig)
        with g2:
            flows = np.geomspace(0.25, 3.0, 14) * flow
            rs = [plate.solve(f, power, t_in, coolant, frac, k_minor) for f in flows]
            fig, ax = figure(3.6, 2.8)
            ax.plot([r.pumping_power for r in rs], [r.r_th * 1e3 for r in rs], "-o", ms=3, color=BLUE)
            ax.plot([res.pumping_power], [res.r_th * 1e3], "o", color="#b5651d")
            ax.set_xscale("log")
            ax.set(xlabel="pumping power (W)", ylabel="R_th (mK/W)", title="Flow trade-off")
            st.pyplot(fig)
        show_notes(res.notes)
        st.caption("Laminar: thermal entry solved numerically for the actual channel shape, with a heated-liquid "
                   "viscosity correction. Checked against the Qu & Mudawar (2002) experiments (thermocouples 8.2 % "
                   "mean error, Δp 8.4 %) and 40 conjugate CFD runs (R_th 3.6 %, Δp 2.1 %). Turbulent: "
                   "Gnielinski, blended through transition. Base conduction is 1D, so a small hot spot on a "
                   "large plate needs the stack page for spreading.")


PAGES = {"Chip stack": page_stack, "Air heat sink": page_heatsink, "Liquid cold plate": page_coldplate}

st.sidebar.title("thermokit")
st.sidebar.caption("Hand-calc thermal design for electronics cooling. Answers in seconds, "
                   "before you spend a day on CFD.")
choice = st.sidebar.radio("Tool", list(PAGES))
PAGES[choice]()
st.sidebar.markdown("---")
st.sidebar.caption("Early design estimates only. Check final designs with CFD or test.")
