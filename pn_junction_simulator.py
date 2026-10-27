"""PN Junction Physics Simulator
================================

Educational, interactive model of a one-dimensional abrupt silicon p-n junction.

Implemented features
-------------------
* Interactive sliders for acceptor doping N_A, donor doping N_D, temperature T,
  and applied bias V_a.
* Temperature-dependent silicon band gap using the Varshni relation.
* Approximate intrinsic carrier concentration n_i(T), scaled from a stated
  room-temperature reference value.
* Ideal non-degenerate built-in potential.
* Depletion width and p-side/n-side depletion boundaries using the depletion
  approximation and charge neutrality.
* Piecewise depletion-region charge density, electric field, and electrostatic
  potential profiles.
* Approximate equilibrium majority/minority carrier concentrations in the
  neutral p- and n-type regions.
* Approximate Fermi-level offsets relative to the intrinsic level in the
  neutral regions.
* Five startup consistency checks and a live numerical readout.
* Reset button and informative handling of invalid bias/model conditions.

Model assumptions and limitations
---------------------------------
This is an educational electrostatic model, not a full semiconductor-device
solver. It assumes a one-dimensional abrupt junction, complete dopant
ionisation, non-degenerate statistics, the depletion approximation, and a
quasi-static applied bias. The simple depletion-region voltage is
V_dep = V_bi - V_a.

The n_i(T) model uses the Varshni band-gap relation and T^(3/2) scaling,
normalised to an approximate n_i(300 K) = 1e10 cm^-3. Its temperature outputs
are best treated as trends, not precision material data. The simple majority-
carrier-equals-doping approximation becomes unreliable when n_i is not small
compared with the doping concentration; the live readout flags this regime.

The simulator does NOT implement an I-V model, experimental calibration,
minority-carrier injection, drift-diffusion transport, recombination, series
resistance, tunnelling, avalanche breakdown, interface traps, high-level
injection, incomplete ionisation, or band-gap narrowing. The potential profile
is obtained by numerically integrating the depletion-approximation electric
field; it is not a self-consistent numerical solution of Poisson's equation.

Dependencies
------------
    numpy
    matplotlib

Run
---
    python pn_junction_simulator.py
"""

from __future__ import annotations

import math
import sys
from typing import Any

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button


# ---------------------------------------------------------------------------
# Physical constants and approximate silicon parameters
# ---------------------------------------------------------------------------

Q = 1.602176634e-19             # elementary charge, C
K_B = 1.380649e-23              # Boltzmann constant, J/K
K_B_EV = 8.617333262e-5         # Boltzmann constant, eV/K
EPS0 = 8.8541878128e-12         # vacuum permittivity, F/m
EPS_R_SI = 11.7                 # relative permittivity of silicon (approx.)
EPS_SI = EPS_R_SI * EPS0        # permittivity of silicon, F/m

# Varshni band-gap parameters for silicon (approximate).
EG0_EV = 1.17                   # extrapolated band gap at 0 K, eV
VARSHNI_ALPHA = 4.73e-4         # eV/K
VARSHNI_BETA = 636.0             # K

# The intrinsic concentration reference is intentionally approximate.
# 1e16 m^-3 = 1e10 cm^-3.
NI_300_REFERENCE_M3 = 1.00e16

# Interactive-control ranges.
DOPING_MIN_CM3 = 1.0e14
DOPING_MAX_CM3 = 1.0e18
TEMPERATURE_MIN_K = 200.0
TEMPERATURE_MAX_K = 450.0
BIAS_MIN_V = -2.0
BIAS_MAX_V = 0.65


def bandgap_si_eV(T_K: float) -> float:
    """Return the approximate silicon band gap E_g(T), in eV."""
    if not (1.0 <= T_K <= 600.0):
        raise ValueError("Temperature must be between 1 K and 600 K.")
    return EG0_EV - VARSHNI_ALPHA * T_K**2 / (T_K + VARSHNI_BETA)


def intrinsic_concentration_si(T_K: float) -> float:
    """Return approximate n_i(T) in m^-3.

    The model uses n_i(T) proportional to T^(3/2) exp[-E_g(T)/(2 k_B T)] and
    normalises it to NI_300_REFERENCE_M3 at 300 K. The normalisation is an
    educational reference, not a precision silicon-material data source.
    """
    if not (1.0 <= T_K <= 600.0):
        raise ValueError("Temperature must be between 1 K and 600 K.")

    eg_t = bandgap_si_eV(T_K)
    eg_300 = bandgap_si_eV(300.0)
    power_factor = (T_K / 300.0) ** 1.5
    exponent = -eg_t / (2.0 * K_B_EV * T_K) + eg_300 / (2.0 * K_B_EV * 300.0)
    # Clamp only to prevent floating-point overflow at extreme test inputs.
    exponent = max(-700.0, min(700.0, exponent))
    return NI_300_REFERENCE_M3 * power_factor * math.exp(exponent)


def built_in_potential_V(T_K: float, NA_m3: float, ND_m3: float, ni_m3: float) -> float:
    """Ideal non-degenerate built-in potential V_bi, in volts."""
    if min(T_K, NA_m3, ND_m3, ni_m3) <= 0:
        raise ValueError("Temperature, doping, and intrinsic concentration must be positive.")
    return (K_B * T_K / Q) * math.log((NA_m3 * ND_m3) / (ni_m3**2))


def fermi_level_offsets_eV(
    T_K: float, NA_m3: float, ND_m3: float, ni_m3: float
) -> tuple[float, float]:
    """Approximate neutral-region Fermi-level offsets from E_i, in eV.

    Returns (E_i - E_F on p-side, E_F - E_i on n-side), using non-degenerate
    expressions. Values are interpreted as positive offsets only in the usual
    extrinsic regime where the relevant doping exceeds n_i.
    """
    if min(T_K, NA_m3, ND_m3, ni_m3) <= 0:
        raise ValueError("Temperature, doping, and intrinsic concentration must be positive.")
    kT_eV = K_B_EV * T_K
    p_offset = kT_eV * math.log(NA_m3 / ni_m3)
    n_offset = kT_eV * math.log(ND_m3 / ni_m3)
    return p_offset, n_offset


def equilibrium_carriers_m3(
    doping_m3: float, ni_m3: float, region: str
) -> tuple[float, float]:
    """Approximate equilibrium carrier concentrations (n, p), in m^-3.

    Uses local charge neutrality and n*p = n_i^2. This applies to the neutral
    regions only; it does not model spatial carrier profiles in depletion.
    """
    if doping_m3 <= 0 or ni_m3 <= 0:
        raise ValueError("Doping and intrinsic concentration must be positive.")

    majority = 0.5 * (doping_m3 + math.sqrt(doping_m3**2 + 4.0 * ni_m3**2))
    minority = ni_m3**2 / majority
    region = region.lower()
    if region == "p":
        return minority, majority  # electron concentration n, hole concentration p
    if region == "n":
        return majority, minority
    raise ValueError("region must be 'p' or 'n'.")


def depletion_solution(
    T_K: float, NA_m3: float, ND_m3: float, Va_V: float, ni_m3: float,
    samples_per_side: int = 401,
) -> dict[str, Any]:
    """Solve the abrupt-junction depletion approximation for the chosen inputs.

    Bias convention: V_a > 0 is forward bias; V_a < 0 is reverse bias.
    The effective depletion-region voltage is V_dep = V_bi - V_a. If this is
    non-positive, this simplified depletion model is outside its valid range.

    Coordinates use x=0 at the metallurgical junction. The depletion interval
    is [-x_p, x_n]. The charge density is -q*N_A on the p-side and +q*N_D on
    the n-side. E = -d(phi)/dx and phi(-x_p) is set to zero as a reference.
    """
    if min(T_K, NA_m3, ND_m3, ni_m3) <= 0:
        raise ValueError("Temperature, doping, and intrinsic concentration must be positive.")
    if samples_per_side < 10:
        raise ValueError("Use at least 10 samples per side for a stable profile.")

    v_bi = built_in_potential_V(T_K, NA_m3, ND_m3, ni_m3)
    v_dep = v_bi - Va_V
    if v_dep <= 0:
        raise ValueError(
            "Applied forward bias is at or above the model's built-in potential. "
            "Reduce forward bias for this depletion-approximation simulator."
        )

    # Abrupt-junction depletion width.
    width = math.sqrt(
        (2.0 * EPS_SI / Q) * (1.0 / NA_m3 + 1.0 / ND_m3) * v_dep
    )

    # Charge neutrality: N_A*x_p = N_D*x_n and W=x_p+x_n.
    x_p = width * ND_m3 / (NA_m3 + ND_m3)
    x_n = width * NA_m3 / (NA_m3 + ND_m3)

    # Separate grids guarantee that x=0 is represented exactly.
    x_p_side = np.linspace(-x_p, 0.0, samples_per_side, endpoint=True)
    x_n_side = np.linspace(0.0, x_n, samples_per_side, endpoint=True)
    x = np.concatenate((x_p_side, x_n_side[1:]))

    rho = np.where(x < 0.0, -Q * NA_m3, Q * ND_m3)
    # Electric field is continuous at the junction and zero at both depletion edges.
    electric_field = np.empty_like(x)
    p_mask = x <= 0.0
    electric_field[p_mask] = -(Q * NA_m3 / EPS_SI) * (x[p_mask] + x_p)
    electric_field[~p_mask] = (Q * ND_m3 / EPS_SI) * (x[~p_mask] - x_n)

    # Potential profile from E = -d(phi)/dx, using trapezoidal integration.
    potential = np.zeros_like(x)
    dx = np.diff(x)
    potential[1:] = -np.cumsum(0.5 * (electric_field[1:] + electric_field[:-1]) * dx)
    numerical_drop = float(potential[-1] - potential[0])

    return {
        "V_bi": v_bi,
        "V_dep": v_dep,
        "W": width,
        "x_p": x_p,
        "x_n": x_n,
        "x": x,
        "rho": rho,
        "E": electric_field,
        "phi": potential,
        "numerical_drop": numerical_drop,
        "E_peak_magnitude": abs(-Q * NA_m3 * x_p / EPS_SI),
    }


def format_concentration(value_m3: float) -> str:
    """Format a concentration in the commonly used cm^-3 unit."""
    return f"{value_m3 / 1e6:.3g} cm⁻³"


def run_self_checks() -> list[str]:
    """Run five grouped internal consistency checks on a reference junction."""
    checks: list[str] = []
    temperature = 300.0
    doping = 1.0e22  # 1e16 cm^-3 expressed in m^-3
    ni = intrinsic_concentration_si(temperature)
    equilibrium = depletion_solution(temperature, doping, doping, 0.0, ni)

    # Check 1: positive width and positive boundaries.
    assert equilibrium["W"] > 0.0 and equilibrium["x_p"] > 0.0 and equilibrium["x_n"] > 0.0
    checks.append("PASS 1/5: depletion width and both depletion-side widths are positive.")

    # Check 2: charge neutrality.
    charge_error = abs(doping * equilibrium["x_p"] - doping * equilibrium["x_n"])
    charge_scale = max(doping * equilibrium["x_p"], 1.0)
    assert charge_error / charge_scale < 1e-10
    checks.append("PASS 2/5: charge neutrality N_A*x_p = N_D*x_n holds.")

    # Check 3: numerically integrated field reproduces the analytic drop.
    relative_drop_error = abs(equilibrium["numerical_drop"] - equilibrium["V_dep"]) / equilibrium["V_dep"]
    assert relative_drop_error < 0.01
    checks.append("PASS 3/5: integrating the electric field reproduces V_dep within 1%.")

    # Check 4: expected qualitative bias trends at fixed T and doping.
    forward = depletion_solution(temperature, doping, doping, 0.2, ni)
    reverse = depletion_solution(temperature, doping, doping, -2.0, ni)
    assert forward["W"] < equilibrium["W"] and reverse["W"] > equilibrium["W"]
    checks.append("PASS 4/5: forward bias narrows and reverse bias widens the depletion region.")

    # Check 5: intrinsic concentration increases with temperature in this model.
    assert intrinsic_concentration_si(350.0) > intrinsic_concentration_si(250.0)
    checks.append("PASS 5/5: approximate intrinsic carrier concentration rises with temperature.")
    return checks


def main() -> None:
    """Launch the interactive Matplotlib simulator."""
    # Default values chosen for a conventional room-temperature silicon junction.
    defaults = {"NA": 1.0e16, "ND": 1.0e16, "T": 300.0, "Va": 0.0}

    fig, axes = plt.subplots(4, 1, figsize=(11.5, 12.6))
    fig.canvas.manager.set_window_title("PN Junction Physics Simulator")
    plt.subplots_adjust(left=0.12, right=0.96, top=0.88, bottom=0.31, hspace=0.62)
    ax_region, ax_rho, ax_field, ax_potential = axes
    readout = fig.text(0.12, 0.975, "", fontsize=8.4, va="top", family="monospace")

    slider_color = "#d9e6f2"
    slider_specs = [
        ("NA", "Acceptor N_A (cm⁻³)", 1.0e14, 1.0e18, defaults["NA"], "%.2e"),
        ("ND", "Donor N_D (cm⁻³)", 1.0e14, 1.0e18, defaults["ND"], "%.2e"),
        ("T", "Temperature (K)", TEMPERATURE_MIN_K, TEMPERATURE_MAX_K, defaults["T"], "%.1f"),
        ("Va", "Applied bias V_a (V)", BIAS_MIN_V, BIAS_MAX_V, defaults["Va"], "%.3f"),
    ]
    slider_objects: dict[str, Slider] = {}
    y_positions = [0.245, 0.205, 0.165, 0.125]
    for (key, label, low, high, initial, fmt), y in zip(slider_specs, y_positions):
        slider_axis = fig.add_axes([0.20, y, 0.66, 0.024], facecolor=slider_color)
        slider_objects[key] = Slider(
            slider_axis, label, low, high, valinit=initial, valfmt=fmt
        )

    reset_axis = fig.add_axes([0.87, 0.055, 0.09, 0.04])
    reset_button = Button(reset_axis, "Reset")

    def update(_value: float | None = None) -> None:
        temperature = float(slider_objects["T"].val)
        na_cm3 = float(slider_objects["NA"].val)
        nd_cm3 = float(slider_objects["ND"].val)
        applied_bias = float(slider_objects["Va"].val)
        na_m3 = na_cm3 * 1e6  # cm^-3 -> m^-3
        nd_m3 = nd_cm3 * 1e6
        ni_m3 = intrinsic_concentration_si(temperature)
        eg = bandgap_si_eV(temperature)
        p_offset, n_offset = fermi_level_offsets_eV(temperature, na_m3, nd_m3, ni_m3)
        n_p, p_p = equilibrium_carriers_m3(na_m3, ni_m3, "p")
        n_n, p_n = equilibrium_carriers_m3(nd_m3, ni_m3, "n")

        # Clear all panels before drawing the latest state.
        for axis in axes:
            axis.clear()
            axis.set_axis_on()

        try:
            solution = depletion_solution(
                temperature, na_m3, nd_m3, applied_bias, ni_m3
            )
            x_nm = solution["x"] * 1e9
            xp_nm = solution["x_p"] * 1e9
            xn_nm = solution["x_n"] * 1e9

            # Panel 1: depletion region and junction location.
            ax_region.axvspan(-xp_nm, 0.0, alpha=0.25, label="p-side depletion")
            ax_region.axvspan(0.0, xn_nm, alpha=0.25, label="n-side depletion")
            ax_region.axvline(0.0, linestyle="--", linewidth=1.0, label="junction x=0")
            ax_region.set_title("Depletion region (schematic; depletion approximation)")
            ax_region.set_xlabel("Position relative to junction (nm)")
            ax_region.set_ylabel("Region")
            ax_region.set_yticks([])
            ax_region.set_xlim(-max(xp_nm, 1.0), max(xn_nm, 1.0))
            ax_region.legend(loc="upper right", fontsize=8, ncols=3)
            ax_region.grid(True, alpha=0.25)

            # Panel 2: fixed ionised-dopant charge in the depletion approximation.
            ax_rho.plot(x_nm, solution["rho"] / 1e6)
            ax_rho.axvline(0.0, linestyle="--", linewidth=1.0)
            ax_rho.set_title("Depletion-region charge density")
            ax_rho.set_xlabel("Position relative to junction (nm)")
            ax_rho.set_ylabel("Charge density (C/cm³)")
            ax_rho.grid(True, alpha=0.25)

            # Panel 3: electric field; negative sign follows the chosen x-axis.
            ax_field.plot(x_nm, solution["E"] / 1e5)
            ax_field.axvline(0.0, linestyle="--", linewidth=1.0)
            ax_field.set_title("Electric field across the depletion region")
            ax_field.set_xlabel("Position relative to junction (nm)")
            ax_field.set_ylabel("Electric field (×10⁵ V/m)")
            ax_field.grid(True, alpha=0.25)

            # Panel 4: electrostatic potential referenced to zero at -x_p.
            ax_potential.plot(x_nm, solution["phi"])
            ax_potential.axvline(0.0, linestyle="--", linewidth=1.0)
            ax_potential.set_title("Electrostatic potential (reference: p-side depletion edge)")
            ax_potential.set_xlabel("Position relative to junction (nm)")
            ax_potential.set_ylabel("Potential (V)")
            ax_potential.grid(True, alpha=0.25)

            # This warning makes the assumptions visible in near-intrinsic conditions.
            min_doping = min(na_m3, nd_m3)
            if ni_m3 >= 0.1 * min_doping:
                validity = "WARNING: n_i is not small vs. the lower doping; extrinsic approximation is unreliable."
            else:
                validity = "Model regime: depletion approximation; no experimental validation."

            readout.set_text(
                f"V_bi={solution['V_bi']:.3f} V | V_dep=V_bi−V_a={solution['V_dep']:.3f} V | "
                f"W={solution['W']*1e9:.2f} nm | x_p={xp_nm:.2f} nm | x_n={xn_nm:.2f} nm\n"
                f"n_i(T)≈{format_concentration(ni_m3)} | E_g(T)≈{eg:.3f} eV | "
                f"E_i−E_F (p)≈{p_offset:.3f} eV | E_F−E_i (n)≈{n_offset:.3f} eV\n"
                f"Neutral-region estimates: n_p={format_concentration(n_p)}, p_p={format_concentration(p_p)}; "
                f"n_n={format_concentration(n_n)}, p_n={format_concentration(p_n)}\n"
                f"Numerical Δφ={solution['numerical_drop']:.3f} V | {validity}"
            )
        except ValueError as error:
            for axis in axes:
                axis.text(
                    0.5, 0.5, str(error), transform=axis.transAxes,
                    ha="center", va="center", wrap=True
                )
                axis.set_axis_off()
            readout.set_text(
                "Outside the valid range of the simplified depletion model. "
                "Reduce forward bias or choose another temperature/doping combination."
            )

        fig.canvas.draw_idle()

    def reset(_event: Any) -> None:
        for slider in slider_objects.values():
            slider.reset()

    for slider in slider_objects.values():
        slider.on_changed(update)
    reset_button.on_clicked(reset)

    update()

    print("PN Junction Physics Simulator — internal consistency checks")
    try:
        for message in run_self_checks():
            print(message)
    except AssertionError as error:
        print(f"CHECK FAILED: {error}", file=sys.stderr)

    print("\nModel notes:")
    print("- Bias convention: positive = forward bias; negative = reverse bias.")
    print("- 1D abrupt silicon junction; depletion approximation and quasi-static bias.")
    print("- n_i(T) is an approximate trend model normalised to 1e10 cm^-3 at 300 K.")
    print("- Neutral-region carrier concentrations and Fermi offsets are approximate.")
    print("- No I-V model, experimental calibration, or full drift-diffusion solution is implemented.")
    print("- Treat numerical outputs as educational estimates, not precision device data.")

    plt.show()


if __name__ == "__main__":
    main()