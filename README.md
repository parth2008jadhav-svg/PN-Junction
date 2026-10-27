# PN Junction Physics Simulator

An interactive Python visualization of a one-dimensional abrupt silicon p–n junction, created to explore how doping, temperature, and applied bias affect the depletion region and its electrostatic properties.

## Features

- Interactive controls for acceptor doping (\(N_A\)) and donor doping (\(N_D\)), from \(10^{14}\) to \(10^{18}\,\mathrm{cm}^{-3}\).
- Temperature control from 200 K to 450 K.
- Applied-bias control from −2.0 V to +0.65 V.
- Numerical readout and plots for the depletion region, charge density, electric field, and electrostatic potential.
- Approximate temperature-dependent silicon band-gap and intrinsic-carrier-concentration models.
- Input/condition warnings and a built-in self-test with five internal consistency/trend checks.

## Requirements

- Python 3.10 or later recommended
- NumPy
- Matplotlib

Install dependencies:

```bash
python -m pip install numpy matplotlib
```

## Run

Download or clone this repository, then run:

```bash
python pn_junction_simulator.py
```

The simulator opens an interactive Matplotlib window. Adjust the sliders to explore the model. Use the self-test option in the program to run its built-in checks.

## Model and assumptions

The program uses a one-dimensional, abrupt silicon junction and the depletion approximation. It computes quantities such as built-in potential, depletion widths, charge density, electric field, and electrostatic potential. Temperature dependence is represented using an approximate band-gap relation and an intrinsic carrier concentration scaled to a reference value at 300 K.

The built-in checks test mathematical consistency and expected qualitative trends within the implementation. They are **not** experimental validation, and passing them does not establish that the model captures all real-device behavior.

## Limitations

- This is an educational depletion-approximation model, not a full semiconductor-device simulator.
- It does not use a self-consistent Poisson solver or experimental calibration.
- It does not implement a completed diode current–voltage (I–V) model, carrier-injection/transport simulation, or breakdown model.
- Results depend on the stated approximations and should be interpreted as illustrative rather than as experimental measurements.

## Project purpose

The project began with a question about how temperature affects the built-in potential of a p–n junction. The simulator was developed to make the relevant electrostatic quantities visible and to explore the effects of changing temperature, doping asymmetry, and applied bias.
