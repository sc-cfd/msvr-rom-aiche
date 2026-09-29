# MSVR reduced-order model and CO2–MEA absorption UDF

Code accompanying the article

> S. Chen, A. Kourou, B. Wang, J. Hua, K. Van Dooren, T. Verspeelt, Q. Xiong, G. J. Heynderickx, K. M. Van Geem, Y. Ouyang.
> Scale-up of vortex reactors for CO2 capture: from reactive CFD to design models. *AIChE Journal* (2026). DOI: to be added.

The repository contains the reduced-order model of the cyclone-integrated multistage vortex reactor (MSVR), the design scripts of Section 4.4 of the article, and the ANSYS Fluent user-defined function (UDF) for CO2 absorption into aqueous MEA used in the CFD simulations.

## Contents

| File | Content |
|---|---|
| `msvr_rom.py` | CFD results of the 23 calibration cases, model equations (Supporting Information, Eqs. S2–S18) and calibration of the constants of Table 2 |
| `msvr_design.py` | Design of a new MSVR at G = 60 Nm³/h and L = 216 kg/h and design-space map (Figure 8) |
| `CO2_MEA_absorption_293K.c` | Fluent UDF with the source terms of CO2 absorption into 30 wt% MEA at 293.15 K |
| `requirements.txt` | Python packages |

## Reduced-order model

Python 3.9 or newer with numpy and scipy is required. matplotlib is needed for the figures and plotly for the interactive three-dimensional view.

```
pip install -r requirements.txt
```

### Calibration

```
python msvr_rom.py
```

The script fits the ten constants of Table 2 to the 90 CFD values by multi-start nonlinear least squares (about 20 s). It prints the constants and the mean and maximum deviations from the CFD results, and writes the model and CFD values of every case to `parity.csv` (Figure S3).

### Design and figures

```
python msvr_design.py
```

The script uses the constants of Table 2. It selects the geometry with the minimum pressure drop that meets all four targets over a ±15% turndown, and writes

| Output | Content |
|---|---|
| `design_space.csv` | 6000 sampled geometries and the Pareto-optimal set |
| `Figure8.png` | design-space map |
| `Figure8_3D.png`, `Figure8_3D.html` | static and interactive three-dimensional view of the design space |

`msvr_design.py` imports the model from `msvr_rom.py`, so both files have to be in the same folder.

### Units and constants

G in Nm³/h, L in kg/h, Q_G = G/3600 in m³/s, lengths in m, velocities in m/s, liquid load φ = L/G in kg/Nm³, pressure drop in Pa.

The swirl coefficient SF_c of Eq. (S3) enters the model only through the combinations Eu·SF_c², v_a/SF_c and v_crit/SF_c. These combinations are fitted as single constants.

## Fluent UDF

`CO2_MEA_absorption_293K.c` was used with ANSYS Fluent 2024 R2 (Euler–Euler model, SST k–ω turbulence model, population balance model) and is loaded as a compiled UDF.

The UDF assumes the following setup.

- Phase 0 (primary phase) is the liquid. MEA is species 1 of the liquid mixture, and the mixture contains the reaction products MEAH⁺ and MEACOO⁻.
- Phase 1 (secondary phase) is the gas, a mixture of CO2 (species 0) and N2 (species 1). The bubble diameter is taken from the population balance model.
- The turbulent kinetic energy and the specific dissipation rate are read from the liquid phase.
- Two user-defined memory locations are needed. UDM 0 stores the turbulent dissipation rate and UDM 1 the MEA source term.

The source terms are hooked as follows.

| Function | Phase | Equation |
|---|---|---|
| `gas_src` | gas | mass |
| `CO2_src` | gas | CO2 species |
| `liq_src` | liquid | mass |
| `liqmea_src` | liquid | MEA species |
| `liqmeah_src` | liquid | MEAH⁺ species |
| `liqmeacoo_src` | liquid | MEACOO⁻ species |

The CO2 flux follows the two-film theory with the enhancement factor E = (1 + Ha²)^0.5. The liquid-side mass transfer coefficient follows from the small-eddy model, and the CO2 diffusivity from the N2O analogy. The interfacial CO2 concentration follows from Henry's law at the local absolute pressure. The overall reaction is CO2 + 2 MEA → MEAH⁺ + MEACOO⁻ with k2 = 4450 L/(mol·s) at 293.15 K, and the bulk liquid CO2 concentration is taken as zero. The model is described in detail in

> S. Chen, X. Lang, A. Kourou, S. Dutta, K. M. Van Geem, Y. Ouyang, G. J. Heynderickx. Enhancing CO2 capture efficiency: computational fluid dynamics investigation of gas-liquid vortex reactor configurations for process intensification. *Chemical Engineering Journal* 493 (2024) 152535. https://doi.org/10.1016/j.cej.2024.152535

## Citation

Please cite the article above when using this code.

## License

MIT License, see `LICENSE`.
