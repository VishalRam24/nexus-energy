# EC044 — Monocrystalline Silicon PV — F1a Single-Diode Model

## Overview
Five-parameter single-diode model (De Soto) for mono-Si PV modules, wrapped from pvlib. Predicts MPP power, voltage, and current as a function of irradiance and cell temperature.

## Inputs
| Name | Unit | Range | Description |
|------|------|-------|-------------|
| irradiance | W/m2 | [0, 1200] | Plane-of-array irradiance |
| cell_temperature | degC | [-10, 80] | Cell temperature |

## Outputs
| Name | Unit | Description |
|------|------|-------------|
| v_mp | V | Voltage at maximum power point |
| i_mp | A | Current at maximum power point |
| p_mp | W | Maximum power |
| v_oc | V | Open-circuit voltage |
| i_sc | A | Short-circuit current |
| efficiency | - | Module efficiency |

## Sources
1. De Soto et al. (2006). Solar Energy, 80(1), 78-88.
2. pvlib v0.15 (BSD-3 license)
3. Canadian Solar CS6K-280M module datasheet

## Limitations
- Single module only (no string/array modeling)
- No partial shading effects
- No spectral or angular corrections
- Cell temperature must be provided (not estimated)
