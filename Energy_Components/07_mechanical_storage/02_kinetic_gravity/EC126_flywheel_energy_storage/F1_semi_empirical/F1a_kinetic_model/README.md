# EC126 — Flywheel Energy Storage — F1a Kinetic Model

## Overview
Semi-empirical kinetic model for a 100 kW / 25 kWh steel flywheel energy storage system. Computes stored energy, SOC, electrical power, self-discharge, and round-trip efficiency from rotor speed and torque.

## Model Equations
```
omega        = speed_rpm * 2*pi / 60                            [rad/s]
E            = 0.5 * J * omega^2 / 3.6e6                       [kWh]
SOC          = (omega^2 - omega_min^2) / (omega_max^2 - omega_min^2)
P_mech       = torque * omega                                   [W]
P_elec       = P_mech / eta_motor  (charging, torque > 0)
             = P_mech * eta_gen    (discharging, torque < 0)
dE/dt        = -k_sd * E           (self-discharge)
E(t)         = E0 * exp(-k_sd * t)
RTE          = eta_motor * eta_gen * exp(-k_sd * t_standby)
```

## Inputs
| Name | Unit | Range | Description |
|------|------|-------|-------------|
| speed_rpm | rpm | [8000, 16000] | Rotor angular speed |
| torque_nm | N·m | [-100, 100] | Shaft torque (+ charge, - discharge) |
| time_hours | h | [0, 24] | Standby time for self-discharge calc |

## Outputs
| Name | Unit | Description |
|------|------|-------------|
| energy_stored_kwh | kWh | Kinetic energy in flywheel |
| soc | - | State of charge (0=empty, 1=full) |
| power_kw | kW | Electrical power (+ draw, - supply) |
| self_discharge_kw | kW | Self-discharge power loss |
| round_trip_efficiency | - | RTE including standby losses |

## Parameters
| Parameter | Value | Unit | Source |
|-----------|-------|------|--------|
| J | 100 | kg·m² | Arani et al. (2017) |
| omega_max | 1675.94 | rad/s (16000 rpm) | Arani et al. (2017) |
| omega_min | 837.97 | rad/s (8000 rpm) | Arani et al. (2017) |
| P_rated | 100 | kW | Nameplate |
| E_rated | 25 | kWh | Computed |
| k_sd | 0.01 | 1/h | Arani et al. (~1%/h steel) |
| eta_motor | 0.95 | - | Arani et al. |
| eta_gen | 0.95 | - | Arani et al. |

## Sources
1. Arani, A.A.K. et al. (2017). Review of Flywheel Energy Storage Systems Structures and Applications. Energies, 10, 1361. doi:10.3390/en10091361

## Physics Checks
- E proportional to omega^2 (E ratio at 16k/8k rpm = 4.0)
- SOC in [0, 1] with SOC=0 at omega_min, SOC=1 at omega_max
- Self-discharge > 0 at all speeds
- RTE < 1, decreasing with standby time
- Charging (torque>0) → positive P_elec; Discharging (torque<0) → negative P_elec

## Limitations
- No thermal model (bearing and eddy current heat generation)
- No rotor dynamics / spin-up/down transient
- Steel flywheel parameters only (carbon fiber: higher speed, lower k_sd)
- No vacuum/magnetic bearing modeling for ultra-low k_sd variants
