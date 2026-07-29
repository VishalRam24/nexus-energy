# EC116 -- PWR -- F1b Load-Following with Xenon Dynamics

## Overview
Load-following model for a Pressurized Water Reactor with simplified Xe-135/I-135 transient kinetics. Builds on F1a (steady-state power map) by tracking xenon poison buildup after power changes, ramp rate constraints, and restart capability.

## Physics
- **Xenon-135 dynamics**: dXe/dt = gamma_Xe*Sigma_f*phi + lambda_I*I - (lambda_Xe + sigma_Xe*phi)*Xe
- **Iodine-135 dynamics**: dI/dt = gamma_I*Sigma_f*phi - lambda_I*I
- **Xenon peak**: After power reduction, Xe peaks at ~8-12 hours (less flux to burn Xe, continued I->Xe production)
- **Ramp rate limit**: 5%/min maximum
- **Restart capability**: Determined by available reactivity margin minus xenon penalty

## Parameters
| Parameter | Value | Unit |
|-----------|-------|------|
| P_thermal | 3400 | MW_th |
| eta_thermal | 0.33 | - |
| sigma_Xe | 2.65e-18 | cm2 |
| lambda_Xe | 2.09e-5 | 1/s |
| lambda_I | 2.87e-5 | 1/s |
| gamma_I | 0.064 | - |
| gamma_Xe | 0.003 | - |
| Ramp rate limit | 5 | %/min |

## References
- Todreas, N.E. & Kazimi, M.S. (2012). Nuclear Systems, 2nd ed. CRC Press.
- Stacey, W.M. (2007). Nuclear Reactor Physics, 2nd ed. Wiley.
- Duderstadt, J.J. & Hamilton, L.J. (1976). Nuclear Reactor Analysis. Wiley.
