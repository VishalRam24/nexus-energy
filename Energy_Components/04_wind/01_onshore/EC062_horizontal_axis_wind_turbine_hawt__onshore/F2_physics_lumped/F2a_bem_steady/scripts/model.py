"""
EC062 -- HAWT Onshore -- F2a BEM Steady-State

Blade Element Momentum (BEM) method. Divides blade into N radial elements.
For each element, iteratively solve for axial (a) and tangential (a') induction
factors by balancing momentum theory with blade-element lift/drag forces.

Simplified NACA 4412 airfoil Cl/Cd used.

Reference:
    Burton et al. (2011), Wind Energy Handbook, 2nd ed., Wiley
    Hansen (2015), Aerodynamics of Wind Turbines, 3rd ed., Routledge
"""

import numpy as np


class HAWT_BEM_F2a:
    """Horizontal axis wind turbine -- BEM steady-state model."""

    def __init__(self, params: dict):
        t = params["turbine"]
        self.B = t["N_blades"]["value"]
        self.R = t["R"]["value"]
        self.N_el = t["N_elements"]["value"]
        self.rho = t["rho_air"]["value"]
        self.chord_root = t["chord_root"]["value"]
        self.chord_tip = t["chord_tip"]["value"]
        self.twist_root = np.radians(t["twist_root"]["value"])
        self.twist_tip = np.radians(t["twist_tip"]["value"])
        self.tsr_design = t["tip_speed_ratio_design"]["value"]

        # Element radial positions (avoid hub center)
        r_hub = 0.1 * self.R
        self.r = np.linspace(r_hub, self.R * 0.99, self.N_el)
        self.dr = np.diff(
            np.concatenate([[r_hub], (self.r[:-1] + self.r[1:]) / 2, [self.R]])
        )

        # Linear chord and twist distributions
        frac = (self.r - r_hub) / (self.R * 0.99 - r_hub)
        self.chord = self.chord_root + (self.chord_tip - self.chord_root) * frac
        self.twist = self.twist_root + (self.twist_tip - self.twist_root) * frac

    # ------------------------------------------------------------------
    # Simplified NACA 4412 Cl, Cd
    # ------------------------------------------------------------------
    @staticmethod
    def airfoil_cl_cd(alpha_rad):
        """
        Simplified NACA 4412 aerodynamic coefficients.
        alpha in radians.
        """
        alpha_deg = np.degrees(alpha_rad)
        # Linear Cl up to stall (~15 deg), then flat
        if alpha_deg < -5:
            Cl = -0.5
        elif alpha_deg < 15:
            Cl = 0.1 + 0.11 * alpha_deg  # ~2*pi per radian ≈ 0.11/deg
        else:
            Cl = 1.2  # post-stall plateau
        # Cd: parabolic drag polar
        Cd = 0.008 + 0.005 * alpha_deg ** 2 / 100.0
        Cd = max(Cd, 0.008)
        return Cl, Cd

    # ------------------------------------------------------------------
    # BEM solve for one element
    # ------------------------------------------------------------------
    def _solve_element(self, r, dr, chord, twist, V_inf, omega, pitch_rad):
        """Iterative BEM for one blade element."""
        sigma = self.B * chord / (2.0 * np.pi * r)
        a = 0.1
        a_prime = 0.01

        for _ in range(200):
            # Flow angle
            V_ax = V_inf * (1.0 - a)
            V_tan = omega * r * (1.0 + a_prime)
            W = np.sqrt(V_ax ** 2 + V_tan ** 2)
            phi = np.arctan2(V_ax, V_tan)

            # Angle of attack
            alpha = phi - twist - pitch_rad
            Cl, Cd = self.airfoil_cl_cd(alpha)

            # Normal and tangential force coefficients
            Cn = Cl * np.cos(phi) + Cd * np.sin(phi)
            Ct = Cl * np.sin(phi) - Cd * np.cos(phi)

            sin_phi = np.sin(phi)
            cos_phi = np.cos(phi)

            # New induction factors
            if sin_phi < 1e-6:
                break

            a_new = 1.0 / (4.0 * sin_phi ** 2 / (sigma * Cn + 1e-12) + 1.0)
            a_prime_new = 1.0 / (4.0 * sin_phi * cos_phi / (sigma * Ct + 1e-12) - 1.0)

            # Glauert correction for heavily loaded rotors
            if a_new > 0.4:
                # Empirical Glauert correction
                CT = sigma * (1.0 - a_new) ** 2 * Cn / (sin_phi ** 2 + 1e-12)
                if CT > 0.96:
                    a_new = 1.0 / 3.0 * (2.0 + CT * (1.0 - 2.0 * 0.4) - np.sqrt(
                        max(0, (CT * (1.0 - 2.0 * 0.4) + 2.0) ** 2 - 4.0 * (CT - 0.4 ** 2 * CT))))

            a_new = np.clip(a_new, 0.0, 0.95)
            a_prime_new = np.clip(a_prime_new, -0.5, 0.95)

            # Relaxation
            if abs(a_new - a) < 1e-6 and abs(a_prime_new - a_prime) < 1e-6:
                a = a_new
                a_prime = a_prime_new
                break
            a = 0.5 * a + 0.5 * a_new
            a_prime = 0.5 * a_prime + 0.5 * a_prime_new

        # Forces per unit span
        V_ax = V_inf * (1.0 - a)
        V_tan = omega * r * (1.0 + a_prime)
        W = np.sqrt(V_ax ** 2 + V_tan ** 2)
        phi = np.arctan2(V_ax, V_tan)
        alpha = phi - twist - pitch_rad
        Cl, Cd = self.airfoil_cl_cd(alpha)

        dF_N = 0.5 * self.rho * W ** 2 * chord * (Cl * np.cos(phi) + Cd * np.sin(phi)) * dr
        dF_T = 0.5 * self.rho * W ** 2 * chord * (Cl * np.sin(phi) - Cd * np.cos(phi)) * dr

        dThrust = self.B * dF_N
        dTorque = self.B * dF_T * r
        dPower = dTorque * omega

        return {
            "a": a, "a_prime": a_prime,
            "alpha_deg": np.degrees(alpha),
            "Cl": Cl, "Cd": Cd,
            "dThrust_N": dThrust,
            "dTorque_Nm": dTorque,
            "dPower_W": dPower,
        }

    # ------------------------------------------------------------------
    # Full BEM solve
    # ------------------------------------------------------------------
    def solve(self, wind_speed_m_s, pitch_deg=0.0, rpm=None, tip_speed_ratio=None):
        """
        Run BEM analysis.

        Parameters
        ----------
        wind_speed_m_s : float
        pitch_deg : float (blade pitch, default 0)
        rpm : float (rotor speed) OR
        tip_speed_ratio : float (lambda = omega*R/V)

        Returns
        -------
        dict: power_kw, thrust_kN, Cp, Ct, blade_loads (per element)
        """
        V = wind_speed_m_s
        pitch_rad = np.radians(pitch_deg)

        if rpm is not None:
            omega = rpm * 2.0 * np.pi / 60.0
        elif tip_speed_ratio is not None:
            omega = tip_speed_ratio * V / self.R
        else:
            omega = self.tsr_design * V / self.R

        total_power = 0.0
        total_thrust = 0.0
        blade_loads = []

        for i in range(self.N_el):
            res = self._solve_element(
                self.r[i], self.dr[i], self.chord[i], self.twist[i],
                V, omega, pitch_rad
            )
            total_power += res["dPower_W"]
            total_thrust += res["dThrust_N"]
            blade_loads.append(res)

        A_rotor = np.pi * self.R ** 2
        P_avail = 0.5 * self.rho * A_rotor * V ** 3
        Cp = total_power / max(P_avail, 1e-6)
        Ct = total_thrust / (0.5 * self.rho * A_rotor * V ** 2 + 1e-6)

        return {
            "power_kw": total_power / 1000.0,
            "thrust_kN": total_thrust / 1000.0,
            "Cp": Cp,
            "Ct": Ct,
            "omega_rad_s": omega,
            "rpm": omega * 60.0 / (2.0 * np.pi),
            "blade_loads": blade_loads,
        }
