"""
EC065 -- Offshore Fixed-Bottom Wind Turbine -- F2a BEM Steady-State

Blade Element Momentum (BEM) method for a large offshore wind turbine.
Based on the NREL 5MW reference turbine (Jonkman et al., 2009).

For each of N radial blade elements, iteratively solves for axial (a)
and tangential (a') induction factors by balancing:
  - Momentum theory:  dT = 4*pi*r*rho*V^2*a*(1-a)*F*dr
  - Blade element:    dT = 0.5*rho*W^2*B*c*(Cl*cos(phi)+Cd*sin(phi))*dr

Includes:
  - Prandtl tip-loss and hub-loss corrections
  - Glauert empirical correction for heavily-loaded rotors (a > ~0.4)
  - Simplified airfoil Cl/Cd (linear + flat-plate post-stall with Viterna)
  - Drivetrain efficiency loss
  - Pitch control for power limiting above rated wind speed

References:
    Jonkman et al. (2009) NREL/TP-500-38060 -- NREL 5MW Reference Turbine
    Burton et al. (2011) Wind Energy Handbook, 2nd ed., Wiley
    Hansen (2015) Aerodynamics of Wind Turbines, 3rd ed., Routledge
    Buhl (2005) NREL/TP-500-36834 -- New empirical relationship for a > 0.4
"""

import numpy as np


class OffshoreWindBEM_F2a:
    """Offshore fixed-bottom wind turbine -- BEM steady-state model."""

    def __init__(self, params: dict):
        t = params["turbine"]
        af = params.get("airfoil", {})

        self.B = t["N_blades"]["value"]
        self.R = t["R"]["value"]
        self.N_el = t["N_elements"]["value"]
        self.rho = t["rho_air"]["value"]
        self.rated_power_kw = t["rated_power"]["value"]
        self.rated_wind = t["rated_wind"]["value"]
        self.cut_in = t["cut_in"]["value"]
        self.cut_out = t["cut_out"]["value"]
        self.tsr_design = t["tip_speed_ratio_design"]["value"]
        self.rpm_rated = t.get("rpm_rated", {}).get("value", 12.1)
        self.eta_dt = t.get("drivetrain_efficiency", {}).get("value", 0.944)

        # Chord and twist distribution
        self.chord_root = t["chord_root"]["value"]
        self.chord_tip = t["chord_tip"]["value"]
        self.twist_root = np.radians(t["twist_root"]["value"])
        self.twist_tip = np.radians(t["twist_tip"]["value"])

        # Hub radius
        r_hub_abs = t.get("r_hub", {}).get("value", 1.5)
        self.r_hub = max(r_hub_abs, 0.05 * self.R)

        # Airfoil parameters
        self.Cl_slope = af.get("Cl_slope", {}).get("value", 6.28)   # 2*pi / rad
        self.Cl_zero = af.get("Cl_zero_alpha", {}).get("value", 0.2)
        self.alpha_stall = np.radians(af.get("alpha_stall_deg", {}).get("value", 14.0))
        self.Cd_min = af.get("Cd_min", {}).get("value", 0.008)

        # Build element positions
        self.r = np.linspace(self.r_hub, self.R * 0.995, self.N_el)
        self.dr = np.diff(
            np.concatenate([[self.r_hub], (self.r[:-1] + self.r[1:]) / 2.0, [self.R]])
        )

        # Linear chord and twist distributions
        frac = (self.r - self.r_hub) / (self.R * 0.995 - self.r_hub)
        self.chord = self.chord_root + (self.chord_tip - self.chord_root) * frac
        self.twist = self.twist_root + (self.twist_tip - self.twist_root) * frac

    # ------------------------------------------------------------------
    # Airfoil aerodynamics
    # ------------------------------------------------------------------
    def airfoil_cl_cd(self, alpha_rad):
        """
        Simplified airfoil Cl/Cd for NREL S-series.

        Linear region up to stall, Viterna-style flat plate post-stall.
        Alpha in radians.
        """
        alpha_deg = np.degrees(alpha_rad)

        # Cl: linear below stall, Viterna post-stall
        if alpha_deg < -5.0:
            Cl = -0.5
        elif alpha_deg <= np.degrees(self.alpha_stall):
            Cl = self.Cl_zero + (self.Cl_slope * alpha_rad)
        else:
            # Post-stall: gradual decline (Viterna approximation)
            Cl_stall = self.Cl_zero + self.Cl_slope * self.alpha_stall
            excess = alpha_deg - np.degrees(self.alpha_stall)
            Cl = Cl_stall * max(0.1, 1.0 - 0.04 * excess)

        # Cd: parabolic drag polar
        Cd = self.Cd_min + 0.005 * alpha_deg ** 2 / 100.0
        if alpha_deg > np.degrees(self.alpha_stall):
            # Post-stall drag rise
            excess = alpha_deg - np.degrees(self.alpha_stall)
            Cd += 0.02 * (excess / 10.0) ** 2
        Cd = max(Cd, self.Cd_min)
        return Cl, Cd

    # ------------------------------------------------------------------
    # Prandtl tip and hub loss factors
    # ------------------------------------------------------------------
    def _prandtl_tip_loss(self, r, phi):
        """Prandtl tip-loss correction factor."""
        if abs(np.sin(phi)) < 1e-8:
            return 1.0
        f_exp = self.B / 2.0 * (self.R - r) / (r * abs(np.sin(phi)) + 1e-12)
        f_exp = min(f_exp, 30.0)  # Prevent overflow
        F_tip = (2.0 / np.pi) * np.arccos(min(1.0, np.exp(-f_exp)))
        return F_tip

    def _prandtl_hub_loss(self, r, phi):
        """Prandtl hub-loss correction factor."""
        if abs(np.sin(phi)) < 1e-8:
            return 1.0
        f_exp = self.B / 2.0 * (r - self.r_hub) / (r * abs(np.sin(phi)) + 1e-12)
        f_exp = min(f_exp, 30.0)
        F_hub = (2.0 / np.pi) * np.arccos(min(1.0, np.exp(-f_exp)))
        return F_hub

    # ------------------------------------------------------------------
    # BEM solve for one element
    # ------------------------------------------------------------------
    def _solve_element(self, r, dr, chord, twist, V_inf, omega, pitch_rad):
        """Iterative BEM for one blade element with Prandtl tip/hub loss."""
        sigma = self.B * chord / (2.0 * np.pi * r)
        a = 0.1
        a_prime = 0.01

        for _ in range(300):
            # Flow angle
            V_ax = V_inf * (1.0 - a)
            V_tan = omega * r * (1.0 + a_prime)
            phi = np.arctan2(V_ax, V_tan)

            # Prandtl combined tip + hub loss
            F_tip = self._prandtl_tip_loss(r, phi)
            F_hub = self._prandtl_hub_loss(r, phi)
            F = F_tip * F_hub
            F = max(F, 0.01)  # Floor to prevent division by zero

            # Angle of attack
            alpha = phi - twist - pitch_rad
            Cl, Cd = self.airfoil_cl_cd(alpha)

            # Normal and tangential force coefficients
            Cn = Cl * np.cos(phi) + Cd * np.sin(phi)
            Ct_blade = Cl * np.sin(phi) - Cd * np.cos(phi)

            sin_phi = np.sin(phi)
            cos_phi = np.cos(phi)

            if abs(sin_phi) < 1e-8:
                break

            # New induction factors with Prandtl loss
            denom_a = 4.0 * F * sin_phi ** 2 / (sigma * Cn + 1e-12) + 1.0
            a_new = 1.0 / denom_a

            denom_ap = 4.0 * F * sin_phi * cos_phi / (sigma * Ct_blade + 1e-12) - 1.0
            if abs(denom_ap) < 1e-12:
                a_prime_new = 0.0
            else:
                a_prime_new = 1.0 / denom_ap

            # Glauert / Buhl correction for heavily loaded rotors
            if a_new > 0.4:
                ac = 1.0 / 3.0
                CT_local = sigma * (1.0 - a_new) ** 2 * Cn / (sin_phi ** 2 + 1e-12)
                CT_limit = 4.0 * F * ac * (1.0 - ac)
                if CT_local > CT_limit:
                    # Buhl empirical correction
                    K = 4.0 * F * sin_phi ** 2 / (sigma * Cn + 1e-12)
                    a_new = (2.0 + K * (1.0 - 2.0 * ac) -
                             np.sqrt(max(0, (K * (1.0 - 2.0 * ac) + 2.0) ** 2
                                         - 4.0 * (K * (1.0 - ac ** 2) - 1.0)))) / (
                                2.0 * (1.0 + K))
                    a_new = max(a_new, ac)

            a_new = np.clip(a_new, 0.0, 0.95)
            a_prime_new = np.clip(a_prime_new, -0.5, 0.95)

            # Convergence check
            if abs(a_new - a) < 1e-6 and abs(a_prime_new - a_prime) < 1e-6:
                a = a_new
                a_prime = a_prime_new
                break

            # Under-relaxation for stability
            a = 0.5 * a + 0.5 * a_new
            a_prime = 0.5 * a_prime + 0.5 * a_prime_new

        # Final forces per unit span
        V_ax = V_inf * (1.0 - a)
        V_tan = omega * r * (1.0 + a_prime)
        W = np.sqrt(V_ax ** 2 + V_tan ** 2)
        phi = np.arctan2(V_ax, V_tan)
        alpha = phi - twist - pitch_rad
        Cl, Cd = self.airfoil_cl_cd(alpha)

        dF_N = 0.5 * self.rho * W ** 2 * chord * (
            Cl * np.cos(phi) + Cd * np.sin(phi)) * dr
        dF_T = 0.5 * self.rho * W ** 2 * chord * (
            Cl * np.sin(phi) - Cd * np.cos(phi)) * dr

        dThrust = self.B * dF_N
        dTorque = self.B * dF_T * r
        dPower = dTorque * omega

        return {
            "r": r,
            "a": a,
            "a_prime": a_prime,
            "phi_deg": np.degrees(phi),
            "alpha_deg": np.degrees(alpha),
            "Cl": Cl,
            "Cd": Cd,
            "dThrust_N": dThrust,
            "dTorque_Nm": dTorque,
            "dPower_W": dPower,
        }

    # ------------------------------------------------------------------
    # Full BEM solve
    # ------------------------------------------------------------------
    def solve(self, wind_speed_m_s, pitch_deg=0.0, rpm=None,
              tip_speed_ratio=None, air_density=None):
        """
        Run BEM analysis for the offshore wind turbine.

        Parameters
        ----------
        wind_speed_m_s : float
            Free-stream wind speed at hub height (m/s).
        pitch_deg : float
            Collective blade pitch angle in degrees (default 0).
        rpm : float or None
            Rotor speed in rpm. If None, derived from TSR.
        tip_speed_ratio : float or None
            If given, overrides TSR. Otherwise uses design TSR.
        air_density : float or None
            Override air density (kg/m3). Default uses stored value.

        Returns
        -------
        dict with keys:
            power_kw, thrust_kN, torque_kNm, Cp, Ct, omega_rad_s, rpm,
            blade_loads (list of per-element dicts)
        """
        V = wind_speed_m_s
        if air_density is not None:
            rho_orig = self.rho
            self.rho = air_density

        pitch_rad = np.radians(pitch_deg)

        if rpm is not None:
            omega = rpm * 2.0 * np.pi / 60.0
        elif tip_speed_ratio is not None:
            omega = tip_speed_ratio * V / self.R
        else:
            omega = self.tsr_design * V / self.R

        # Enforce rpm limits (realistic operational range)
        omega_rated = self.rpm_rated * 2.0 * np.pi / 60.0
        if rpm is None and tip_speed_ratio is None:
            omega = min(omega, omega_rated)

        total_power = 0.0
        total_thrust = 0.0
        total_torque = 0.0
        blade_loads = []

        for i in range(self.N_el):
            res = self._solve_element(
                self.r[i], self.dr[i], self.chord[i], self.twist[i],
                V, omega, pitch_rad,
            )
            total_power += res["dPower_W"]
            total_thrust += res["dThrust_N"]
            total_torque += res["dTorque_Nm"]
            blade_loads.append(res)

        # Apply drivetrain efficiency
        total_power_elec = total_power * self.eta_dt

        A_rotor = np.pi * self.R ** 2
        P_avail = 0.5 * self.rho * A_rotor * V ** 3
        Cp = total_power_elec / max(P_avail, 1e-6)
        Ct = total_thrust / (0.5 * self.rho * A_rotor * V ** 2 + 1e-6)

        if air_density is not None:
            self.rho = rho_orig

        return {
            "power_kw": total_power_elec / 1000.0,
            "thrust_kN": total_thrust / 1000.0,
            "torque_kNm": total_torque / 1000.0,
            "Cp": Cp,
            "Ct": Ct,
            "omega_rad_s": omega,
            "rpm": omega * 60.0 / (2.0 * np.pi),
            "blade_loads": blade_loads,
        }

    # ------------------------------------------------------------------
    # Power curve with simple pitch control above rated
    # ------------------------------------------------------------------
    def power_curve(self, wind_speeds, pitch_control=True):
        """
        Compute power curve across a range of wind speeds.

        Above rated wind speed, iteratively finds pitch angle that
        limits power to rated power (simple proportional pitch schedule).

        Parameters
        ----------
        wind_speeds : array-like
            Wind speeds in m/s.
        pitch_control : bool
            If True, pitch blades above rated to limit power.

        Returns
        -------
        dict of arrays: power_kw, thrust_kN, torque_kNm, Cp, Ct, pitch_deg
        """
        wind_speeds = np.atleast_1d(wind_speeds)
        results = {k: np.zeros(len(wind_speeds)) for k in
                   ["power_kw", "thrust_kN", "torque_kNm", "Cp", "Ct", "pitch_deg"]}

        for i, V in enumerate(wind_speeds):
            if V < self.cut_in or V > self.cut_out:
                continue

            # Below rated: fine pitch, optimal TSR
            if V <= self.rated_wind or not pitch_control:
                r = self.solve(V, pitch_deg=0.0)
                results["power_kw"][i] = min(r["power_kw"], self.rated_power_kw)
                results["thrust_kN"][i] = r["thrust_kN"]
                results["torque_kNm"][i] = r["torque_kNm"]
                results["Cp"][i] = r["Cp"]
                results["Ct"][i] = r["Ct"]
                results["pitch_deg"][i] = 0.0
            else:
                # Above rated: find pitch to limit power
                pitch = 0.0
                for _ in range(30):
                    r = self.solve(V, pitch_deg=pitch)
                    if r["power_kw"] <= self.rated_power_kw * 1.01:
                        break
                    # Simple proportional pitch increment
                    pitch += 0.5 * (r["power_kw"] - self.rated_power_kw) / self.rated_power_kw * 5.0
                    pitch = min(pitch, 30.0)

                results["power_kw"][i] = min(r["power_kw"], self.rated_power_kw)
                results["thrust_kN"][i] = r["thrust_kN"]
                results["torque_kNm"][i] = r["torque_kNm"]
                results["Cp"][i] = r["Cp"]
                results["Ct"][i] = r["Ct"]
                results["pitch_deg"][i] = pitch

        return results
