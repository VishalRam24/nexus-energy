# EC060 — Solar Pond — F1b: Incidence Angle Modifier (Fresnel/Snell)

## Model Summary
Extends F1a Hottel-Whillier model for the Lower Convective Zone (LCZ) by adding a physics-based Incidence Angle Modifier derived from Snell's law refraction and Fresnel reflectance at the brine surface.

## Physics
- **IAM**: Snell-law refraction into brine (n = 1.40), Fresnel s+p polarization reflectance. IAM(θ) = τ(θ)/τ(0)
- **Useful heat**: Q_u = A × [τ_pond × IAM(θ) × α_lcz × G − U_lcz × (T_lcz − T_amb)]
- **LCZ extraction**: T_out = T_lcz + Q_u / (ṁ × cp)

## Key Physics Notes
- Solar ponds use brine density gradient to suppress convection; NCZ is transparent, LCZ absorbs
- Brine refraction index ~1.40 significantly reduces reflection loss vs flat glass (n~1.5)
- IAM is important for tilted sun angles, especially in high-latitude installations

## Reference Parameters
- Area: 1000 m², U_lcz = 1.2 W/m²K, τ_pond = 0.70, n_brine = 1.40

## References
- Duffie & Beckman (2013), Solar Engineering of Thermal Processes, Ch. 9
- Singh et al. (2011), Renew. Sust. Energy Rev. 15(4), 1773–1781
- Tabor & Matz (1965), Solar Energy 9(4), 177–182
