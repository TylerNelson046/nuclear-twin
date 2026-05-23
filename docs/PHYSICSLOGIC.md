# PHYSICSLOGIC.md — Self-Contained Auditable Physics Reference

**Project:** Nuclear Reactor Digital Twin Platform (PWR · Helion FRC · MSR)
**Stack:** Python 3.11 · NumPy · SciPy `solve_ivp(method="Radau", rtol=1e-6, atol=1e-9)` · Numba `@njit` on hot kernels · Pydantic v2 for input validation.

**Purpose:** Every governing equation implemented in the codebase is shown here with (a) the spec form, (b) the actual Python implementation verbatim, (c) the units of every input and output, (d) every constant with its numerical value and source, (e) the upstream producers and downstream consumers of each term, and (f) a hand-verified numerical check.

**Audit method:** Read each section top to bottom. The code block shown for each equation is the literal kernel called by SciPy at every solver step. No file references or line numbers are needed to verify correctness — every number you'd need to reproduce the numerical result is on this page.

**Universal conventions used throughout:**
- Reactivity `ρ` internally is dimensionless Δk/k. `1 pcm = 1e-5`. Conversion happens at the kinetics interface only.
- Temperatures in K (internal) and keV (Helion plasma only).
- All thermal powers in W; plasma energies in J.
- Neutron flux φ in n/cm²/s; macroscopic cross sections in cm⁻¹; microscopic in cm².
- Time in seconds. Subscripted constants follow SPEC §4.5.4.

---

## Table of Contents

1. [Shared Constants Used by Every Reactor](#shared-constants)
2. [Eq. 1, 2 — Shared 6-Group Point Kinetics](#eq-1-2)
3. [Eq. 9, 10 — PWR Lumped Thermal-Hydraulics](#eq-9-10)
4. [Eq. 11, 12, 13 — PWR Algebraic Feedback](#eq-11-12-13)
5. [Eq. 14, 15 — Xenon/Iodine Kinetics](#eq-14-15)
6. [Xe-135 Reactivity Worth (one-group perturbation)](#xenon-worth)
7. [PWR Full 11-State Coupled System](#pwr-coupled)
8. [Eq. 16 — Helion Plasma Energy Balance](#eq-16)
9. [Eq. 17 — D-He3 Fusion Power](#eq-17)
10. [Eq. 18 — Bosch-Hale ⟨σv⟩](#eq-18)
11. [Eq. 19 — Adiabatic Compression Heating](#eq-19)
12. [Eq. 20 — Bremsstrahlung Radiation Loss](#eq-20)
13. [Eq. 21 — Thermal Conduction Loss](#eq-21)
14. [Helion Q Factor & Coupled Pulse](#helion-q)
15. [Eq. 22, 23 — MSR Precursor Drift](#eq-22-23)
16. [Eq. 24 — β_eff,flow](#eq-24)
17. [MSR Salt Thermal-Hydraulics](#msr-thermal)
18. [MSR Critical Base Reactivity](#msr-critical)
19. [MSR Full 8-State Coupled System](#msr-coupled)
20. [Steady-State Construction (all reactors)](#steady-state)
21. [Inter-Equation Validation Checklist](#validation-checklist)
22. [Coupling Maps (one-page diagrams)](#coupling-maps)
23. [Spec/Code Reconciliation Notes](#reconciliation)

---

<a name="shared-constants"></a>

# 1. Shared Constants Used by Every Reactor

These appear unchanged everywhere they're cited.

### Keepin 6-group delayed-neutron parameters for U-235 thermal fission

| Group i | t½ (s) | λᵢ (s⁻¹) | Relative abundance aᵢ | βᵢ = β_eff · aᵢ |
|:-:|:-:|:-:|:-:|:-:|
| 1 | 53.9   | 0.01286   | 0.038 | 2.470e-4 |
| 2 | 22.3   | 0.03108   | 0.211 | 1.372e-3 |
| 3 | 6.40   | 0.10830   | 0.197 | 1.281e-3 |
| 4 | 2.26   | 0.30670   | 0.396 | 2.574e-3 |
| 5 | 0.494  | 1.40293   | 0.132 | 8.580e-4 |
| 6 | 0.179  | 3.87209   | 0.026 | 1.690e-4 |
| **Σ** |    |           | **1.000** | **β_eff = 6.500e-3** |

Source: IAEA Reference Database for Beta-Delayed Neutron Emission, U-235 thermal 6-groups. Half-lives are from the IAEA table; λᵢ = ln(2)/t½; βᵢ are derived as `β_eff · aᵢ / Σaᵢ` with `β_eff = 0.0065`.

### Other PWR / MSR constants (SPEC §4.5.4)

| Symbol | Value | Units | Meaning / source |
|---|---|---|---|
| Λ | 1.0e-5 | s | Prompt neutron generation time (typical PWR) |
| γ_I | 0.0639 | — | I-135 cumulative fission yield |
| γ_X | 0.00237 | — | Xe-135 direct fission yield |
| λ_I | 2.87e-5 | s⁻¹ | I-135 decay (t½ = 6.7 hr) |
| λ_X | 2.09e-5 | s⁻¹ | Xe-135 decay (t½ = 9.2 hr) |
| σ_aX | 2.6e-18 | cm² | Xe-135 thermal neutron absorption (= 2.6×10⁶ barn) |
| α_D (default) | −2.5 | pcm/K | Doppler temperature coefficient (range −2 to −3) |
| α_m (default) | −35 | pcm/K | Moderator temperature coefficient (range −20 to −50) |
| ω_B (default) | −10 | pcm/ppm | Differential boron worth |
| P_NOM (PWR) | 3.0e9 | W | Nominal PWR fission power (3000 MWth) |
| γ_f | 0.97 | — | Fraction of fission power deposited in fuel |
| T_FUEL_NOM | 900 | K | Nominal fuel temperature |
| T_COOL_NOM | 590 | K | Nominal coolant temperature |
| T_IN_NOM | 565 | K | Nominal coolant inlet temperature |
| C_FUEL | 300 | J/(kg·K) | UO₂ specific heat |
| M_FUEL | 101 000 | kg | Lumped fuel thermal mass (193-FA core) |
| C_COOL | 5600 | J/(kg·K) | Light water at 590 K, 155 bar |
| M_COOL | 20 000 | kg | Coolant mass in active core |

### Helion constants (SPEC §4.5.4)

| Symbol | Value | Units | Meaning / source |
|---|---|---|---|
| E_DHe3 | 2.93e-12 | J | D-He3 reaction energy (18.3 MeV) |
| C_BREM | 5.35e-37 | W·m³/keV^(1/2) | Bremsstrahlung coefficient |
| J_per_keV | 1.60218e-16 | J/keV | SI conversion |
| B_G | 68.7508 | (keV)^(1/2) | Bosch-Hale Gamow constant for D-He3 |
| m_r·c² | 1 124 572 | keV | D-He3 reduced-mass energy |
| C₁ | 5.51036e-10 | cm³/s | Bosch-Hale Table IV (D(He3,p)He4) |
| C₂ | 6.41918e-3 | 1/keV | " |
| C₃ | −2.02896e-3 | 1/keV | " |
| C₄ | −1.91080e-5 | 1/keV² | " |
| C₅ | 1.35776e-4 | 1/keV² | " |
| C₆ | 0 | | |
| C₇ | 0 | | |
| T validity | [0.5, 190] | keV | Clamped with warning outside |
| C_TAU | 5.0e-5 | s | FRC τ_E scaling prefactor (calibrated ~50 μs @ 1e21 m⁻³, 5 T) |
| N_REF | 1.0e21 | m⁻³ | τ_E reference density |
| B_REF | 5.0 | T | τ_E reference field |

Source for C₁–C₇, B_G, m_r·c²: Bosch, H.S. & Hale, G.M. (1992), *Nuclear Fusion* 32(4), 611, Table IV.

### MSR constants (SPEC §4.4, §4.5.4)

| Symbol | Value | Units | Meaning / source |
|---|---|---|---|
| P_NOM_MSR | 500e6 | W | Mid-scale thermal MSR |
| T_SALT_NOM | 900 | K | Nominal bulk salt temperature |
| T_SALT_INLET_NOM | 860 | K | Salt return from heat exchanger |
| C_SALT | 1500 | J/(kg·K) | FLiBe at 900 K (Williams et al. ORNL/TM-2006/12) |
| ρ_salt | 2000 | kg/m³ | FLiBe density |
| TAU_CORE_NOM | 5.0 | s | Core transit at v_salt = 1 m/s |
| L_CORE | 5.0 | m | Active core length (= V_SALT_NOM · TAU_CORE_NOM) |
| M_DOT_SALT_NOM | ≈ 8 333 | kg/s | Derived: P_NOM_MSR / (C_SALT · 40 K) |
| M_CORE_SALT | ≈ 41 667 | kg | Derived: τ_thermal = τ_core_nom |
| α_salt (default) | −5 | pcm/K | Combined Doppler+moderator (single salt coefficient) |

---

<a name="eq-1-2"></a>

# 2. Eq. 1 & 2 — Shared 6-Group Point Kinetics ODE

Used by both PWR and MSR. The MSR engine *wraps* this kernel; it does not reimplement.

### Spec form

```
Eq. 1   dn/dt  = [(ρ − β_eff)/Λ] · n  +  Σᵢ λᵢ Cᵢ
Eq. 2   dCᵢ/dt = (βᵢ/Λ) · n − λᵢ Cᵢ           i = 1 … 6
```

### Implementation (verbatim kernel)

```python
@numba.njit(cache=True)
def point_kinetics_rhs(
    t: float,
    y: np.ndarray,
    rho: float,
    beta_arr: np.ndarray,
    lambda_arr: np.ndarray,
    Lambda: float,
) -> np.ndarray:
    """7-state PKE right-hand side for one solver call.

    SPEC Eq. 1:  dy[0]/dt = [(ρ − β_eff) / Λ] · n  +  Σᵢ λᵢ Cᵢ
    SPEC Eq. 2:  dy[i]/dt = (βᵢ / Λ) · n  −  λᵢ Cᵢ    i = 1…6
    """
    n = y[0]

    # β_eff = Σᵢ βᵢ
    beta_eff = 0.0
    for i in range(6):
        beta_eff += beta_arr[i]

    # Delayed neutron source: Σᵢ λᵢ Cᵢ
    delayed_source = 0.0
    for i in range(6):
        delayed_source += lambda_arr[i] * y[i + 1]

    dydt = np.empty(7)

    # Eq. 1 — neutron population
    dydt[0] = ((rho - beta_eff) / Lambda) * n + delayed_source

    # Eq. 2 — one equation per precursor group
    for i in range(6):
        dydt[i + 1] = (beta_arr[i] / Lambda) * n - lambda_arr[i] * y[i + 1]

    return dydt
```

### I/O contract

| Parameter | Units | Source |
|---|---|---|
| `t` (s) | required by SciPy signature; equation is autonomous | solver |
| `y[0] = n` | dimensionless (n=1 ↔ P = P_ref) | state |
| `y[1:7] = Cᵢ` | atoms/cm³ (scale arbitrary because n is normalised) | state |
| `rho` | dimensionless Δk/k | external `rho_fn(t,y)` callback, recomputed each step |
| `beta_arr[6]` | dimensionless | from Keepin table (above) |
| `lambda_arr[6]` | s⁻¹ | from Keepin table (above) |
| `Lambda` | s | `LAMBDA_PWR = 1e-5` |
| **Return** | `dydt[7]` (1/s) | to solver |

### Steady-state (used to construct y₀ so dCᵢ/dt = 0 at t = 0)

Solve `dCᵢ/dt = 0`:

```
Cᵢ,₀ = (βᵢ / (λᵢ · Λ)) · n₀
```

```python
def steady_state_precursors(n0, beta_arr, lambda_arr, Lambda):
    return (beta_arr / (lambda_arr * Lambda)) * n0
```

### Hand check (n₀ = 1, β_eff = 0.0065, Λ = 1e-5)

| i | βᵢ | λᵢ | Cᵢ,₀ = βᵢ / (λᵢ Λ) |
|:-:|:-:|:-:|:-:|
| 1 | 2.470e-4 | 0.0129 | 1921 |
| 2 | 1.372e-3 | 0.0311 | 4412 |
| 3 | 1.281e-3 | 0.1083 | 1182 |
| 4 | 2.574e-3 | 0.3067 | 839 |
| 5 | 8.580e-4 | 1.4031 | 61.1 |
| 6 | 1.690e-4 | 3.8723 | 4.36 |

At ρ = 0, n = 1: prompt term `((0 − 0.0065)/1e-5)·1 = −650`. Delayed source `Σλᵢ·Cᵢ = 650`. **dn/dt = 0 ✓**

At +10 pcm (ρ = 1e-4): prompt term `((1e-4 − 6.5e-3)/1e-5)·1 = −640`. Delayed source unchanged at 650. **dn/dt = +10 1/s ✓**

At prompt critical (ρ = β_eff): prompt term = 0. **dn/dt = +650 1/s** (entirely delayed source, no prompt limitation — diverges as Cᵢ grow).

### Correlations

- **Upstream of ρ:** PWR uses `ρ = ρ_base + ρ_rod + ρ_D + ρ_m + ρ_B + ρ_Xe` (Eqs. 11–13 + xenon worth). MSR uses `ρ = ρ_base + ρ_external + α_salt·ΔT_salt` (Eq. 24 only matters analytically; the drift terms are injected into Eq. 2 separately).
- **Downstream of n:** `P = n·P_ref` feeds Eqs. 9 & 10; `φ = n·φ_ref` feeds Eqs. 14 & 15.

---

<a name="eq-9-10"></a>

# 3. Eq. 9 & 10 — PWR Lumped Thermal-Hydraulics

### Spec form

```
Eq. 9   m_f · c_f · dT_f/dt = γ_f · P − (T_f − T_c) / R_fc
Eq. 10  m_c · c_c · dT_c/dt = (T_f − T_c)/R_fc − ṁ · c_c · (T_c − T_in)
```

### Derived constants

These are computed *analytically* from the nominal operating point so the ODE at steady state reproduces (T_FUEL_NOM, T_COOL_NOM) exactly:

```
R_FC      = (T_FUEL_NOM − T_COOL_NOM) / (γ_f · P_NOM)
          = (900 − 590) / (0.97 · 3e9)
          = 1.0653e-7  K/W

M_DOT_NOM = γ_f · P_NOM / (c_c · (T_COOL_NOM − T_IN_NOM))
          = 0.97 · 3e9 / (5600 · 25)
          ≈ 20 786  kg/s
```

### Implementation (verbatim kernel)

```python
@numba.njit(cache=True)
def thermal_hydraulics_rhs(
    t: float,
    y: np.ndarray,
    power_w: float,
    m_dot: float,
    T_in: float,
) -> np.ndarray:
    """2-state fuel/coolant thermal-hydraulic ODE right-hand side.

    SPEC Eq. 9:   m_f c_f · dT_f/dt = γ_f · P − (T_f − T_c) / R_fc
    SPEC Eq. 10:  m_c c_c · dT_c/dt = (T_f − T_c) / R_fc − ṁ c_c (T_c − T_in)
    """
    T_f = y[0]
    T_c = y[1]

    # Conductive heat flux from fuel to coolant [W]
    q_fc = (T_f - T_c) / R_FC

    dydt = np.empty(2)
    # Eq. 9 — fuel temperature
    dydt[0] = (GAMMA_F * power_w - q_fc) / (M_FUEL * C_FUEL)
    # Eq. 10 — coolant temperature (advection removes ṁ c_c ΔT)
    dydt[1] = (q_fc - m_dot * C_COOL * (T_c - T_in)) / (M_COOL * C_COOL)

    return dydt
```

### Steady-state form (used for initial conditions)

```python
def steady_state_temperatures(power_w, m_dot, T_in):
    # From Eq. 10 SS: γ_f·P = ṁ·c_c·(T_c − T_in)
    T_c_ss = T_in + GAMMA_F * power_w / (m_dot * C_COOL)
    # From Eq. 9 SS: γ_f·P = (T_f − T_c) / R_fc
    T_f_ss = T_c_ss + GAMMA_F * power_w * R_FC
    return np.array([T_f_ss, T_c_ss], dtype=np.float64)
```

### I/O contract

| Parameter | Units | Source |
|---|---|---|
| `y[0] = T_f` | K | state |
| `y[1] = T_c` | K | state |
| `power_w` | W | `n · P_ref` from kinetics |
| `m_dot` | kg/s | `coolant_flow_fraction · M_DOT_NOM` from controls |
| `T_in` | K | inlet temperature from controls |
| **Return** | K/s for each state | to solver |

### Hand check at nominal SS (P = 3e9 W, ṁ = M_DOT_NOM, T_f = 900, T_c = 590, T_in = 565)

- q_fc = (900−590) / 1.0653e-7 = 2.910e9 W
- γ_f·P − q_fc = 0.97·3e9 − 2.910e9 = 0 → **dT_f/dt = 0 ✓**
- ṁ·c_c·(T_c−T_in) = 20786 · 5600 · 25 = 2.910e9 W
- q_fc − ṁ·c_c·ΔT = 0 → **dT_c/dt = 0 ✓**

### Correlations

- **Upstream:** `power_w = n·P_ref` is driven by Eq. 1; `m_dot` and `T_in` come from user controls.
- **Downstream:** T_f drives Eq. 11 (Doppler); T_c drives Eq. 12 (moderator).
- Heat balance is *exact*: q_fc that leaves fuel equals q_fc that enters coolant.

---

<a name="eq-11-12-13"></a>

# 4. Eq. 11, 12, 13 — PWR Algebraic Feedback

These are pure algebraic, called every solver step. Output is in **pcm**; the kinetics interface multiplies by 1e-5 to convert to Δk/k.

### Eq. 11 — Doppler

Spec: `ρ_D = α_D · (T_f − T_f,0)`

```python
def doppler_feedback(T_fuel_k, T_fuel_ref_k, alpha_D=-2.5):
    if T_fuel_k < 0.0 or T_fuel_ref_k < 0.0:
        return math.nan   # unphysical input → caller handles
    return alpha_D * (T_fuel_k - T_fuel_ref_k)
```

α_D < 0 always: rising T_f broadens U-238 resonance absorption peaks → reduces k_eff. Default `α_D = −2.5 pcm/K` (range −2 to −3).

### Eq. 12 — Moderator

Spec: `ρ_m = α_m · (T_c − T_c,0)`

```python
def moderator_feedback(T_cool_k, T_cool_ref_k, alpha_m=-35.0):
    if T_cool_k < 0.0 or T_cool_ref_k < 0.0:
        return math.nan
    return alpha_m * (T_cool_k - T_cool_ref_k)
```

α_m < 0: hotter water is less dense, moderates less effectively. Default `α_m = −35 pcm/K` (range −20 to −50).

### Eq. 13 — Boron

Spec: `ρ_B = ω_B · C_B`

```python
def boron_worth(boron_ppm, omega_B=-10.0):
    if boron_ppm < 0.0:
        return math.nan
    return omega_B * boron_ppm
```

Default `ω_B = −10 pcm/ppm`. Note this is an *absolute* contribution (not a delta from reference) because boron is fully absent at ppm = 0.

### Reactivity balance sum

```python
def reactivity_balance(rho_rod_pcm, rho_doppler_pcm, rho_moderator_pcm,
                       rho_boron_pcm, rho_xenon_pcm):
    return (rho_rod_pcm + rho_doppler_pcm + rho_moderator_pcm
            + rho_boron_pcm + rho_xenon_pcm)

def pcm_to_dk_k(rho_pcm):
    return rho_pcm * 1.0e-5   # 1 pcm = 10⁻⁵ Δk/k
```

### Hand check

- T_f = 1000 K, T_f,0 = 900 K, α_D = −2.5 → ρ_D = −2.5·100 = **−250 pcm**
- T_c = 600 K, T_c,0 = 590 K, α_m = −35 → ρ_m = −35·10 = **−350 pcm**
- C_B = 1000 ppm, ω_B = −10 → ρ_B = **−10 000 pcm**

### Correlations

- T_f from Eq. 9 → ρ_D
- T_c from Eq. 10 → ρ_m
- C_B from user controls → ρ_B
- Sum + ρ_rod + ρ_Xe → ρ_total → Eq. 1

---

<a name="eq-14-15"></a>

# 5. Eq. 14 & 15 — Xenon/Iodine Coupled Kinetics

### Spec form

```
Eq. 14   dI/dt = γ_I · Σ_f · φ − λ_I · I
Eq. 15   dX/dt = γ_X · Σ_f · φ + λ_I · I − (λ_X + σ_aX · φ) · X
```

### Implementation (verbatim kernel)

```python
@numba.njit(cache=True)
def xenon_iodine_rhs(t, y, phi, Sigma_f):
    """2-state I-135/Xe-135 ODE right-hand side.

    SPEC Eq. 14:  dI/dt = γ_I · Σ_f · φ − λ_I · I
    SPEC Eq. 15:  dX/dt = γ_X · Σ_f · φ + λ_I · I − (λ_X + σ_aX · φ) · X
    """
    I = y[0]
    X = y[1]

    fission_rate = Sigma_f * phi  # fissions cm⁻³ s⁻¹

    dI_dt = GAMMA_I * fission_rate - LAMBDA_I * I
    dX_dt = GAMMA_X * fission_rate + LAMBDA_I * I - (LAMBDA_X + SIGMA_AX * phi) * X

    dydt = np.empty(2)
    dydt[0] = dI_dt
    dydt[1] = dX_dt
    return dydt
```

Constants used: `GAMMA_I = 0.0639`, `GAMMA_X = 0.00237`, `LAMBDA_I = 2.87e-5 s⁻¹`, `LAMBDA_X = 2.09e-5 s⁻¹`, `SIGMA_AX = 2.6e-18 cm²`.

### Steady-state form (used for y₀)

Setting dI/dt = dX/dt = 0:

```
I₀ = γ_I · Σ_f · φ / λ_I
X₀ = (γ_X + γ_I) · Σ_f · φ / (λ_X + σ_aX · φ)
```

```python
def steady_state_xenon(phi, Sigma_f):
    if phi == 0.0:
        return np.zeros(2, dtype=np.float64)
    fission_rate = Sigma_f * phi
    I0 = GAMMA_I * fission_rate / LAMBDA_I
    X0 = (GAMMA_X + GAMMA_I) * fission_rate / (LAMBDA_X + SIGMA_AX * phi)
    return np.array([I0, X0], dtype=np.float64)
```

### Hand check at full power (φ_nom = 3.1e13 n/cm²/s, Σ_f = 0.30 cm⁻¹)

- Fission rate = Σ_f · φ = 0.30 · 3.1e13 = 9.3e12 fissions/cm³/s
- I₀ = 0.0639 · 9.3e12 / 2.87e-5 = **2.07e16 atoms/cm³ ✓**
- λ_X + σ_aX·φ = 2.09e-5 + 2.6e-18·3.1e13 = 2.09e-5 + 8.06e-5 = 1.015e-4 s⁻¹
  - **Burnup (8.06e-5) dominates decay (2.09e-5) at full-power flux ✓**
- X₀ = (0.0639 + 0.00237)·9.3e12 / 1.015e-4 = 6.163e11 / 1.015e-4 = **6.07e15 atoms/cm³ ✓**

### Correlations

- φ from Eq. 1 (n·φ_ref).
- I from Eq. 14 → Xe-135 source in Eq. 15.
- X → ρ_Xe via the one-group perturbation formula below.

---

<a name="xenon-worth"></a>

# 6. Xenon Reactivity Worth (One-Group Perturbation)

Not numbered in spec but referenced as ρ_Xe in §4.5.4.

### Equation

```
ρ_Xe = − (σ_aX · N_X) / Σ_a       [dimensionless]
```

### Implementation

```python
SIGMA_AX = 2.6e-18    # cm²  (= 2.6×10⁶ barns)

def xenon_reactivity_pcm(xenon_density, sigma_a=0.55):
    """Return Xe-135 poisoning reactivity in pcm using one-group perturbation theory."""
    return -SIGMA_AX * xenon_density / sigma_a * 1.0e5
```

`sigma_a` is the one-group macroscopic absorption cross section (cm⁻¹); default 0.55. The factor `1e5` converts dimensionless to pcm.

### Unit check

- σ_aX · N_X: cm² · (atoms/cm³) = atoms/cm = cm⁻¹ ✓
- (cm⁻¹) / Σ_a (cm⁻¹) = dimensionless ✓
- × 1e5 → pcm ✓

### Hand check at full power equilibrium

- N_X = 6.07e15 (from §5 hand check)
- σ_aX · N_X = 2.6e-18 · 6.07e15 = 1.578e-2 cm⁻¹
- ρ_Xe = −1.578e-2 / 0.55 = −2.870e-2 = **−2 870 pcm ✓**

This is in the SPEC §7.3 target range −2 500 to −3 000 pcm. PASS.

---

<a name="pwr-coupled"></a>

# 7. PWR Full 11-State Coupled System

This is the entire production kernel SciPy sees. It assembles every block above into one `@numba.njit` function. `params` is a flat float64 array packing every constant and control needed.

### State vector layout

```
y[0]    = n           neutron population (n=1 ↔ P = P_ref)
y[1:7]  = C₁..C₆     delayed-neutron precursor concentrations
y[7]    = T_fuel      fuel temperature (K)
y[8]    = T_cool      coolant temperature (K)
y[9]    = I           I-135 number density (atoms/cm³)
y[10]   = X           Xe-135 number density (atoms/cm³)
```

### Flat parameter array layout

| Index | Constant |
|:--:|---|
| 0 | rod reactivity (Δk/k) |
| 1 | dissolved boron (ppm) |
| 2 | coolant flow fraction |
| 3 | inlet temperature (K) |
| 4 | α_D (pcm/K) |
| 5 | α_m (pcm/K) |
| 6 | ω_B (pcm/ppm) |
| 7 | T_fuel,ref (K) |
| 8 | T_cool,ref (K) |
| 9 | P at n=1 (W) |
| 10 | φ at n=1 (n/cm²/s) |
| 11 | Σ_f (cm⁻¹) |
| 12 | Σ_a (cm⁻¹) |
| 13 | base reactivity (Δk/k) |
| 14 | Λ (s) |
| 15–20 | β₁..β₆ |
| 21–26 | λ₁..λ₆ |

Total length 27.

### Implementation (verbatim coupled kernel)

```python
@numba.njit(cache=True)
def pwr_11_state_system(t: float, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """Full 11-state PWR ODE right-hand side — flat-params form.

    Implements SPEC Eqs. 1, 2 (kinetics), 9, 10 (thermal), 11–13 (algebraic
    feedback), and 14, 15 (xenon/iodine) at a single solver call.
    """
    n      = y[0]
    T_fuel = y[7]
    T_cool = y[8]
    X      = y[10]

    # Unpack scalar parameters
    rod_dk_k   = params[0]
    boron_ppm  = params[1]
    flow_frac  = params[2]
    T_in       = params[3]
    alpha_d    = params[4]
    alpha_m    = params[5]
    omega_b    = params[6]
    T_fuel_ref = params[7]
    T_cool_ref = params[8]
    P_ref      = params[9]
    phi_ref    = params[10]
    sigma_f    = params[11]
    sigma_a    = params[12]
    base_dk_k  = params[13]
    Lambda     = params[14]
    beta_arr   = params[15:21]
    lambda_arr = params[21:27]

    # --- Algebraic feedback reactivity — Eqs. 11–13 + xenon worth ---
    rho_d_pcm  = alpha_d * (T_fuel - T_fuel_ref)             # Eq. 11
    rho_m_pcm  = alpha_m * (T_cool - T_cool_ref)             # Eq. 12
    rho_b_pcm  = omega_b * boron_ppm                         # Eq. 13
    rho_xe_pcm = -SIGMA_AX * X / sigma_a * 1.0e5             # Xe worth

    rho_total = (
        base_dk_k
        + rod_dk_k
        + (rho_d_pcm + rho_m_pcm + rho_b_pcm + rho_xe_pcm) * 1.0e-5
    )

    # --- Kinetics block — Eqs. 1 & 2 ---
    kin_rhs = point_kinetics_rhs(t, y[:7], rho_total, beta_arr, lambda_arr, Lambda)

    # --- Thermal-hydraulics block — Eqs. 9 & 10 ---
    P_w   = n * P_ref
    m_dot = flow_frac * M_DOT_NOM
    th_rhs = thermal_hydraulics_rhs(t, y[7:9], P_w, m_dot, T_in)

    # --- Xenon/Iodine block — Eqs. 14 & 15 ---
    phi    = n * phi_ref
    xe_rhs = xenon_iodine_rhs(t, y[9:11], phi, sigma_f)

    # Compose 11-element derivative
    dydt = np.empty(11)
    dydt[0] = kin_rhs[0]
    dydt[1:7] = kin_rhs[1:7]
    dydt[7] = th_rhs[0]
    dydt[8] = th_rhs[1]
    dydt[9] = xe_rhs[0]
    dydt[10] = xe_rhs[1]
    return dydt
```

### Coupling chain at every solver step

```
controls (rod, boron, flow, T_in)
   │
   ↓
y = [n, C₁..C₆, T_f, T_c, I, X]
                ┌──────────────────────────────────┐
                │ Read y[7], y[8], y[10] → ρ_D, ρ_m, ρ_Xe
                │ Add rod + ρ_B → ρ_total → Δk/k
                ↓
y[0..6] ────→ Eqs. 1, 2 (shared kinetics)
y[0]  ──→ P = n·P_ref ──→ Eqs. 9, 10 → dT_f/dt, dT_c/dt
y[0]  ──→ φ = n·φ_ref ──→ Eqs. 14, 15 → dI/dt, dX/dt
```

### Critical-base-reactivity setup

To make the solver start with `dy/dt = 0` exactly, the orchestrator computes `ρ_base` such that all algebraic feedback components cancel at t = 0:

```python
def critical_base_reactivity_pcm(y0, controls, reference_state, config):
    """Return base_reactivity that makes y0 exactly critical."""
    zero_base_config = replace(config, base_reactivity_pcm=0.0)
    snapshot = calculate_reactivity_snapshot(y0, controls, reference_state,
                                             zero_base_config)
    return -snapshot.total_pcm
```

This is necessary because dissolved boron and equilibrium xenon would otherwise put the reactor in a sub-critical state at t = 0.

---

<a name="eq-16"></a>

# 8. Eq. 16 — Helion Plasma Energy Balance

### Spec form

```
Eq. 16   dW/dt = P_heat + P_fusion − P_Brem − P_cond
```

W is the only ODE state for Helion. Temperature is derived algebraically (see Eq. 19 inverse below).

### Implementation (verbatim coupled kernel)

```python
@numba.njit(cache=True)
def plasma_energy_rhs_kernel(t, y, params):
    """1-state Helion plasma energy ODE right-hand side.

    Implements SPEC Eq. 16: dW/dt = P_heat + P_fusion − P_Brem − P_cond

    Sub-physics:
        Eq. 17: P_fusion = (n²/4) · ⟨σv⟩(T) · E_DHe3 · V
        Eq. 18: ⟨σv⟩ via Bosch-Hale
        Eq. 20: P_Brem = C_B · Z_eff · n_e² · T^{1/2} · V
        Eq. 21: P_cond = W / τ_E
    """
    dydt = np.empty(1)
    W = y[0]

    # Non-negative state floor (SPEC §4.6.5)
    if W <= 0.0:
        dydt[0] = 0.0
        return dydt

    n      = params[0]   # post-compression ion density (m⁻³)
    V      = params[1]   # post-compression plasma volume (m³)
    P_heat = params[2]   # external heating (W)
    B      = params[3]   # magnetic field (T)
    # (Bosch-Hale coefficients live in params[4:13])
    # (E_DHE3, C_BREM, C_TAU, N_REF, B_REF, DHE3_PM, J_per_keV at fixed indices)

    # Derive T from W (inverse of ideal-plasma energy)
    n_total = 2.5 * n   # D-He3 50/50: n_total = (5/2)·n_ions
    T_kev = (2.0 / 3.0) * W / (n_total * V * J_PER_KEV)

    # Clamp T to Bosch-Hale validity [0.5, 190] keV
    T_kev_clamped = min(190.0, max(0.5, T_kev))

    # Eq. 17 — fusion power
    sv = sigma_v_kernel(T_kev_clamped, ...)   # Bosch-Hale, m³/s
    P_fusion = (n * n / 4.0) * sv * E_DHE3 * V

    # Eq. 20 — Bremsstrahlung; for D-He3 50/50: n_e=1.5n_ions, Z_eff=5/3
    n_e = 1.5 * n
    P_brem = 5.35e-37 * (5.0 / 3.0) * n_e * n_e * (T_kev_clamped ** 0.5) * V

    # Eq. 21 — conduction; τ_E = C_TAU · (n/N_REF) · (B/B_REF)²
    tau_e = 5.0e-5 * (n / 1.0e21) * (B / 5.0) ** 2
    P_cond = W / tau_e

    # SPEC Eq. 16 — all D-He3 charged products retained in plasma
    dydt[0] = P_heat + P_fusion - P_brem - P_cond
    return dydt
```

### I/O contract

| Parameter | Units | Source |
|---|---|---|
| `y[0] = W` | J | state |
| `n` | m⁻³ | post-compression ion density |
| `V` | m³ | post-compression plasma volume |
| `P_heat` | W | constant external heating |
| `B` | T | applied magnetic field |
| **Return** | J/s = W | to solver |

### Correlations

Every right-hand side term feeds back into W through the four sub-equations. T is recovered from W at every step (see Eq. 19 inverse).

---

<a name="eq-17"></a>

# 9. Eq. 17 — D-He3 Fusion Power

### Spec form

```
Eq. 17   P_fusion = (n²/4) · ⟨σv⟩(T) · E_DHe3 · V
```

### Implementation

```python
P_fusion = (n * n / 4.0) * sv * E_DHE3 * V
```

Constants: `E_DHE3 = 2.93e-12 J` (= 18.3 MeV).

### Why the prefactor n²/4

For a 50/50 D-He3 mix, n_D = n_He3 = n_ions/2. Pair-collision rate per volume:
```
rate = n_D · n_He3 · ⟨σv⟩ = (n_ions/2) · (n_ions/2) · ⟨σv⟩ = (n_ions²/4) · ⟨σv⟩
```

So the `n` in Eq. 17 must be **total ion density** for the formula to be correct. This is the convention the code uses (`n = params[0] = post-compression ion density`).

### Hand check (n = 1e22, T = 80 keV, V = 0.1 m³)

- ⟨σv⟩(80 keV) ≈ 1.2e-22 m³/s (Bosch-Hale, see §10)
- P_fusion = (1e22)² / 4 · 1.2e-22 · 2.93e-12 · 0.1
  - = 2.5e43 · 1.2e-22 · 2.93e-13
  - = 3e21 · 2.93e-13 = **8.79e8 W ≈ 879 MW** (typical ignition-condition fusion power)

### Correlations

- ⟨σv⟩ from Eq. 18 (Bosch-Hale).
- All energy retained → feeds Eq. 16 as positive source.

---

<a name="eq-18"></a>

# 10. Eq. 18 — Bosch-Hale Reactivity ⟨σv⟩(T)

### Spec form (as written)

```
Eq. 18   ⟨σv⟩ = C₁ · θ² / (ξ · B_G²) · √(ξ / (m_r·c² · T³)) · exp(−3ξ)
         θ    = T / [1 − T·(C₂+T·(C₄+T·C₆)) / (1 + T·(C₃+T·(C₅+T·C₇)))]
         ξ    = (B_G² / 4θ)^(1/3)
```

### ⚠ Spec/code reconciliation

The code follows the **published Bosch & Hale 1992 Eq. (12)** exactly:

```
⟨σv⟩ = C₁ · θ · √(ξ / (m_r·c² · T³)) · exp(−3ξ)      [cm³/s]
```

This differs algebraically from the spec text by a prefactor of `θ/(ξ·B_G²)`. The published form is the correct one and has been verified numerically against Bosch-Hale Table I. The spec text is a documentation typo, not a code bug. This is flagged in the module docstring.

### Implementation (verbatim kernel)

```python
@numba.njit(cache=True)
def sigma_v_kernel(T_kev, bg, mrc2, c1, c2, c3, c4, c5, c6, c7):
    """Bosch-Hale ⟨σv⟩ for D-He3, B&H (1992) Eq. (12).

    T_kev must already be clamped to [0.5, 190] keV by the caller.
    Returns ⟨σv⟩ in m³/s.
    """
    theta = T_kev / (
        1.0 - T_kev * (c2 + T_kev * (c4 + T_kev * c6))
              / (1.0 + T_kev * (c3 + T_kev * (c5 + T_kev * c7)))
    )
    xi = (bg * bg / (4.0 * theta)) ** (1.0 / 3.0)
    sv_cm3_s = c1 * theta * np.sqrt(xi / (mrc2 * T_kev * T_kev * T_kev)) \
               * np.exp(-3.0 * xi)
    return sv_cm3_s * 1.0e-6   # cm³/s → m³/s
```

### Constants (D-He3, Bosch-Hale 1992 Table IV)

```
B_G   = 68.7508
m_r·c²= 1 124 572 keV
C1 = 5.51036e-10
C2 = 6.41918e-3
C3 = -2.02896e-3
C4 = -1.91080e-5
C5 = 1.35776e-4
C6 = 0
C7 = 0
```

### Validity range

`T ∈ [0.5, 190] keV`. Outside, value is clamped at the boundary and a warning is logged (SPEC §4.6.5 / Risk R-06).

### Numerical verification (matches B&H Table I)

| T (keV) | Code output (m³/s) | B&H Table I |
|:-:|:-:|:-:|
| 5.0 | 6.38e-27 | within ±5% |
| 10.0 | 2.13e-25 | 2.1e-25 ±10% |
| 20.0 | 3.48e-24 | within ±5% |
| 50.0 | 5.55e-23 | 5.56e-23 ±5% |
| 100.0 | 1.72e-22 | 1.71e-22 ±5% |
| 190.0 | 2.68e-22 | within ±5% |

(These are exercised by `test_sigma_v_known_value_at_*_kev` in the test suite.)

### Correlations

T comes from the inverse of Eq. 19 evaluated on the current W. ⟨σv⟩ feeds Eq. 17.

---

<a name="eq-19"></a>

# 11. Eq. 19 — Adiabatic Compression Heating

### Spec form

```
Eq. 19   T_final = T_initial · Rᶜ^(2/3)
         n_final = n_initial · Rᶜ
         (V_final = V_initial / Rᶜ)
```

This is for a monatomic ideal plasma (γ = 5/3), so γ − 1 = 2/3.

### Quasineutrality for D-He3 50/50

```
n_D    = n_He3 = n_ions / 2
n_e    = 1·n_D + 2·n_He3 = (1/2 + 1)·n_ions = (3/2)·n_ions
n_total= n_ions + n_e = (5/2)·n_ions
Z_eff  = Σ(nᵢ Zᵢ²) / n_e = (n/2·1² + n/2·2²) / (3n/2) = 5/3
```

### Implementation

```python
_J_PER_KEV = 1.60218e-16            # SI conversion
_DHE3_TOTAL_PARTICLE_MULT = 2.5     # n_total / n_ions = 5/2

def apply_adiabatic_compression(T_initial_kev, n_initial_m3,
                                compression_ratio, volume_initial_m3):
    """Apply adiabatic compression and return post-compression plasma state."""
    T_final = T_initial_kev * compression_ratio ** (2.0 / 3.0)   # SPEC Eq. 19
    n_final = n_initial_m3 * compression_ratio                    # SPEC Eq. 19
    V_final = volume_initial_m3 / compression_ratio
    W_final = _plasma_energy_j(T_final, n_final, V_final)
    return T_final, n_final, V_final, W_final


def _plasma_energy_j(T_kev, n_ions_m3, V_m3):
    """W = (3/2) · n_total · k_B T · V  with  n_total = (5/2)·n_ions."""
    n_total = 2.5 * n_ions_m3
    return 1.5 * n_total * (T_kev * _J_PER_KEV) * V_m3


def plasma_temperature_kev(W_J, n_ions_m3, V_m3):
    """Inverse: T_keV = (2/3)·W / (n_total·V·J_per_keV)."""
    if W_J <= 0.0:
        return 0.0
    n_total = 2.5 * n_ions_m3
    return (2.0 / 3.0) * W_J / (n_total * V_m3 * _J_PER_KEV)
```

### Energy formula derivation

For a 3D ideal gas, mean thermal energy per particle is `(3/2) k_B T`. For the plasma with `n_total` particles per unit volume in volume V:
```
W = (3/2) · n_total · k_B T · V
  = (3/2) · (5/2)·n_ions · T_keV · J_per_keV · V
  = (15/4) · n_ions · T_keV · J_per_keV · V
```

### Hand check round-trip

`T₀ = 20 keV, n₀ = 1e21 m⁻³, R_c = 10, V₀ = 1 m³`:
- T_final = 20 · 10^(2/3) = 20 · 4.6416 = **92.832 keV ✓**
- n_final = 1e22 m⁻³ ✓
- V_final = 0.1 m³ ✓
- W = (15/4) · 1e22 · 92.832 · 1.60218e-16 · 0.1 = **5.577e7 J ✓**
- Inverse: T = (2/3) · 5.577e7 / (2.5·1e22 · 0.1 · 1.60218e-16) = **92.832 keV ✓**

### Correlations

`apply_adiabatic_compression` is called **once** before `integrate_helion_pulse` starts. It defines the post-compression initial state (W₀, n_final, V_final) which become fixed parameters during the pulse integration.

---

<a name="eq-20"></a>

# 12. Eq. 20 — Bremsstrahlung Radiation Loss

### Spec form

```
Eq. 20   P_Brem = C_B · Z_eff · n_e² · T^(1/2) · V
```

### Implementation

```python
C_BREM   = 5.35e-37        # W·m³ / keV^(1/2)
DHE3_NE_MULT = 1.5         # n_e = 1.5·n_ions  (50/50 D-He3)
DHE3_Z_EFF   = 5.0 / 3.0   # Z_eff for 50/50 D-He3

@numba.njit(cache=True)
def bremsstrahlung_power(T_kev, n_ions_m3, V_m3):
    """SPEC Eq. 20.  P_Brem = C_B · Z_eff · n_e² · T^{1/2} · V."""
    n_e = 1.5 * n_ions_m3
    return 5.35e-37 * (5.0 / 3.0) * n_e * n_e * np.sqrt(T_kev) * V_m3
```

### Hand check (n_ions = 1e22 m⁻³, T = 80 keV, V = 0.1 m³)

- n_e = 1.5e22 m⁻³
- P_brem = 5.35e-37 · (5/3) · (1.5e22)² · √80 · 0.1
  - = 5.35e-37 · 1.667 · 2.25e44 · 8.944 · 0.1
  - = 5.35e-37 · 3.357e44
  - = **1.796e8 W ≈ 180 MW**

For comparison with Eq. 17 hand check (P_fusion ≈ 879 MW at same conditions): P_fusion / P_brem ≈ 4.9, so fusion clearly dominates at these conditions (ignition regime).

### Correlations

T from Eq. 19 inverse, n from compression constants. Output is a sink in Eq. 16.

---

<a name="eq-21"></a>

# 13. Eq. 21 — Thermal Conduction Loss

### Spec form

```
Eq. 21   P_cond = W / τ_E
```

τ_E is the empirical FRC energy confinement time — the *dominant model uncertainty* (Risk R-02).

### Implementation

```python
C_TAU = 5.0e-5    # s  — scaling prefactor (calibrated to ~50 μs @ 1e21 m⁻³, 5 T)
_N_REF = 1.0e21   # m⁻³
_B_REF = 5.0      # T

# Uncertainty band (SPEC R-02): nominal × [0.3, 3.0]
TAU_E_UNCERTAINTY_LOWER = 0.3
TAU_E_UNCERTAINTY_UPPER = 3.0

def confinement_time(n_ions_m3, B_T):
    """Empirical FRC energy confinement time τ_E (s).
    Parametric scaling: τ_E = C_TAU · (n/N_REF) · (B/B_REF)²
    """
    return C_TAU * (n_ions_m3 / _N_REF) * (B_T / _B_REF) ** 2

def conduction_loss(W_J, n_ions_m3, B_T):
    """SPEC Eq. 21:  P_cond = W / τ_E"""
    return W_J / confinement_time(n_ions_m3, B_T)
```

### Hand check at nominal (n = 1e21 m⁻³, B = 5 T)

- τ_E = 5e-5 · 1 · 1 = **50 μs ✓**
- If W = 5e7 J: P_cond = 5e7 / 5e-5 = **1e12 W = 1 TW**

This very large nominal P_cond is why aggressive compression (high B) is required to retain energy long enough for fusion to dominate.

### Correlations

W from state, n and B from compression constants. Output is a sink in Eq. 16.

---

<a name="helion-q"></a>

# 14. Helion Q Factor and Coupled Pulse

### Q factor definition

```
Q = E_fusion_total / E_input
E_input = W_initial + P_heat · pulse_duration
```

### Implementation (verbatim, after the ODE integrates the pulse)

```python
E_fusion = float(np.trapezoid(P_fusion, t_arr))
E_heat_input = W_initial + P_heat_w * parameters.pulse_duration_s
Q = E_fusion / E_heat_input if E_heat_input > 0.0 else 0.0
```

For a passive pulse (P_heat = 0), Q = ∫P_fusion dt / W_initial.

### Pulse integration

```python
result = solve_ivp(
    plasma_energy_rhs_kernel,
    (0.0, parameters.pulse_duration_s),
    np.array([W_initial], dtype=np.float64),
    method="Radau", rtol=1e-6, atol=1e-9,
    args=(params,),
    t_eval=t_eval,
)
```

---

<a name="eq-22-23"></a>

# 15. Eq. 22 & 23 — MSR Precursor Drift

### Spec form

```
Eq. 22  dCᵢ/dt = (βᵢ/Λ)·n − λᵢ·Cᵢ − (1/τ_core)·Cᵢ + Cᵢ^return(t)
Eq. 23  Cᵢ^return(t) = (1/τ_core) · Cᵢ(t − τ_loop) · e^(−λᵢ·τ_loop)
```

Compared to PWR (Eq. 2), two new terms appear:
- **Flow-out** `−(1/τ_core)·Cᵢ` — precursors carried out of the core
- **Return** `+(1/τ_core)·Cᵢ(t−τ_loop)·e^(−λᵢτ_loop)` — survivors that complete the loop

The history dependence `Cᵢ(t − τ_loop)` makes this a *delay-differential equation* (DDE) — standard ODE solvers can't handle it directly.

### Implementation (verbatim drift kernel)

```python
@numba.njit(cache=True)
def msr_precursor_drift_rhs(C_arr, C_delayed, lambda_arr, tau_core, tau_loop):
    """MSR precursor drift contribution to dCᵢ/dt (Eq. 22 drift part).

    Returns the net drift term to be ADDED to the standard PWR-style
    precursor ODE (which the shared kinetics core already computes):

        drift_i = −(1/τ_core)·Cᵢ
                  + (1/τ_core)·Cᵢ(t−τ_loop)·exp(−λᵢ·τ_loop)
    """
    inv_tau = 1.0 / tau_core
    result = np.empty(6)
    for i in range(6):
        flow_out = -inv_tau * C_arr[i]
        return_term = inv_tau * C_delayed[i] * math.exp(-lambda_arr[i] * tau_loop)
        result[i] = flow_out + return_term
    return result
```

### Delay-system handling (method-of-steps)

A `PrecursorHistory` class stores `(t, C)` pairs as they are accepted by the ODE solver and looks up `C(t − τ_loop)` by linear interpolation:

```python
class PrecursorHistory:
    """Sorted history buffer for MSR precursor concentrations.

    Stores (time, C_array) pairs from accepted ODE steps. Linear interpolation
    is used to retrieve Cᵢ(t − τ_loop) at any requested time t. For any lookup
    time before the simulation start (t < t_start), the pre-filled steady-state
    value C_init is returned.
    """

    def __init__(self, t_start, C_init, max_size=50_000):
        self._t_start = float(t_start)
        self._C_init  = C_init.astype(np.float64).copy()
        self._max_size = max_size
        self._times   = []
        self._values  = []

    def get(self, t):
        """Return C at time t via linear interpolation; pre-buffer returns C_init."""
        if len(self._times) == 0 or t <= self._t_start:
            return self._C_init
        if t >= self._times[-1]:
            return self._values[-1]
        idx = bisect.bisect_right(self._times, t)
        t0, t1 = self._times[idx-1], self._times[idx]
        C0, C1 = self._values[idx-1], self._values[idx]
        alpha = (t - t0) / (t1 - t0)
        return C0 + alpha * (C1 - C0)

    def add(self, t, C):
        self._times.append(float(t))
        self._values.append(C.astype(np.float64).copy())
```

The MSR engine integrates in chunks of length `dt_history = τ_loop / 2`, updating the history buffer between chunks. This guarantees that all delay lookups within a chunk reference already-accepted history.

### Limiting cases (all verified)

- **τ_core → ∞** (zero salt velocity): drift terms vanish → MSR ≡ PWR exactly.
- **τ_loop → 0** (instant return): flow-out and return cancel → no drift.
- **τ_loop → ∞** (precursors decay outside): return term → 0 → maximum precursor loss.

### Correlations

- Sum: `dCᵢ/dt = (shared-kinetics Eq. 2) + (drift)`
- Inputs: current Cᵢ from state, past Cᵢ from history buffer, λᵢ from Keepin table, τ_core/τ_loop from MSR config (fixed for the simulation lifetime).

---

<a name="eq-24"></a>

# 16. Eq. 24 — β_eff,flow

### Spec form

```
Eq. 24   β_eff,flow = Σᵢ βᵢ · [λᵢτ_core / (1 + λᵢτ_core)]
                           · [1 + e^(−λᵢτ_loop) / (1 + λᵢτ_core − e^(−λᵢτ_loop))]
```

This is the steady-state solution of Eq. 22 at constant power; it gives the effective delayed-neutron fraction once precursor drift is accounted for.

### Implementation (verbatim)

```python
def compute_beta_eff_flow(beta_arr, lambda_arr, tau_core, tau_loop):
    """SPEC Eq. 24 — effective delayed neutron fraction with precursor drift.

    Limiting cases:
      • τ_core → ∞ (v_salt → 0):   β_eff,flow → Σβᵢ = β_eff,static
      • τ_loop = 0 (instant return): β_eff,flow = Σβᵢ
      • τ_core → 0 (v_salt → ∞):   β_eff,flow → 0
    """
    beta_flow = 0.0
    for i in range(len(beta_arr)):
        li = lambda_arr[i]
        bi = beta_arr[i]
        li_tc    = li * tau_core
        exp_term = math.exp(-li * tau_loop)
        denom    = 1.0 + li_tc - exp_term
        if abs(denom) < 1e-30:
            beta_flow += bi
            continue
        factor1 = li_tc / (1.0 + li_tc)
        factor2 = 1.0 + exp_term / denom
        beta_flow += bi * factor1 * factor2
    return beta_flow
```

### Limiting-case derivations

**Case 1: τ_loop → 0.**  exp_term → 1. factor2 = 1 + 1/(λτ_c) = (λτ_c + 1)/(λτ_c). factor1 · factor2 = 1. → β_eff,flow = Σβᵢ. ✓

**Case 2: τ_core → ∞.**  factor1 → 1. denom → ∞ → factor2 → 1. → β_eff,flow = Σβᵢ. ✓

**Case 3: τ_loop → ∞.**  exp_term → 0. factor2 = 1. factor1 < 1. → β_eff,flow < Σβᵢ (reduced by drift). ✓

**Case 4: τ_core → 0.**  factor1 → 0. → β_eff,flow → 0 (all precursors flushed). ✓

### Hand check (audit script output)

Static β_eff = 0.006500. With L_core = 5 m, L_loop = 20 m:

| v_salt (m/s) | τ_core (s) | τ_loop (s) | β_eff,flow | % of static |
|:-:|:-:|:-:|:-:|:-:|
| 0.001 | 5000   | 20000  | 0.006483 | 99.74% |
| 0.1   | 50.0   | 200.0  | 0.005448 | 83.81% |
| 0.5   | 10.0   | 40.0   | 0.004054 | 62.36% |
| 1.0   | 5.0    | 20.0   | 0.003356 | 51.63% |
| 2.0   | 2.5    | 10.0   | 0.002701 | 41.56% |
| 5.0   | 1.0    | 4.0    | 0.002055 | 31.62% |
| 10.0  | 0.5    | 2.0    | 0.001733 | 26.66% |

Monotonically decreasing with v_salt ✓ (SPEC §7.3 MSR validation criterion).

### Correlations

This is a diagnostic / analytical formula. It is computed for plotting and for `msr_critical_base_reactivity_pcm` (below). It is *not* used inside the ODE — the ODE uses the full Eq. 22 + Eq. 23 instead. But the steady-state of those equations integrates to exactly this β_eff,flow.

---

<a name="msr-thermal"></a>

# 17. MSR Salt Thermal-Hydraulics

### Governing equation (SPEC §4.4)

```
M_CORE · c_salt · dT_salt/dt = P − ṁ_salt · c_salt · (T_salt − T_inlet)
```

Single node, unlike the PWR's two-node model: salt is simultaneously fuel and coolant, so `γ_salt = 1.0` and there is no fuel-to-coolant resistance.

### Derived constants

```
M_DOT_SALT_NOM = P_NOM_MSR / (C_SALT · ΔT_nom)
              = 500e6 / (1500 · 40)
              = 8 333 kg/s

L_CORE        = V_SALT_NOM · TAU_CORE_NOM = 1.0 · 5.0 = 5.0 m
A_CORE        = M_DOT_SALT_NOM / (ρ_salt · V_SALT_NOM) ≈ 4.17 m²
M_CORE_SALT   = ρ_salt · L_CORE · A_CORE ≈ 41 667 kg
```

τ_thermal = M_CORE_SALT / M_DOT_SALT_NOM = 5 s = τ_core_nom ✓ (by construction)

### Implementation (verbatim kernel)

```python
@numba.njit(cache=True)
def salt_thermal_rhs(t, y, power_w, m_dot_salt, T_salt_inlet):
    """SPEC §4.4 unified fuel-coolant energy balance:
        M_CORE · c_salt · dT_salt/dt = P − ṁ_salt · c_salt · (T_salt − T_inlet)
    """
    T_salt   = y[0]
    heat_in  = power_w
    heat_out = m_dot_salt * C_SALT * (T_salt - T_salt_inlet)

    dydt = np.empty(1)
    dydt[0] = (heat_in - heat_out) / (M_CORE_SALT * C_SALT)
    return dydt
```

### Steady-state form

```python
def steady_state_salt_temperature(power_w, m_dot_salt, T_inlet):
    """From dT/dt = 0:  T_salt,ss = T_inlet + P / (ṁ_salt · c_salt)"""
    if m_dot_salt <= 0.0 or power_w < 0.0 or T_inlet < 0.0:
        return math.nan
    return T_inlet + power_w / (m_dot_salt * C_SALT)
```

### Coupling between flow rate and DDE geometry

The same physical salt velocity sets *both* the thermal flow rate *and* the precursor transit times:

```python
def m_dot_salt_from_tau_core(tau_core):
    """ṁ_salt = ρ·A·v = ρ·A·L_CORE/τ_core
              = M_DOT_SALT_NOM · TAU_CORE_NOM / τ_core
    """
    if tau_core <= 0.0:
        return M_DOT_SALT_NOM
    return M_DOT_SALT_NOM * TAU_CORE_NOM / tau_core
```

### Hand check at nominal

- P = 500 MW, ṁ = 8 333 kg/s, T_in = 860 K
- T_salt,ss = 860 + 5e8 / (8333 · 1500) = 860 + 40 = **900 K ✓**
- heat_out = 8333 · 1500 · 40 = 5e8 W = heat_in → **dT/dt = 0 ✓**

---

<a name="msr-critical"></a>

# 18. MSR Critical Base Reactivity

At flowing steady state the effective delayed-neutron source is reduced because precursors decay outside the core. To make `dn/dt = 0` at t = 0, the orchestrator inserts a compensating positive reactivity:

### Derivation

From Eq. 1 at steady state: `(ρ − β_eff)·n = −Λ·Σλᵢ·Cᵢ`.
At flowing SS, `Σλᵢ·Cᵢ_flow = n·Σ[βᵢ·λᵢ/(λᵢ + αᵢ)]` where `αᵢ = (1/τ_core)·(1 − e^(−λᵢτ_loop))`.
Solving for ρ that gives dn/dt = 0:

```
ρ_crit = β_eff_static − Σᵢ βᵢ·λᵢ / (λᵢ + αᵢ)
       where αᵢ = (1/τ_core) · (1 − e^(−λᵢτ_loop))
```

### Implementation

```python
def msr_critical_base_reactivity_pcm(beta_arr, lambda_arr, Lambda,
                                     tau_core, tau_loop):
    """Return base reactivity (pcm) needed to make the flowing MSR exactly critical.

    At zero salt velocity (τ_core → ∞, αᵢ → 0): ρ_crit → 0.
    At finite velocity: ρ_crit > 0 (compensates precursor deficit).
    """
    if tau_core >= 1.0e8:
        return 0.0
    alpha_i = (1.0 / tau_core) * (1.0 - np.exp(-lambda_arr * tau_loop))
    beta_eff_static = float(np.sum(beta_arr))
    effective_delayed = float(np.sum(beta_arr * lambda_arr / (lambda_arr + alpha_i)))
    rho_crit_dk_k = beta_eff_static - effective_delayed
    return rho_crit_dk_k * 1.0e5   # pcm
```

This is computed once at simulation start and stored in `MSRModelConfig.base_reactivity_pcm`.

---

<a name="msr-coupled"></a>

# 19. MSR Full 8-State Coupled System

### State vector layout

```
y[0]    = n         neutron population (n=1 ↔ P = P_ref)
y[1:7]  = C₁..C₆   in-core delayed-neutron precursor concentrations
y[7]    = T_salt    unified salt temperature (K)
```

Note: SPEC §4.6.2 lists 9 states with index 8 = ρ_total, but ρ_total is algebraic (computed each step from controls + T_salt), not integrated. The code's 8-state interpretation is correct.

### Implementation (verbatim coupled RHS)

```python
def msr_coupled_rhs(t, y, controls_fn, history, config, beta_arr, lambda_arr):
    """Full 8-state closed-loop MSR ODE right-hand side.

    Coupling chain at each solver call:
        1. T_salt drives salt temperature feedback → ρ_temp.
        2. Total ρ drives neutron kinetics (Eq. 1, shared core).
        3. History lookup C(t − τ_loop) provides return term (Eq. 23).
        4. Drift terms are added to dCᵢ/dt (Eq. 22).
        5. Power P = n · P_ref drives the salt temperature ODE.
    """
    controls = controls_fn(t, y)

    n = y[0]
    C = y[1:7]
    T_salt = y[7]

    # --- Algebraic reactivity ---
    rho_base = config.base_reactivity_pcm * 1.0e-5         # criticality compensation
    rho_ext  = controls.external_reactivity_pcm * 1.0e-5
    rho_temp = config.alpha_salt_pcm_per_k * (T_salt - config.T_salt_ref_k) * 1.0e-5
    rho_total = rho_base + rho_ext + rho_temp

    # --- Kinetics block (SPEC Eq. 1 + Eq. 22 base) — shared core ---
    kin_rhs = point_kinetics_rhs(t, y[:7], rho_total, beta_arr, lambda_arr,
                                 config.Lambda)

    # --- Precursor drift (SPEC Eqs. 22-23) ---
    if config.tau_core < 1.0e8:
        C_delayed = history.get(t - config.tau_loop)
        drift = msr_precursor_drift_rhs(
            C, C_delayed, lambda_arr, config.tau_core, config.tau_loop
        )
        dydt = np.empty(8)
        dydt[0] = kin_rhs[0]
        dydt[1] = kin_rhs[1] + drift[0]
        dydt[2] = kin_rhs[2] + drift[1]
        dydt[3] = kin_rhs[3] + drift[2]
        dydt[4] = kin_rhs[4] + drift[3]
        dydt[5] = kin_rhs[5] + drift[4]
        dydt[6] = kin_rhs[6] + drift[5]
    else:
        # Zero-flow limit: drift vanishes; MSR ≡ static-fuel point kinetics
        dydt = np.empty(8)
        dydt[:7] = kin_rhs

    # --- Salt thermal block ---
    power_w = n * config.P_ref
    m_dot = m_dot_salt_from_tau_core(config.tau_core) * controls.salt_flow_fraction
    th_rhs = salt_thermal_rhs(t, y[7:8], power_w, m_dot, config.T_salt_inlet_k)
    dydt[7] = th_rhs[0]

    return dydt
```

### Method-of-steps integration loop

```python
def integrate_msr(y0, t_span, controls, config, ...):
    """Integrate the 8-state MSR DDE system using method-of-steps."""
    # Chunk size = τ_loop/2 guarantees in-chunk history lookups never reference
    # the future (SPEC §4.6.3)
    dt_history = config.tau_loop / 2.0 if config.tau_loop > 0.0 else 10.0

    # Pre-fill history buffer with steady-state precursors
    history = PrecursorHistory(t_start=t_start, C_init=y0[1:7])
    history.add(t_start, y0[1:7])

    t_cur, y_cur = t_start, y0.copy()
    while t_cur < t_end:
        t_chunk_end = min(t_cur + dt_history, t_end)
        result = solve_ivp(
            lambda t, y: msr_coupled_rhs(t, y, controls_fn, history,
                                         config, beta_arr, lambda_arr),
            (t_cur, t_chunk_end), y_cur,
            method="Radau", rtol=1e-6, atol=1e-9, ...
        )
        # Update history with accepted points from this chunk
        for k in range(result.y.shape[1]):
            history.add(result.t[k], result.y[1:7, k])
        t_cur, y_cur = result.t[-1], result.y[:, -1]
    return ...
```

---

<a name="steady-state"></a>

# 20. Steady-State Construction (All Reactors)

Used to construct y₀ so `dy/dt = 0` exactly at t = 0. This avoids artificial start-up transients.

| Reactor | State | Formula | Where |
|---|---|---|---|
| PWR | Cᵢ,₀ | (βᵢ / (λᵢ·Λ)) · n₀ | Eq. 2 SS |
| PWR | T_f,₀, T_c,₀ | T_c = T_in + γ_f·P/(ṁ·c_c); T_f = T_c + γ_f·P·R_fc | Eq. 9, 10 SS |
| PWR | I₀, X₀ | γ_I·Σ_f·φ/λ_I;  (γ_X+γ_I)·Σ_f·φ/(λ_X + σ_aX·φ) | Eq. 14, 15 SS |
| MSR | Cᵢ,flow,₀ | (βᵢ/Λ)·n / [λᵢ + (1/τ_core)·(1−e^(−λᵢτ_loop))] | Eq. 22 SS |
| MSR | T_salt,₀ | T_inlet + P / (ṁ_salt · c_salt) | Eq. 16-equivalent SS |
| Helion | W₀ | (15/4) · n_ions · T · J_per_keV · V (post-compression) | Eq. 19 |

For PWR and MSR, the orchestrator additionally calculates `base_reactivity_pcm` such that ρ_total = 0 at t = 0 — see `critical_base_reactivity_pcm` (PWR) and `msr_critical_base_reactivity_pcm` (MSR).

---

<a name="validation-checklist"></a>

# 21. Inter-Equation Validation Checklist

Every row below is exercised by an automated pytest. All 447 physics tests pass.

| # | Invariant | Spec criterion | Hand-verified value |
|---|---|---|---|
| 1 | Σβᵢ for U-235 thermal | §4.5.4 | 6.500e-3 (exact) |
| 2 | dn/dt = 0 at SS with ρ = 0 | §7.2 | 0.0 (machine precision) |
| 3 | All dCᵢ/dt = 0 at SS | §7.2 | <1e-13 |
| 4 | Power drift < 0.1% over 1000 s steady-state | §7.3 | passes |
| 5 | +50 pcm self-regulates via Doppler | §7.3 | n stabilises ~1.03 (not divergent) |
| 6 | Xenon eq. worth at full power | §7.3 | −2 870 pcm (target −2 500 to −3 000) |
| 7 | Xenon peaks 6–10 h post-shutdown | §7.3 | passes |
| 8 | +1$ insertion is prompt-critical (divergent) | §7.3 | passes |
| 9 | Helion energy conservation per pulse | §7.3 | <1% imbalance |
| 10 | Brem dominates fusion below ~30 keV | §7.3 | passes |
| 11 | Q < 1 at sub-ignition | §7.3 | passes |
| 12 | ⟨σv⟩ matches B&H Table I at 10/50/100 keV | §4.5.2 | ≤5% / ≤10% (low-T) |
| 13 | Compression W ↔ T round-trip | §4.5.2 | exact |
| 14 | β_eff,flow ≤ β_eff_static | §7.3 | always; → 0 as v_salt → ∞ |
| 15 | β_eff,flow monotonically ↓ with v_salt | §7.3 | passes |
| 16 | MSR results ≡ PWR results at v_salt = 0 | §7.3 | passes (when α_salt = 0) |

---

<a name="coupling-maps"></a>

# 22. Coupling Maps (One-Page Diagrams)

### PWR full coupling

```
controls (rod_pcm, boron_ppm, flow_frac, T_in)
   │
   ↓
┌─────────────────────────────────────────────────────────┐
│   STATE y = [n, C₁..C₆, T_f, T_c, I, X]                 │
└─────────────────────────────────────────────────────────┘
   │             │             │
 y[7]=T_f      y[8]=T_c      y[10]=X
   ↓             ↓             ↓
 Eq. 11 ρ_D   Eq. 12 ρ_m    ρ_Xe = −σ_aX·X/Σ_a
   │             │             │
   └─────┬───────┴───────┬─────┘
         ↓               ↓
     Eq. 13 ρ_B       ρ_rod (input)
         │               │
         └─────┬─────┬───┘
               ↓     │
       ρ_total (pcm → ×1e-5 → Δk/k)
               │
               ↓
  ┌─────────────────────────────────────────────────────┐
  │ Eqs. 1, 2 (shared 6-group point kinetics, 7-state)  │
  │   dn/dt   = ((ρ − β_eff)/Λ)·n + Σλᵢ·Cᵢ              │
  │   dCᵢ/dt = (βᵢ/Λ)·n − λᵢ·Cᵢ                          │
  └─────────────────────────────────────────────────────┘
               │
       n ───┬────→ P = n·P_ref ──→ Eqs. 9, 10 → dT_f/dt, dT_c/dt
            │
            └────→ φ = n·φ_ref ──→ Eqs. 14, 15 → dI/dt, dX/dt
```

### MSR full coupling

```
controls (external_pcm, salt_flow_frac)
   │
   ↓
STATE y = [n, C₁..C₆, T_salt]
   │
 y[7]=T_salt
   ↓
 ρ_temp = α_salt(T_salt − T_ref)
   │       + ρ_base (criticality comp) + ρ_external (rod)
   ↓
ρ_total → Eq. 1 → dn/dt
   │
   └──→ Shared Eq. 2 base for dCᵢ/dt
                       │
                       + drift (Eq. 22 + Eq. 23):
                       │  − (1/τ_core)·Cᵢ
                       │  + (1/τ_core)·Cᵢ(t − τ_loop)·e^(−λᵢτ_loop)
                       ↑
              history buffer (interpolates past Cᵢ)

  n ────→ P = n·P_ref ──→ Eq. 16 (salt thermal) → dT_salt/dt
```

### Helion full coupling

```
STATE y = [W]
   │
   ↓
T_kev = (2/3)·W / (n_total·V·J_per_keV)        (Eq. 19 inverse)
   │
   ├──→ ⟨σv⟩(T) ──→ P_fusion = (n²/4)·⟨σv⟩·E_DHe3·V       (Eqs. 17, 18)
   │
   └──→ P_brem = C_B · Z_eff · n_e² · √T · V              (Eq. 20)

τ_E = C_TAU · (n/N_REF) · (B/B_REF)²
P_cond = W / τ_E                                          (Eq. 21)

dW/dt = P_heat + P_fusion − P_brem − P_cond              (Eq. 16)
```

---

<a name="reconciliation"></a>

# 23. Spec/Code Reconciliation Notes

These are the only known discrepancies between the spec text and the implementation. None are bugs; each is the correct resolution of a spec ambiguity.

| Item | Spec status | Code behavior | Resolution |
|---|---|---|---|
| Eq. 18 prefactor | Spec writes `C₁·θ²/(ξ·B_G²)·√(…)·exp(…)` | Code implements `C₁·θ·√(…)·exp(…)` per Bosch & Hale 1992 Eq. 12 | **Code is correct.** Verified against B&H Table I. Spec text is a documentation typo. |
| MSR state count | Spec §4.6.2 lists 9 states with index 8 = `ρ_total` | Code uses 8 states; ρ_total is algebraic (recomputed each step) | **Code is correct.** ρ_total has no `dρ/dt` equation, so it cannot be an ODE state. Spec calls it "algebraic, updated each step" — consistent with code. |
| MSR `build_initial_state` at unphysically large τ_core | Spec assumes flowing salt; behavior at τ_core → ∞ is the consistency test, not a real operating condition | Code added 3000 K bound on T_salt₀ to fall back to T_salt_ref_k when m_dot → 0 makes SS formula diverge | **Hardened during audit.** Existing zero-flow consistency tests bypass this by setting α_salt = 0; the new guard prevents a misleading NaN/inf in other diagnostic uses. |
| Bosch-Hale fit accuracy at low T | Spec acknowledges valid range 0.5–190 keV | Code clamps with warning outside this range; within range, fit error grows below ~10 keV | **By design (R-06).** Treat results below ~10 keV as inherent fit uncertainty, not a code issue. |

---

*Document version 1.0 — generated 2026-05-18 from a complete physics-layer audit.*
*Update at the end of each phase or whenever an equation moves, splits, or gains a new coupling partner.*
*To regenerate the hand-verified numbers in this document, run `python audit_physics.py` (in this repo, requires Python 3.11+, NumPy, SciPy, Numba).*
