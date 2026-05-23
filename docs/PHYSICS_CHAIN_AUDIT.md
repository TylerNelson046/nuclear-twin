# Physics Chain Audit — Input → Equation → Output → Graph

**Purpose:** Every user-facing control input is traced step-by-step through the governing
equations to every displayed output and the chart that shows it.  Each step includes the
literal equation number, the constant values used, and a hand-checkable numerical example.
Use this to verify that the UI wiring is correct and the physics is sensible.

**Code references:** `PHYSICSLOGIC.md` has the verbatim kernel implementations; this document
focuses on the *chain* from control knob → ODE state → displayed metric → chart.

---

## Table of Contents

1. [PWR — Inputs](#pwr-inputs)
2. [PWR — Full Output Derivation Table](#pwr-outputs)
3. [PWR — Graph Inventory](#pwr-graphs)
4. [Helion — Refined Input Design](#helion-inputs)
5. [Helion — Full Output Derivation Table](#helion-outputs)
6. [Helion — Graph Inventory](#helion-graphs)
7. [Manual Spot-Check Worksheet](#spot-checks)

---

<a name="pwr-inputs"></a>

## 1. PWR — User Inputs

The four operator controls are passed as `PWRControls` (a Pydantic-validated dataclass)
and then packed into a flat `params[27]` array before entering the Numba JIT kernel
`pwr_11_state_system`.

| UI Control | Python field | Units | Physical range | Default |
|---|---|---|---|---|
| Control rod reactivity | `rod_reactivity_pcm` | pcm (1 pcm = 10⁻⁵ Δk/k) | −1000 to +1000 | 0 |
| Soluble boron | `boron_ppm` | ppm (mg/kg) | 0 to 2500 | 0 |
| Coolant flow rate | `coolant_flow_fraction` | fraction of nominal | 0.20 to 1.20 | 1.0 |
| Core inlet temperature | `inlet_temperature_k` | K | 540 to 610 | 565 |

**UI note:** Inlet temperature is displayed in °C (subtract 273.15) but stored internally in K.
Flow fraction is displayed with the actual kg/s shown as a tooltip (nominal ≈ 20 786 kg/s).

### 1A. Control Rod Insertion — chain

```
rod_reactivity_pcm (input, pcm)
    │
    × 1e-5  →  rod_dk_k  [params[0], Δk/k]
    │
    added into ρ_total = base_dk_k + rod_dk_k
                         + (ρ_D + ρ_m + ρ_B + ρ_Xe) × 1e-5
    │
    → Eq. 1: dn/dt = ((ρ_total − β_eff)/Λ)·n + Σᵢ λᵢ·Cᵢ
```

**Physical meaning:**
- Positive pcm → positive Δk/k → numerator of Eq. 1 becomes less negative → n rises → P rises.
- Negative pcm = shutdown rod insertion → n drops → P drops.
- ρ_rod is the *external* reactivity; it adds algebraically to all the feedback terms.

**Numerical check (+100 pcm insertion at steady state):**
- rod_dk_k = 100 × 1e-5 = 1e-3
- prompt term: ((1e-3 − 6.5e-3)/1e-5)·1 = −550 s⁻¹  (was −650 at ρ=0)
- delayed source unchanged at +650 s⁻¹
- **dn/dt = +100 s⁻¹** → neutron population rises, power rises, Doppler kicks in within seconds

### 1B. Boron Concentration — chain

```
boron_ppm (input, ppm)
    │
    Eq. 13:  ρ_B = ω_B × boron_ppm    [ω_B = −10 pcm/ppm default]
    │
    → ρ_B (pcm, always ≤ 0)
    │
    × 1e-5  →  Δk/k contribution to ρ_total
```

**Physical meaning:**
- Boric acid dissolved in coolant absorbs thermal neutrons.
- ω_B = −10 pcm/ppm is a constant linear model (real reactors see slight nonlinearity).
- At 1000 ppm: ρ_B = −10 000 pcm = −0.1 Δk/k → deeply subcritical.
- Boron does NOT change with time in this model (it's a boundary condition, not an ODE state).

**Numerical check (500 ppm):**
- ρ_B = −10 × 500 = **−5 000 pcm**
- Without a compensating base reactivity or rod withdrawal, the reactor would be shut down.
- The orchestrator computes `critical_base_reactivity_pcm` at startup so equilibrium xenon +
  boron + Doppler sum to exactly 0 at the initial state.
- **Hidden calibration constant:** `critical_base_reactivity_pcm` is not a user input and does
  not appear in the output table. It is a computed offset that enforces the zero-reactivity
  initial condition — it absorbs any numerical residual from the sum of feedback terms at the
  reference state. If you change `T_fuel_ref`, `T_cool_ref`, or equilibrium xenon, this constant
  must be recomputed by calling `critical_base_reactivity_pcm(controls, config)` again. Its
  value is the single source of the nominal-zero-reactivity guarantee.

### 1C. Coolant Flow Rate — chain

```
coolant_flow_fraction (input, dimensionless)
    │
    × M_DOT_NOM  →  m_dot (kg/s)     M_DOT_NOM ≈ 20 786 kg/s
    │
    Eq. 10: dT_c/dt = (q_fc − m_dot·C_COOL·(T_c − T_in)) / (M_COOL·C_COOL)
    │
    ⚠ Model note: T_c in this equation represents the coolant OUTLET temperature, not
    the bulk core average. The advection term ṁ·C_COOL·(T_c − T_in) is exact for
    outlet-temperature interpretation. If T_c were bulk average then T_out = 2T_c − T_in
    and the advection term would carry a factor of 2. The current form is kept because it
    correctly drives moderator feedback from the highest-temperature coolant in the core.
    │
    T_c (state y[8]) changes over time
    │
    Eq. 12: ρ_m = α_m × (T_c − T_c_ref)   [α_m = −35 pcm/K]
    │
    → moderator feedback contribution to ρ_total → Eq. 1 → n → P
    │
    (also: T_c affects q_fc = (T_f − T_c)/R_fc which feeds back into T_f via Eq. 9)
```

**Physical meaning:**
- Reducing flow → less heat removed from coolant → T_c rises → ρ_m becomes more negative →
  reactor moderates itself down (moderator temperature coefficient is negative).
- The flow fraction also affects fuel temperature indirectly: less effective cooling raises T_c,
  which narrows (T_f − T_c), which reduces q_fc, which reduces the driving force in Eq. 9
  (fuel temperature ODE slows its descent).
- At 0.2× nominal: m_dot = 0.2 × 20 786 = 4 157 kg/s.

**Numerical check (flow drops to 0.5× at full power):**
- New SS T_c = T_in + γ_f·P / (m_dot·C_COOL) = 565 + (0.97·3e9)/(0.5·20786·5600) = 565 + 50 = **615 K**
- ρ_m = −35 × (615 − 590) = **−875 pcm** (was 0 at nominal)
- Net negative reactivity → power drops until Eq. 1 reaches new equilibrium.

### 1D. Inlet Temperature — chain

```
inlet_temperature_k (input, K)
    │
    Eq. 10: enters as T_in in the advection term  m_dot·C_COOL·(T_c − T_in)
    │
    Higher T_in → smaller (T_c − T_in) → less heat removed → T_c rises
    │
    Eq. 12: ρ_m = α_m × (T_c − T_c_ref) → more negative
    │
    → Eq. 1 → n falls → P falls
    │
    (T_c change also propagates to T_f via q_fc = (T_f − T_c)/R_fc)
```

**Physical meaning:**
- Hotter inlet water is a less effective heat sink. The coolant outlet temperature rises.
- The moderator feedback acts as a self-regulating mechanism.
- Physically, changing T_in models the secondary loop steam generator performance.

**Nominal SS check (T_in = 565 K, m_dot = nominal):**
- T_c_ss = 565 + (0.97·3e9)/(20786·5600) = 565 + **25 K** = 590 K ✓
- T_f_ss = 590 + 0.97·3e9 × R_FC = 590 + (2.91e9 × 1.0653e-7) = 590 + **310 K** = 900 K ✓

---

<a name="pwr-outputs"></a>

## 2. PWR — Full Output Derivation Table

The 11 ODE states plus algebraic outputs derived from them:

| # | Output | Source | Governing equation | Units | Depends on inputs |
|---|---|---|---|---|---|
| 1 | Neutron population `n` | ODE state y[0] | Eq. 1 | dimensionless (1 = nominal) | rod, boron, flow, T_in |
| 2–7 | Precursors C₁–C₆ | ODE states y[1:7] | Eq. 2 | dimensionless (normalized, same scale as n) | rod, boron (through n) |
| 8 | Fuel temperature T_f | ODE state y[7] | Eq. 9 | K | flow, T_in (through P) |
| 9 | Coolant temperature T_c | ODE state y[8] | Eq. 10 | K | flow, T_in |
| 10 | I-135 concentration I | ODE state y[9] | Eq. 14 | atoms/cm³ | rod, boron (through φ) |
| 11 | Xe-135 concentration X | ODE state y[10] | Eq. 15 | atoms/cm³ | rod, boron (through φ) |
| 12 | Thermal power | algebraic: n × P_ref | — | W → displayed in MW | all |
| 13 | Thermal flux φ | algebraic: n × φ_ref | — | n/cm²/s | all |
| 14 | Doppler reactivity | algebraic: Eq. 11 | ρ_D = α_D·(T_f − T_f0) | pcm | flow, T_in (through T_f) |
| 15 | Moderator reactivity | algebraic: Eq. 12 | ρ_m = α_m·(T_c − T_c0) | pcm | flow, T_in |
| 16 | Boron reactivity | algebraic: Eq. 13 | ρ_B = ω_B·C_B | pcm | boron |
| 17 | Xenon reactivity | algebraic: −σ_aX·X/Σ_a × 1e5 | — | pcm | rod, boron (through X) |
| 18 | Total reactivity | sum of all components | — | pcm | all |
| 19 | ΔT fuel–coolant | algebraic: T_f − T_c | — | K (or °C equiv) | flow, T_in, rod |
| 20 | State derivatives | dy/dt evaluated each step | — | 1/s, K/s | all |

### How each output number is calculated:

**Thermal Power (output #12):**
```python
P_w = n * P_ref            # P_ref = 3.0e9 W at n=1
power_mw = P_w / 1e6       # displayed as MWth
power_normalized = n / 1.0  # fraction of nominal (n=1 = 100%)
```
At n=1.03 (after +100 pcm insertion settles): P = 1.03 × 3000 = **3 090 MWth**

**Thermal Flux (output #13):**
```python
phi = n * PHI_NOM     # PHI_NOM = 3.1e13 n/cm²/s
```

**Doppler Reactivity (output #14) — Eq. 11:**
```python
rho_d_pcm = alpha_d * (T_fuel - T_fuel_ref)   # alpha_d = −2.5 pcm/K
```
At T_f = 950 K (50 K above ref): ρ_D = −2.5 × 50 = **−125 pcm**

**Moderator Reactivity (output #15) — Eq. 12:**
```python
rho_m_pcm = alpha_m * (T_cool - T_cool_ref)   # alpha_m = −35 pcm/K
```
At T_c = 600 K (10 K above ref): ρ_m = −35 × 10 = **−350 pcm**

**Boron Reactivity (output #16) — Eq. 13:**
```python
rho_b_pcm = omega_b * boron_ppm               # omega_b = −10 pcm/ppm
```
At 750 ppm: ρ_B = −10 × 750 = **−7 500 pcm**

**Xenon Reactivity (output #17):**
```python
rho_xe_pcm = -SIGMA_AX * X / sigma_a * 1.0e5
# SIGMA_AX = 2.6e-18 cm², sigma_a = 0.55 cm⁻¹
```
At equilibrium (X = 6.07e15 atoms/cm³, full power):
ρ_Xe = −2.6e-18 × 6.07e15 / 0.55 × 1e5 = **−2 870 pcm** ✓

**Total Reactivity (output #18):**
```python
rho_total_dk_k = base_dk_k + rod_dk_k + (rho_d_pcm + rho_m_pcm + rho_b_pcm + rho_xe_pcm) * 1e-5
rho_total_pcm  = rho_total_dk_k * 1e5   # for display
```
At nominal SS (everything at reference): ρ_total = 0 pcm ✓ (by construction via base_reactivity)

**Precursor Concentrations (output #2–7) — Eq. 2:**
```python
dCi_dt = (beta_arr[i] / Lambda) * n - lambda_arr[i] * Ci
```
Steady state: Ci_ss = (βᵢ / (λᵢ × Λ)) × n
At n=1: C₁_ss = 2.470e-4 / (0.0129 × 1e-5) = **1 921** (dimensionless normalized units, same scale as n — NOT atoms/cm³) ✓
Note: because n=1 is a dimensionless normalized population (not an absolute density), the Cᵢ
computed from dCᵢ/dt = (βᵢ/Λ)·n − λᵢ·Cᵢ are also dimensionless. I-135 and Xe-135 (outputs
#10–11) ARE in atoms/cm³ because they use absolute neutron flux φ = n × PHI_NOM as their
production source, which re-introduces physical units.

**I-135 (output #10) — Eq. 14:**
```python
dI_dt = GAMMA_I * Sigma_f * phi - LAMBDA_I * I   # GAMMA_I = 0.0639, LAMBDA_I = 2.87e-5
```
SS at full power: I₀ = 0.0639 × 9.3e12 / 2.87e-5 = **2.07e16 atoms/cm³** ✓

**Xe-135 (output #11) — Eq. 15:**
```python
dX_dt = GAMMA_X * Sigma_f * phi + LAMBDA_I * I - (LAMBDA_X + SIGMA_AX * phi) * X
```
SS: X₀ = (0.0639 + 0.00237) × 9.3e12 / (2.09e-5 + 2.6e-18 × 3.1e13)
       = 6.163e11 / 1.015e-4 = **6.07e15 atoms/cm³** ✓

**ΔT (output #19):**
```python
delta_t_k = T_fuel - T_coolant    # both in K, ΔT same in K and °C
```
At nominal SS: ΔT = 900 − 590 = **310 K** (= 310 °C difference)

---

<a name="pwr-graphs"></a>

## 3. PWR — Graph Inventory

Every chart in the PWR panel, what data feeds it, and what to verify.

| Chart title | Key traces | History keys in `session["history"]` | What to verify |
|---|---|---|---|
| **Core Thermal Power** | Thermal power (MWth), Power level (%) | `power_mw`, `power_normalized` | At steady state: 3000 MWth / 100%. After +100 pcm: rises then plateaus via Doppler. |
| **Fuel and Coolant Temperatures** | T_fuel (K or °C), T_coolant (K or °C) | `fuel_temperature_k`, `coolant_temperature_k` | Nominal SS: ~627 °C / ~317 °C. Increasing flow → lower T_c. |
| **Reactivity Breakdown** | Net, Rod, Doppler, Moderator, Boron, Xenon | `total_reactivity_pcm`, `rod_reactivity_pcm`, `doppler_reactivity_pcm`, `moderator_reactivity_pcm`, `boron_reactivity_pcm`, `xenon_reactivity_pcm` | At SS: net ≈ 0. Each component labeled. Xenon band −2500 to −3000 pcm shown. |
| **Iodine and Xenon Concentrations** | I-135, Xe-135 | `iodine_concentration`, `xenon_concentration` | Both rise to SS. After shutdown: Xe peaks at 6–10 h (xenon pit). |
| **Neutron Population & Thermal Flux** | n (normalized), φ (n/cm²·s) | `neutron_population`, `thermal_flux_n_cm2_s` | φ = n × 3.1e13. Both stay ≈1 at SS. |
| **Power vs Total Reactivity** | Phase-plane trajectory | `total_reactivity_pcm` (x), `power_mw` (y) | Should trace an arc: rod insertion → ρ drops → P drops; self-regulating rod ejection returns to near-nominal. |
| **State Derivatives** | dn/dt, dC₁/dt–dC₆/dt, dT_f/dt, dT_c/dt, dI/dt, dX/dt (all 11 states) | `dn_dt`, `dC1_dt`–`dC6_dt`, `dT_fuel_dt`, `dT_coolant_dt`, `dI_dt`, `dX_dt` | All ≈ 0 at SS. Non-zero during transient. Precursor derivatives are typically small compared to n but decay visibly after a shutdown. |
| **Delayed Neutron Precursor Groups** | C₁–C₆ | `precursor_C1` … `precursor_C6` | Slower groups (C₁, t½ 54 s) are largest; faster groups smaller. After shutdown: decay in order of half-life. |
| **Fuel–Coolant ΔT** | ΔT = T_f − T_c | computed from `fuel_temperature_k`, `coolant_temperature_k` | Nominal: 310 K. Drops with reduced power. Rises if flow decreases. |
| **Control Inputs Schedule** | Boron ppm, flow fraction, ṁ (kg/s), rod (pcm), T_in (°C) | `boron_ppm`, `coolant_flow_fraction`, `rod_reactivity_pcm`, `inlet_temperature_k` | Shows exact operator input history — use to correlate inputs to output responses. |
| **Xenon Worth vs Concentration** | Phase-plane: ρ_Xe vs N_Xe | `xenon_concentration` (x), `xenon_reactivity_pcm` (y) | Should be a straight line: slope = −σ_aX/Σ_a × 1e5 = −(2.6e-18/0.55) × 1e5 = **−4.73e-13 pcm per atom/cm³**. |

---

<a name="helion-inputs"></a>

## 4. Helion — Refined Input Design

### 4A. Current inputs (in UI)

| UI label | Python field | Conversion | Units (internal) | Default | Pydantic bounds |
|---|---|---|---|---|---|
| Initial Temperature | `ion_temperature_kev` | none | keV | 20.0 | 0.5 – 190 |
| Plasma Density | `plasma_density_m3` | UI value × 10²¹ | m⁻³ | 1.0 (= 1×10²¹ m⁻³ internal) | 0.01 – 100 (UI units; 1e19–1e23 m⁻³ internal) |
| Compression Ratio Rᶜ | `compression_ratio` | none | dimensionless | 10.0 | 1 – 1000 |
| Magnetic Field | `magnetic_field_t` | none | T | 5.0 | 0.1 – 100 |
| Plasma Volume | `plasma_volume_m3` | none | m³ | 1.0 | 0.01 – 100 |
| Pulse Duration | `pulse_duration_s` | UI value × 10⁻⁶ | s | 10 μs | 100 ns – 10 ms |

### 4B. Inputs to add — design rationale from Helion

Based on Helion Energy's published design documents and patent filings:

**1. External Heating Power P_heat (W)**

In Helion's FRC design the plasma is pre-heated during the FRC formation phase before
magnetic compression. The parameter `P_heat_w` is already a **kernel-level argument** of
`integrate_helion_pulse(parameters, P_heat_w=0.0, ...)` — it appears directly in Eq. 16
as a positive energy source term with a default of 0.0 (adiabatic pulse). It is **not**
currently wired through `helion_runner.run_helion_pulse()` or exposed in the UI; it is
hardcoded to 0.0 at the orchestrator boundary. Adding it requires only a UI control and
a pass-through in the runner, not any ODE change.

- Helion specification: "field-reversed configuration heating via flux injection and
  ohmic heating during plasma formation".
- Typical range: 0 to ~500 MW (W units in the ODE; useful UI range: 0 – 1 × 10⁹ W).
- Physical effect: higher P_heat → more input energy → Q denominator increases → Q may
  drop even as fusion rises.

**2. Pulse Repetition Rate (Hz)**

Helion's commercial design is a *pulsed* device. The mean electrical output is:
```
Mean fusion power (W) = E_fusion_per_pulse (J) × pulse_rate (Hz)
Net electrical output (W) = Mean_fusion_power × η_direct − P_recirculating
```
Where η_direct ≈ 0.7–0.95 for Helion's direct energy conversion scheme.
Helion target: ~1 Hz pulse rate for commercial-scale 50 MWe generator.
Valid range: 0.01 – 10 Hz.

This is a **post-processing** parameter only — it does not enter the ODE.

**3. Number of Pulses**

For multi-pulse runs (beyond the single pulse currently modelled), the quantities of
interest are:
- Total fusion energy = n_pulses × E_fusion_per_pulse
- Cumulative materials damage (DPA, thermal fatigue, capacitor cycles)
- Average steady-state power

The ODE integrates one pulse. Multi-pulse behaviour is additive for energy and
multiplicative for materials degradation. Valid range: 1 – 10 000.

### 4C. Complete proposed input set

| Parameter | Symbol | Units | Role | Enters ODE? | Current state |
|---|---|---|---|---|---|
| Pre-compression temperature | T₀ | keV | Sets post-compression T via Eq. 19 | Yes (sets W₀) | ✓ in UI |
| Pre-compression density | n₀ | m⁻³ | Sets post-compression n via Eq. 19 | Yes (fixed during ODE) | ✓ in UI |
| Compression ratio | Rᶜ | — | Eq. 19: T_f = T₀·Rᶜ^(2/3), n_f = n₀·Rᶜ | Yes (sets initial conditions) | ✓ in UI |
| Magnetic field | B | T | τ_E = C_TAU·(n/N_REF)·(B/B_REF)² | Yes (in P_cond at every step) | ✓ in UI |
| Pre-compression volume | V₀ | m³ | V_f = V₀/Rᶜ; V_f fixed during ODE | Yes (fixed during ODE) | ✓ in UI |
| Pulse duration | Δt | μs→s | ODE integration span | Yes (sets t_span) | ✓ in UI |
| External heating power | P_heat | W | dW/dt += P_heat (Eq. 16 source term) | **Yes** | ❌ MISSING |
| Pulse repetition rate | f_rep | Hz | Mean power = E_fusion × f_rep | No (post-processing) | ❌ MISSING |
| Number of pulses | n_pulses | count | Total energy, materials degradation | No (post-processing) | ❌ MISSING |

---

<a name="helion-outputs"></a>

## 5. Helion — Full Output Derivation Table

### Step 1: Pre-compression (Eq. 19) — computed once, before ODE

```
INPUTS: T₀ (keV), n₀ (m⁻³), Rᶜ, V₀ (m³)

T_final = T₀ × Rᶜ^(2/3)              keV        (adiabatic compression, γ=5/3)
n_final = n₀ × Rᶜ                    m⁻³
V_final = V₀ / Rᶜ                    m³
n_total = 2.5 × n_final              m⁻³        (D-He3 50/50: n_total/n_ions = 5/2)
W₀ = 1.5 × n_total × (T_final × J_per_keV) × V_final    J
```

**Numerical example (defaults: T₀=20 keV, n₀=1e21, Rᶜ=10, V₀=1 m³):**
- T_final = 20 × 10^(2/3) = 20 × 4.642 = **92.8 keV**
- n_final = 1e22 m⁻³
- V_final = 0.1 m³
- n_total = 2.5e22 m⁻³
- W₀ = 1.5 × 2.5e22 × (92.8 × 1.602e-16) × 0.1 = **5.58e7 J** (~55.8 MJ)

### Step 2: ODE integration (Eq. 16) — at every solver step

**ODE state: W(t) — plasma thermal energy (J)**

At each step t:

```
T_kev = (2/3) × W / (n_total × V_final × J_per_keV)      [Eq. 19 inverse]
T_kev_clamped = clip(T_kev, 0.5, 190.0)                   [Bosch-Hale validity]

σv = sigma_v_kernel(T_kev_clamped, B-H coeffs)            [Eq. 18, m³/s]
P_fusion = (n_final² / 4) × σv × E_DHe3 × V_final        [Eq. 17, W]

n_e = 1.5 × n_final                                        [D-He3 quasineutrality]
P_brem = 5.35e-37 × (5/3) × n_e² × √T_kev × V_final      [Eq. 20, W]

τ_E = 5e-5 × (n_final/1e21) × (B/5)²                      [Eq. 21, s]
P_cond = W / τ_E                                           [Eq. 21, W]

dW/dt = P_heat + P_fusion − P_brem − P_cond               [Eq. 16, W]
```

**Model assumption — 100% charged particle heating:**
The ODE assumes 100% of P_fusion deposits into the plasma as heat. This is valid for the
primary D-He3 reaction: D + ³He → p (14.7 MeV) + α (3.6 MeV). Both products are charged
and magnetically confined, so ~100% of their energy thermalizes in the plasma.

However, D-D side reactions occur at fusion temperatures and produce 2.45 MeV neutrons that
escape the magnetic bottle instantly. At fusion-relevant temperatures these side reactions
contribute ~1–5% of total yield. Their neutron energy is excluded from dW/dt. For the
purposes of this model: **assumes pure D-He3 fuel with no D-D side reactions and complete
charged-particle thermalization.**

### Step 3: Post-processing — computed from W(t) array after ODE

```
E_fusion = ∫P_fusion dt  (trapezoid rule over t array)     J
E_brem   = ∫P_brem dt                                     J
E_cond   = ∫P_cond dt                                     J
E_input  = W₀ + P_heat × Δt                               J
Q_plasma = E_fusion / E_input                              dimensionless  [Plasma Q — see note]
```

**Q factor labeling note — Plasma Q vs Engineering Q:**
The Q computed above is **Plasma Q** (thermal): it measures fusion energy out divided by
thermal energy put into the plasma (initial stored energy W₀ plus external heating P_heat·Δt).

It does **not** include the electrical work done by the magnetic compression coils to achieve
the high post-compression W₀ in the first place. Helion's FRC is a pulsed machine; each pulse
requires a capacitor-bank discharge to drive the compression coils. Recovering that energy
during plasma expansion is never 100% efficient (Helion targets ~85–90% direct energy
conversion, i.e. η_mag ≈ 0.85).

```
Engineering Q = E_fusion / E_total_input
where E_total_input = W₀/η_mag + P_heat·Δt
```

At η_mag = 0.85, Engineering Q ≈ Plasma Q × 0.85.  A system with Plasma Q = 1.2 has
Engineering Q ≈ 1.0 — the commercial viability threshold.  The UI displays Plasma Q; the
documentation should always label it as such.

### Step 4: New post-processing with proposed additional inputs

```
Mean_fusion_power = E_fusion × f_rep                                W   (E_fusion is already J/pulse)
Net_avg_power     = Mean_fusion_power − P_heat × Δt × f_rep         W   (P_heat × Δt = recirculated J/pulse; × f_rep = average W)
Total_fusion_energy = n_pulses × E_fusion                           J
Net_electrical_out  = Net_avg_power × η_direct − P_recirculating    W   (η_direct ≈ 0.7–0.95; see note below)
```

**η_direct and P_recirculating — model constants, not yet in UI:**
- `η_direct` — Helion's direct energy conversion efficiency (magnetic flux energy recovered during
  plasma decompression). Helion target: 0.7–0.95. Not currently a simulation parameter; used only
  in the net electrical output formula above.
- `P_recirculating` — parasitic facility power (magnets, controls, cooling). Order-of-magnitude:
  a few MW for a 50 MWe-class device. Also not a simulation parameter.
Both are documentation constants for context. If net electrical output is added to the UI, they
would need to be input fields with explicit bounds.

### Full output table

| # | Output | Formula | Units | Depends on | Graph |
|---|---|---|---|---|---|
| 1 | Post-compression temperature T_final | T₀ × Rᶜ^(2/3) | keV | T₀, Rᶜ | Compression bar chart |
| 2 | Post-compression density n_final | n₀ × Rᶜ | m⁻³ | n₀, Rᶜ | Compression bar chart |
| 3 | Post-compression volume V_final | V₀ / Rᶜ | m³ | V₀, Rᶜ | Compression bar chart |
| 4 | Initial plasma energy W₀ | (15/4)·n_f·T_f·J/keV·V_f | J | T₀, n₀, Rᶜ, V₀ | Metric tile: W_initial |
| 5 | Plasma temperature T(t) | (2/3)·W/(n_total·V·J/keV) | keV | T₀, n₀, Rᶜ, V₀, B, Δt | Temperature chart |
| 6 | Fusion reactivity ⟨σv⟩(t) | Bosch-Hale Eq. 18 of T(t) | m³/s | same + T route | Reactivity & τ_E chart |
| 7 | Fusion power P_fusion(t) | (n²/4)·⟨σv⟩·E_DHe3·V | W | n₀, Rᶜ, T route | Power Balance chart |
| 8 | Bremsstrahlung loss P_brem(t) | 5.35e-37·(5/3)·n_e²·√T·V | W | n₀, Rᶜ, T route | Power Balance chart |
| 9 | Confinement time τ_E(t) | 5e-5·(n/N_REF)·(B/B_REF)² | s | n₀, Rᶜ, B | Reactivity & τ_E chart |
| 10 | Conduction loss P_cond(t) | W / τ_E | W | W(t), n₀, B | Power Balance chart |
| 11 | Net power dW/dt(t) | P_heat + P_fusion − P_brem − P_cond | W | all | Net Power chart |
| 12 | Plasma energy W(t) | ODE state (integrated Eq. 16) | J | all | Plasma Energy chart |
| 13 | Electron density n_e(t) | 1.5 × n_final (constant) | m⁻³ | n₀, Rᶜ | n_e chart |
| 14 | **Pulse-integrated gain** Q_plasma | E_fusion / E_input — total fusion energy ÷ total thermal input energy over the pulse | dimensionless | all | Metric tile: Q |
| 15 | **Instantaneous physics gain** Q(t) | P_fusion / (P_brem + P_cond) — ratio of fusion power to instantaneous loss power at each solver step | dimensionless | T(t), n₀, B | Q(t) chart |
| 16 | P_brem / P_fusion ratio | P_brem / P_fusion | dimensionless | T(t), n₀ | Q(t) chart (2nd y-axis) |
| 17 | E_fusion (integrated) | ∫P_fusion dt | J | all | Metric tile: E_fusion |
| 18 | E_brem (integrated) | ∫P_brem dt | J | T route, n₀ | Metric tile: E_brem |
| 19 | E_cond (integrated) | ∫P_cond dt | J | W(t), n₀, B | Metric tile: E_cond |
| 20 | Mean fusion power* | E_fusion × f_rep | W | E_fusion, f_rep | (new metric tile) |
| 21 | Total fusion energy* | n_pulses × E_fusion | J | E_fusion, n_pulses | (new metric tile) |

*Items 20–21 require the proposed new inputs (f_rep, n_pulses) and are not yet in the UI.

### Key relationships to verify manually

**Why T drops during the pulse (sub-ignition case):**
- At T < ~30 keV: P_brem > P_fusion (see criterion below)
- P_cond is very large (τ_E is small at low B/n)
- dW/dt < 0 → W falls → T falls → ⟨σv⟩ falls → reinforcing collapse

**Bremsstrahlung break-even temperature:**
P_fusion = P_brem when:
```
(n²/4)·⟨σv⟩·E_DHe3 = C_B·Z_eff·n_e²·√T·V
```
For D-He3: n_e = 1.5·n, Z_eff = 5/3.
At n=1e22, V=0.1: break-even ≈ **30 keV** (exact value from B-H coefficients).
Below this temperature, bremsstrahlung dominates — no ignition possible regardless of
compression.

**Compression ratio and Q scaling:**
- Q ∝ n at fixed Rᶜ (density helps both P_fusion and τ_E)
- Q ∝ Rᶜ^(1/3) × ⟨σv⟩(T_initial × Rᶜ^(2/3)) when T not clamped at 190 keV
- At Rᶜ=1000, n=1e23: ignition expected (Q > 1 from scenarios test)

**τ_E calibration check:**
- At n=N_REF=1e21, B=B_REF=5 T: τ_E = 5e-5 s = **50 μs** ✓ (nominal FRC confinement)
- At n=1e22, B=10 T: τ_E = 5e-5 × (1e22/1e21) × (10/5)² = 5e-5 × 10 × 4 = **2 ms**
- Higher B and n → longer confinement → more fusion per pulse

---

<a name="helion-graphs"></a>

## 6. Helion — Graph Inventory

| Chart title | Traces | Source keys in pulse store | What to verify |
|---|---|---|---|
| **Plasma Temperature** | T(t) in keV | `T_kev` | Starts at T_final (post-compression). Sub-ignition: drops as W decays. Ignition: rises. |
| **Power Balance** | P_fusion, P_brem, P_cond (+ P_heat if >0) | `P_fusion_w`, `P_brem_w`, `P_cond_w` | At T < 30 keV: P_brem > P_fusion. At ignition: P_fusion > both losses. |
| **Plasma Energy W(t)** | W in J | `W_J` | Starts at W₀. Sub-ignition decays to 0. Ignition grows. Area under P_fusion curve = E_fusion. |
| **Fusion Reactivity & τ_E** | ⟨σv⟩ (m³/s), τ_E (s, log y) | `sigma_v_m3_s`, `tau_e_s` | ⟨σv⟩ is a strong function of T. τ_E is constant during the pulse (n, B fixed). |
| **Q(t) & Bremsstrahlung Ratio** | Q(t) = P_fusion/(P_brem+P_cond); P_brem/P_fusion | computed in callback | Q=1 line shown. Ratio <1 means fusion wins over brem. |
| **Net Power dW/dt** | net power (W), zero line | `net_power_w` | Negative → plasma cooling. Zero crossing = W at extremum. Positive → plasma heating. |
| **Compression: Pre vs Post** | Grouped bar: T, n, V (pre vs post) | `parameters` dict | Verify T ratio = Rᶜ^(2/3), n ratio = Rᶜ, V ratio = 1/Rᶜ. |
| **Electron Density n_e** | n_e = 1.5·n_final (constant line) | `n_e_m3` | Should be flat (n is fixed post-compression). Value = 1.5 × n₀ × Rᶜ. |
| **Ignition Boundary** (heatmap) | log₁₀(Q) vs log₁₀(n) and log₁₀(Rᶜ); Q=1 white contour | `Q_map`, `ignition_mask` | Contour at Q=1 should shift right/up with higher B. Cyan/red dotted lines show τ_E × 0.3/3.0 uncertainty. |

---

<a name="spot-checks"></a>

## 7. Manual Spot-Check Worksheet

Use these to verify the live simulation without running code.

### PWR checks (run in UI, verify by hand)

**SS-1: Nominal steady-state (zero all inputs)**
Expected at t = 60 s (after initial transient settles):
- Power: 3 000 MWth (100%)
- T_fuel: ~627 °C (900 K)
- T_coolant: ~317 °C (590 K)
- Total reactivity: 0 pcm
- Xenon reactivity: between −2 500 and −3 000 pcm (check after 50 h accelerated)

**SS-2: +500 pcm rod insertion**
Expected immediate response (within 1 s):
- n rises rapidly → P rises above 3 000 MW
- Doppler feedback kicks in: ΔT_f ≈ +500/2.5 = +200 K upward → ρ_D ≈ −500 pcm → new equilibrium
- New steady power: slightly above 3 000 MW (Doppler doesn't fully cancel rod, moderator also contributes)
- Verify: reactivity breakdown shows rod +500, Doppler ≈ −450 to −500, net ≈ 0 at equilibrium

**SS-3: Reduce flow to 0.5× nominal**
Expected (slow transient, ~30 s):
- **Initial feedback magnitude:** if T_coolant were to rise by the full 25 K implied by the
  new steady-state heat-balance (see Section 1C), ρ_m would be −35 × 25 = −875 pcm. However,
  this −875 pcm represents the *peak* initial feedback, not the final steady state.
- **Self-correction:** the −875 pcm drives power down. As power falls, q_fc drops, meaning
  less heat is added to the coolant, so T_coolant partially recovers (falls back from its peak).
  The final new SS T_c lies *between* the original value and the 25 K peak-rise value.
- **Actual new SS:** power and T_c settle at a coupled equilibrium where the reduced-flow heat
  removal exactly balances the (reduced) power output. To find it exactly, read the stabilized
  values from the simulation after ~120 s.
- Power decreases to new SS value < 3 000 MW (how much less depends on the exact α_m / τ_th balance).
- ΔT (fuel−coolant) chart: narrower gap (less heat flux through less coolant)

**SS-4: 500 ppm boron addition**
Expected (instantaneous reactivity change):
- ρ_B = −10 × 500 = −5 000 pcm immediate negative step
- Reactor goes subcritical → n decays → P decays
- No self-regulation: boron is not a feedback effect, it stays at −5000 pcm
- Only way to recover: withdraw rods or reduce boron

### Helion checks (run in UI, verify by hand)

**H-1: Default parameters (T₀=20 keV, n₀=1e21, Rᶜ=10, B=5 T, V=1 m³, Δt=10 μs)**
Expected:
- T_final = 20 × 10^(2/3) = **92.8 keV** (check compression chart)
- n_final = 1e22 m⁻³ (check n_e = 1.5e22 m⁻³)
- W₀ ≈ 5.6e7 J ≈ 56 MJ (check W_initial metric)
- τ_E = 5e-5 × (1e22/1e21) × (5/5)² = **500 μs** (much > 10 μs pulse → check τ_E trace)
- P_cond at t=0: 5.6e7 / 5e-4 = **1.12e11 W = 112 GW** — dominates all losses
- Q << 1 expected (energy lost to conduction far exceeds fusion yield)

**H-2: High compression (Rᶜ=500, scenario "high_compression")**
- T_final = 20 × 500^(2/3) = 20 × 63.0 = **1 260 keV** (clamped to 190 keV for ⟨σv⟩)
- n_final = 500 × 1e21 = 5e23 m⁻³
- τ_E = 5e-5 × (5e23/1e21) × 1 = 5e-5 × 500 = **25 ms** (= 2.5e-2 s) >> pulse duration → conduction negligible
- Expected Q ≈ 0.09 (per scenario tests in `test_helion_scenarios.py`)

**H-3: Ignition condition (Rᶜ=1000, n₀=1e23)**
- T_final = 20 × 1000^(2/3) = 20 × 100 = **2 000 keV** → clamped at 190 keV
- n_final = 1e26 m⁻³
- P_fusion at 190 keV: ⟨σv⟩ ≈ 2.68e-22 m³/s
  → P_fusion = (1e26)²/4 × 2.68e-22 × 2.93e-12 × V_final
  → extremely large → Q > 1 expected

**H-4: Bremsstrahlung dominance check (T₀=2 keV, low Rᶜ=3)**
- T_final = 2 × 3^(2/3) = 2 × 2.08 = **4.16 keV** (well below 30 keV break-even)
- ⟨σv⟩ at 4 keV ≈ 6e-29 m³/s (from B-H table: very small)
- P_fusion ≪ P_brem → ratio chart shows P_brem/P_fusion >> 1
- W decays immediately → T drops → Q << 1

---

## 8. Missing Helion UI Inputs — Implementation Notes

When adding the three missing inputs to the Helion panel, wire them as follows:

**P_heat (external heating power):**

`P_heat_w` is a kernel-level parameter already present in `integrate_helion_pulse(parameters,
P_heat_w=0.0, ...)`. It is currently hardcoded to 0.0 in `helion_runner.run_helion_pulse()` —
the UI has no control for it. Adding it requires two changes only:
1. UI: add a numeric input (MW is a readable scale; convert to W before passing).
2. Runner: read the value and forward it:
```python
# orchestrator/helion_runner.py  run_helion_pulse()
P_heat_w = raw_inputs.get("p_heat_w", 0.0)   # W
result = integrate_helion_pulse(params, P_heat_w=P_heat_w)
```
No ODE change needed.

**Pulse repetition rate f_rep:**
```python
# Post-process after integrate_helion_pulse returns
E_fusion_j = float(np.trapezoid(result.P_fusion, result.t))   # J per pulse
mean_fusion_power_w = E_fusion_j * f_rep                       # W average
```
This is purely additive post-processing. UI: number input in Hz (0.01–10).

**Number of pulses n_pulses:**
```python
total_fusion_energy_j = n_pulses * float(np.trapezoid(result.P_fusion, result.t))
total_input_energy_j  = n_pulses * (result.W_initial + P_heat_w * result.t[-1])
cumulative_Q = total_fusion_energy_j / total_input_energy_j  # same as single-pulse Q
```
UI: integer input (1–10 000).

The ODE itself (`plasma_energy_rhs_kernel`) does not need to change. These are all
wiring-level additions to `helion_runner.py` and `helion_panel.py`.

---

*Created 2026-05-19. Update whenever a new control input, output channel, or chart is added.*
*All numerical checks reference PHYSICSLOGIC.md (now in docs/) for the verbatim equations.*
