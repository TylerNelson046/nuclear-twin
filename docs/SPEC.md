Nuclear Reactor Digital Twin Platform
Engineering Specification Document
 
Table of Contents
1.	Executive Summary
2.	Project Context
3.	Constraints & Assumptions
4.	First Principles Decomposition
5.	Requirements Specification
6.	System Architecture
7.	Testing & Validation Strategy
8.	Milestone & Iterative Development Plan
9.	Glossary
10.	References & Resources
 












1. Executive Summary
This project is a multi-reactor nuclear digital twin platform implementing simplified analytical models of three distinct reactor architectures: a Pressurized Water Reactor (PWR), a Field-Reversed Configuration (FRC) fusion reactor modeled after Helion Energy’s pulsed plasma approach, and a Molten Salt Reactor (MSR). The platform is developed against a formal engineering specification in which every software subsystem is traced directly to its governing physical law, implemented in a Python/Dash/SciPy stack.
Each reactor twin exposes its core physics through live parameter control and real-time system response visualization. The PWR and MSR share a modular neutron kinetics engine built on the 6-group point kinetics equations with delayed neutron precursor dynamics, grounded in IAEA-published nuclear data. The Helion FRC twin models plasma energy balance and D-He3 fusion reactivity using the Bosch-Hale parametrization, enabling pulse-by-pulse yield simulation and ignition boundary mapping across compression and density parameter space. The MSR extends the shared kinetics core with precursor drift dynamics unique to liquid-fueled systems.
The platform is designed in three planned phases: PWR as the foundational build establishing the shared simulation architecture, Helion validating the platform's modularity across physically distinct reactor domains, and MSR as the payoff demonstrating the reusability of the kinetics infrastructure.
The platform serves two audiences simultaneously. For professionals, the models are analytically honest: 6-group point kinetics with temperature feedback coefficients grounded in IAEA data, D-He3 reactivity from the Bosch-Hale parametrization, and MSR precursor transport dynamics not found in any comparable open-source tool. For students new to reactor physics, it provides intuitive access to the feedback mechanisms that make nuclear systems behave the way they do — Doppler broadening, delayed neutron dependence, the stark difference between fission and fusion transient timescales. 
 
2. Project Context
2.1 Project Purpose
This project started as a way to actually understand how these reactors behave. Building the physics yourself enables a level of understanding that reading doesn’t achieve. The secondary objective is demonstrating applied software engineering capability at the intersection of scientific computing, physics simulation, and interactive data visualization. 
2.2 Domain Context
Three reactor architectures are modeled at reduced-order analytical fidelity:
Pressurized Water Reactor (PWR): The dominant commercial reactor design globally, comprising roughly 70% of the world's operating nuclear fleet. PWRs use enriched uranium fuel in solid ceramic pellets, with pressurized light water serving as both coolant and neutron moderator. Their dynamics are governed by neutron kinetics coupled with thermal-hydraulic feedback, with multiple inherent self-regulating mechanisms including the Doppler effect and moderator temperature coefficient.
Helion FRC Fusion Reactor: Helion Energy's approach to fusion uses a Field-Reversed Configuration (FRC). A compact self-organizing plasma configuration compressed by magnetic fields to achieve fusion conditions. Unlike tokamak designs, Helion's system operates in discrete pulses at microsecond timescales, burning deuterium and helium-3 fuel. The physics governing energy balance and fusion yield are well-established in published literature despite Helion's proprietary engineering parameters remaining confidential.
Molten Salt Reactor (MSR): An advanced fission concept in which the nuclear fuel is dissolved directly into a molten fluoride or chloride salt that simultaneously serves as the coolant. This eliminates the distinction between fuel and coolant that defines solid-fueled reactors, producing unique dynamic behavior, the transport of delayed neutron precursors out of the core with the flowing salt, reducing the effective delayed neutron fraction and altering transient response in ways not present in PWR or BWR designs.
2.3 Development Philosophy
This specification applies First Principles Decomposition throughout — each reactor's software subsystems are derived directly from their governing physical laws. Every simplification is named, justified, and documented. The reduced-order modeling approach is not a limitation but rather a deliberate scoping decision consistent with how reactor designers and analysts use these models in early-stage work.
 
3. Constraints & Assumptions
3.1 Technical Constraints
•	All physics models are zero-dimensional (point models) or single-node lumped parameter models. No spatial flux distributions, no multi-channel thermal hydraulics, no 3D geometry.
•	ODE integration performed by SciPy solve_ivp with the Radau stiff solver.
•	Real-time UI response target of 500ms for parameter changes. Physics loops accelerated via Numba JIT compilation where necessary.
•	Deployment on free-tier cloud hosting (Render or Railway). No GPU compute, no high-memory instances.
•	No proprietary data sources. All nuclear constants sourced from publicly available IAEA databases and peer reviewed literature


3.2 Domain Constraints
•	PWR model represents a generic large light water reactor. Not calibrated to any specific plant design (Westinghouse AP1000, EPR, or otherwise).
•	Helion FRC model uses published FRC physics and Bosch-Hale D-He3 reactivity data. Helion's proprietary confinement performance, compression hardware specifications, and direct energy conversion efficiency are not represented and cannot be inferred from public sources.
•	MSR model represents a generic graphite-moderated thermal MSR. Not calibrated to any specific design (MSRE, LFTR, or otherwise).
•	Regulatory compliance, safety system modeling, and licensing basis calculations are out of scope.
3.3 Explicit Out-of-Scope Items
•	Spatial neutron flux distributions or multi-group diffusion theory
•	Two-phase flow and departure from nucleate boiling (DNB) calculations
•	Fuel burnup, isotopic depletion, or long-term reactivity management
•	Control system automation or PID controller modeling
•	Accident progression beyond early transient phase
•	Salt chemistry, redox potential, or fission product solubility for MSR
•	Neutron transport (Monte Carlo or deterministic)
3.4 Assumptions
•	Reactor operates in a single homogeneous core region with uniform neutron flux
•	Delayed neutron precursor data from Keepin 6-group parametrization for U-235 fission is representative for all fission reactors modeled
•	Feedback coefficients (Doppler, moderator temperature) treated as constants across the operating range modeled
•	D-He3 fuel mix assumed at 50/50 by number density for Helion twin
•	MSR loop geometry (core length, external loop length) treated as fixed parameters
•	All simulations begin from a defined steady-state initial condition
•	Temperature feedback coefficients are evaluated at nominal operating conditions and held fixed

4. First Principles Decomposition
4.1 Philosophy
Each reactor subsystem in this platform is derived directly from its governing physical laws. The decomposition proceeds in three steps for each reactor: identify the fundamental conservation laws and interaction physics, derive the reduced-order equations that capture essential behavior within stated constraints, and map each equation to a software subsystem with defined inputs, outputs, and coupling.

4.2 PWR Decomposition
Governing Physics → Software Subsystems:
Conservation of neutrons governs reactor power evolution. Neutrons are produced by fission (prompt and delayed), absorbed by fuel, moderator, structure, and control poisons, and leak from the core. In the point kinetics approximation, this reduces to a coupled ODE system for total neutron population n(t) and delayed precursor concentrations Cᵢ(t). → Point Kinetics Engine (11-state ODE system)
Conservation of energy governs heat transfer from fuel to coolant. Fission energy deposits in the fuel, conducts through the pellet and cladding to the coolant, and is carried to the steam generators by coolant flow. → Thermal-Hydraulic Module (2-state lumped parameter model)
Doppler broadening governs how T_f affects neutron absorption. Rising U-238 temperature broadens resonance absorption peaks, reducing reactivity — the primary inherent safety mechanism in a PWR. → Doppler Feedback Calculator (algebraic: T_f → ρ)
Neutron moderation physics governs how T_c affects the neutron energy spectrum. Hotter, less dense water moderates neutrons less effectively, reducing thermal fission probability. Combined with Doppler, this gives the PWR two independent negative temperature feedback mechanisms. → Moderator Feedback Calculator (algebraic: T_c → ρ)
Fission product accumulation governs Xe-135 and I-135 buildup. I-135 is produced from fission and decays to Xe-135; Xe-135 is also produced directly from fission, decays radioactively, and is destroyed by neutron absorption. Net xenon worth at any time depends on reactor power history. → Xenon/Iodine Kinetics Module (2-state ODE system)
Chemical neutron absorption governs boron reactivity control. Dissolved boric acid absorbs neutrons proportional to concentration C_B, providing long-term reactivity management. → Boron Worth Calculator (algebraic: C_B → ρ)
 
4.3 Helion FRC Decomposition
Governing Physics → Software Subsystems:
Conservation of energy in a plasma governs the evolution of plasma thermal energy W(t). Energy is added by external heating and fusion self-heating, and lost through radiation and conduction across magnetic field lines. → Plasma Energy Balance ODE (1-state, W(t))
Nuclear reaction physics governs fusion power production. The D-He3 fusion rate depends on T and n through the reactivity ⟨σv⟩, computed from the Bosch-Hale parametrization. → Fusion Reactivity Module (Bosch-Hale lookup: T → ⟨σv⟩ → P_fusion)
Adiabatic thermodynamics governs plasma heating during magnetic compression. Compressing plasma volume increases T following the adiabatic relation for an ideal plasma. → Compression Heating Calculator (algebraic: Rᶜ → T_final)
Radiation physics governs Bremsstrahlung energy loss. Electrons decelerated in the Coulomb field of ions emit X-ray radiation ∝ n²T^(1/2), representing an unavoidable energy loss channel. → Radiation Loss Calculator (algebraic: n, T → P_rad)
Plasma confinement physics governs thermal conduction loss through the magnetic field, modeled as an empirical τ_E scaling — the dominant uncertainty in the Helion model. → Confinement Loss Calculator (empirical scaling: n, B → τ_E → P_cond)
 
4.4 MSR Decomposition
Governing Physics → Software Subsystems:
Conservation of neutrons with precursor transport governs MSR neutron kinetics. Unlike a PWR, delayed neutron precursors are dissolved in the salt and transported out of the core, reducing their effectiveness. Precursors decaying outside the core produce delayed neutrons that do not sustain the chain reaction. → Modified Point Kinetics Engine (shared PWR kinetics core + drift modification terms)
Precursor loop transit governs how Cᵢ return to the core after circulating through the external loop. Precursors decay exponentially during transit; the surviving fraction depends on v_salt and λᵢ. → Precursor Return Calculator (delay ODE: τ_loop, λᵢ → Cᵢ^return)
Conservation of energy in flowing salt governs T_salt evolution. Unlike a PWR, the MSR salt is simultaneously fuel and coolant — fission energy deposits directly into the flowing medium. → Salt Thermal-Hydraulic Module (1-state, fuel-coolant unified)
Temperature reactivity feedback governs MSR self-regulation. The strong negative temperature coefficient of the salt (combined fuel + moderator effect) provides inherent load-following behavior. → Salt Temperature Feedback Calculator (algebraic: T_salt → ρ_temp)
 
 
4.5 Governing Equations
All 24 equations implemented in the platform are grouped below by reactor type. Each equation is followed by a 1–2 sentence description. All variables and constants are defined in Section 4.5.4.
 
4.5.1 PWR
Eq. 1 — Neutron Population ODE dn/dt = [(ρ − β_eff) / Λ] · n + Σᵢ λᵢ Cᵢ
The neutron population evolves as the sum of prompt fission production, delayed neutron emission from precursor decay, and losses from absorption and leakage. This is the master equation of reactor kinetics; all power transients are governed by it.
 
Eq. 2 — Precursor Group ODEs (i = 1, …, 6) dCᵢ/dt = (βᵢ / Λ) · n − λᵢ Cᵢ
Each of the 6 Keepin precursor groups is produced proportional to the fission rate and lost by radioactive decay. Together with Eq. 1, these form the 7-state point kinetics core.
 
Eq. 9 — Fuel Temperature ODE m_f c_f · dT_f/dt = γ_f · P − (T_f − T_c) / R_fc
Fuel temperature rises with deposited fission power and falls as heat conducts through the gap and cladding to the coolant. T_f is the direct driver of Doppler feedback.
 
Eq. 10 — Coolant Temperature ODE m_c c_c · dT_c/dt = (T_f − T_c) / R_fc − ṁ c_c (T_c − T_in)
Coolant temperature rises from heat conducted in from the fuel and falls as flowing coolant carries energy out of the core. T_c drives moderator feedback.
 
Eq. 11 — Doppler Reactivity Feedback ρ_D = α_D · (T_f − T_f,0)
Rising T_f broadens U-238 resonance absorption peaks, reducing k_eff proportionally. α_D is negative, making this an inherent self-limiting mechanism.
 
Eq. 12 — Moderator Temperature Feedback ρ_m = α_m · (T_c − T_c,0)
Hotter coolant is less dense, moderates neutrons less effectively, and shifts the spectrum to higher energies, reducing thermal fission probability. α_m is negative, providing a second independent temperature feedback.
 
Eq. 13 — Boron Reactivity Worth ρ_B = ω_B · C_B
Dissolved H₃BO₃ absorbs neutrons in proportion to concentration, linearly reducing reactivity. This is the primary mechanism for long-term reactivity management.
 
Eq. 14 — I-135 ODE dI/dt = γ_I Σ_f φ − λ_I I
I-135 is produced directly from fission and lost by radioactive decay to Xe-135. Its buildup timescale (~6.7 hr half-life) sets the lag in xenon transients.
 
Eq. 15 — Xe-135 ODE dX/dt = γ_X Σ_f φ + λ_I I − (λ_X + σ_aX φ) X
Xe-135 is produced from both direct fission and I-135 decay, and destroyed by radioactive decay and neutron absorption burnout. It represents the largest time-dependent reactivity effect in an operating PWR.
 
4.5.2 Helion
Eq. 16 — Plasma Energy Balance ODE dW/dt = P_heat + P_fusion,self − P_Brem − P_cond
The total plasma thermal energy evolves as the net of heating inputs (external and fusion self-heating) minus radiation and conduction losses. All other Helion equations feed into the terms on the right-hand side of this expression.
 
Eq. 17 — D-He3 Fusion Power P_fusion = (n² / 4) · ⟨σv⟩(T) · E_DHe3 · V
Fusion power scales with the square of plasma density and the temperature-dependent reactivity ⟨σv⟩. Only charged products (p + He-4) are produced — no neutrons — so all fusion energy is retained in the plasma.
 
Eq. 18 — Bosch-Hale Reactivity ⟨σv⟩ = C₁ θ² / (ξ B_G²) · √(ξ / (m_r c² T³)) · exp(−3ξ)
where θ = T / [1 − T(C₂ + T(C₄ + TC₆)) / (1 + T(C₃ + T(C₅ + TC₇)))], ξ = (B_G² / 4θ)^{1/3}, B_G = π α_f Z₁Z₂ √(2m_r c²)
This analytical fit to thermonuclear cross-section data computes ⟨σv⟩ as a function of T across the range 0.5–190 keV. Constants C₁–C₇ are from Bosch & Hale (1992), Table IV; platform enforces the valid temperature bounds (R-06).
 
Eq. 19 — Adiabatic Compression Heating T_final = T_initial · Rᶜ^{2/3}
When the plasma is compressed faster than its energy loss timescale, temperature rises adiabatically with the compression ratio Rᶜ raised to the power 2/3 = γ − 1 for a monatomic ideal plasma (γ = 5/3). Number density scales simultaneously as n_final = n_initial · Rᶜ.
 
Eq. 20 — Bremsstrahlung Radiation Loss P_Brem = C_B · Z_eff · n_e² · T^{1/2} · V
Electrons braking in the Coulomb field of ions radiate power proportional to n² T^{1/2}, representing an unavoidable QED energy loss. This loss grows with Z_eff, penalizing heavier ion species in the plasma mix.
 
Eq. 21 — Thermal Conduction Loss P_cond = W / τ_E
Thermal energy leaks across magnetic field lines at a rate set by the empirical energy confinement time τ_E. τ_E is the dominant model uncertainty in the Helion twin (Risk R-02); results are reported as a range across the documented τ_E scaling uncertainty band.
 
4.5.3 MSR
Eq. 22 — MSR Precursor ODE with Drift (i = 1, …, 6) dCᵢ/dt = (βᵢ/Λ) n − λᵢ Cᵢ − (1/τ_core) Cᵢ + Cᵢ^return(t)
Compared to the PWR (Eq. 2), two additional terms appear: a flow-out term removing precursors from the core at rate 1/τ_core, and a return term adding back precursors that survived the external loop. Precursors that decay outside the core produce no useful delayed neutrons.
 
Eq. 23 — Precursor Return After External Loop Transit Cᵢ^return(t) = (1/τ_core) · Cᵢ(t − τ_loop) · e^{−λᵢ τ_loop}
Of the precursors that left the core τ_loop seconds ago, only the fraction e^{−λᵢ τ_loop} survives to return. The history dependence Cᵢ(t − τ_loop) requires storing full precursor history and is the source of numerical stiffness at high v_salt (Risk R-03).
 
Eq. 24 — Effective β with Precursor Drift β_eff,flow = Σᵢ βᵢ · [λᵢ τ_core / (1 + λᵢ τ_core)] · [1 + e^{−λᵢ τ_loop} / (1 + λᵢ τ_core − e^{−λᵢ τ_loop})]
β_eff,flow is the steady-state solution of Eq. 22 at constant power; it decreases monotonically as v_salt increases. This reduction in effective delayed neutron fraction makes the MSR respond faster than a PWR and is the defining characteristic of MSR dynamics in this platform.
 
4.5.4 Variables & Constants
Symbol	Definition	Units
n	Neutron population (proportional to power)	—
ρ	Total reactivity	pcm (≡ 10⁻⁵)
β_eff	Effective delayed neutron fraction; β_eff = Σᵢ βᵢ (~0.0065 for U-235 thermal)	—
βᵢ	Delayed neutron fraction for precursor group i	—
Λ	Prompt neutron generation time (~10⁻⁵ s for PWR)	s
λᵢ	Radioactive decay constant of precursor group i	s⁻¹
Cᵢ	Concentration of delayed neutron precursor group i	neutrons/cm³
m_f, m_c	Lumped thermal mass of fuel and coolant	kg
c_f, c_c	Specific heat of fuel and coolant	J/kg/K
T_f	Fuel temperature	K
T_c	Coolant (moderator) temperature	K
T_f,0	Reference fuel temperature	K
T_c,0	Reference coolant temperature	K
T_in	Coolant inlet temperature	K
P	Total fission power	W
γ_f	Fraction of fission power deposited in fuel (~0.97)	—
R_fc	Effective fuel-to-coolant thermal resistance	K/W
ṁ	Coolant mass flow rate	kg/s
α_D	Doppler temperature coefficient (~−2 to −3 pcm/°C)	pcm/K
α_m	Moderator temperature coefficient (~−20 to −50 pcm/°C)	pcm/K
ω_B	Differential boron worth (~−10 pcm/ppm)	pcm/ppm
C_B	Dissolved boron concentration	ppm
ρ_D	Doppler reactivity contribution	pcm
ρ_m	Moderator temperature reactivity contribution	pcm
ρ_B	Boron reactivity contribution	pcm
ρ_Xe	Xenon reactivity worth	pcm
ρ_total	Total reactivity: ρ_rod + ρ_D + ρ_m + ρ_B + ρ_Xe	pcm
I	I-135 number density	atoms/cm³
X	Xe-135 number density	atoms/cm³
γ_I	I-135 fission yield (0.0639)	—
γ_X	Xe-135 direct fission yield (0.00237)	—
Σ_f	Macroscopic fission cross section	cm⁻¹
φ	Thermal neutron flux	neutrons/cm²/s
λ_I	I-135 decay constant; 2.87×10⁻⁵ s⁻¹ (t₁/₂ = 6.7 hr)	s⁻¹
λ_X	Xe-135 decay constant; 2.09×10⁻⁵ s⁻¹ (t₁/₂ = 9.2 hr)	s⁻¹
σ_aX	Xe-135 neutron absorption cross section; 2.6×10⁶ barns = 2.6×10⁻¹⁸ cm²	cm²
ν	Average neutrons produced per fission	—
W	Total plasma thermal energy	J
P_heat	External plasma heating power	W
P_fusion,self	Fusion power fraction retained in plasma as heat	W
P_Brem	Bremsstrahlung radiation loss power	W
P_cond	Thermal conduction loss power	W
P_fusion	Total D-He3 fusion power	W
n (plasma)	Total plasma number density	m⁻³
⟨σv⟩	D-He3 thermonuclear reactivity	m³/s
E_DHe3	Energy per D-He3 reaction; 18.3 MeV = 2.93×10⁻¹² J	J
V	Plasma volume	m³
C₁–C₇	Bosch-Hale fit constants for D-He3 (Bosch & Hale 1992, Table IV)	various
θ	Bosch-Hale intermediate temperature variable	keV
ξ	Bosch-Hale intermediate variable; ξ = (B_G² / 4θ)^{1/3}	—
B_G	Gamow constant; B_G = π α_f Z₁Z₂ √(2m_r c²)	—
α_f	Fine-structure constant (~1/137)	—
Z₁, Z₂	Ion charge numbers (D: 1, He3: 2)	—
m_r	Reduced mass of the D-He3 system	kg
T (plasma)	Plasma (electron) temperature	keV
Rᶜ	Compression ratio; V_initial / V_final	—
γ (adiabatic)	Adiabatic index for monatomic ideal plasma (5/3)	—
C_B (Brem)	Bremsstrahlung coefficient; 5.35×10⁻³⁷ W·m³/keV^{1/2}	W·m³/keV^{1/2}
Z_eff	Effective plasma ion charge number (1 < Z_eff < 2 for D-He3)	—
n_e	Electron number density	m⁻³
τ_E	Energy confinement time (empirical FRC scaling)	s
τ_core	Core transit time; L_core / v_salt	s
τ_loop	External loop transit time; L_loop / v_salt	s
v_salt	Salt flow velocity	m/s
L_core	Active core length	m
L_loop	External loop length	m
β_eff,flow	Effective delayed neutron fraction under flowing-salt conditions	—

4.6 Numerical Implementation
4.6.1 Numerical Methods
All ODE systems are integrated using an adaptive-step, explicit Runge-Kutta solver (RK45) with a stiff fallback to an implicit BDF solver when the stiffness ratio exceeds a defined threshold. The point kinetics equations (Eqs. 1–2) are inherently stiff — Λ ~ 10⁻⁵ s while xenon time constants exceed 10⁴ s, a ratio of ~10⁹ — so the BDF solver is the default for all PWR and MSR kinetics. The Helion plasma energy balance (Eq. 16) is non-stiff during the compression phase and uses RK45 unless τ_E drops below 10× the integration timestep, at which point the solver switches to BDF automatically.
Solver tolerances are set to absolute tolerance ε_abs = 10⁻⁸ and relative tolerance ε_rel = 10⁻⁶ for all systems. These values were chosen to keep global truncation error below 0.1 pcm in reactivity and below 0.01 K in temperature across a 3600 s transient — verified by convergence testing against a reference solution at ε_rel = 10⁻¹⁰.
 
4.6.2 State Management
The full simulation state vector is defined for each reactor type as follows:
PWR state vector (11 states):
Index	Variable	Description
0	n	Neutron population
1–6	C₁ … C₆	Delayed neutron precursor concentrations (Keepin groups)
7	T_f	Fuel temperature
8	T_c	Coolant temperature
9	I	I-135 number density
10	X	Xe-135 number density
Helion state vector (1 state, with algebraic outputs):
Index	Variable	Description
0	W	Plasma thermal energy
P_fusion, P_Brem, P_cond, and T are computed algebraically from W and n at each timestep; they are outputs, not integrated states.
MSR state vector (9 states + history buffer):
Index	Variable	Description
0	n	Neutron population
1–6	C₁ … C₆	In-core precursor concentrations
7	T_salt	Salt temperature
8	ρ_total	Total reactivity (algebraic, updated each step)
The precursor history buffer (see Section 4.6.3) is maintained separately from the ODE state vector and is not passed to the solver directly.
All state vectors are initialized from user-supplied or default steady-state values. Steady-state initialization is performed analytically before the solver is launched: n₀ is normalized to the specified initial power level, Cᵢ,₀ = (βᵢ / λᵢ Λ) · n₀, I₀ = γ_I Σ_f φ₀ / λ_I, and X₀ is solved from the steady-state form of Eq. 15 at initial flux φ₀.
 
4.6.3 Delay-System Implementation
The MSR precursor return term Cᵢ^return(t) (Eq. 23) introduces a delay-differential equation (DDE): the right-hand side of Eq. 22 depends on Cᵢ evaluated at t − τ_loop, not at t. Standard ODE solvers cannot handle this directly. The platform implements the following approach:
•	A fixed-size circular buffer stores the value of each Cᵢ at every past timestep for a window of duration τ_loop + δ, where δ is one maximum timestep.
•	At each solver evaluation, Cᵢ(t − τ_loop) is retrieved from the buffer using linear interpolation between the two bracketing stored values.
•	The buffer is pre-filled with the steady-state initial value Cᵢ,₀ for t < 0, consistent with the assumption that the reactor has been operating at steady state before t = 0.
•	If the adaptive solver reduces its stepsize, the buffer resolution is sufficient because it stores values at every accepted step, not at fixed intervals.
τ_loop is treated as a fixed parameter during a simulation run. Variable-τ_loop scenarios (e.g., pump coastdown) require re-initializing the buffer and are flagged as a known limitation (Risk R-03).
 
4.6.4 Unit Consistency
All internal calculations use SI base units unless an exception is explicitly noted in the table below. User-facing inputs and outputs may be displayed in engineering units; conversion is applied at the interface layer only, never inside the solver.
Quantity	Internal Unit	Display Unit	Conversion
Reactivity	dimensionless (Δk/k)	pcm	× 10⁵
Temperature	K	°C or K (user toggle)	− 273.15
Neutron flux φ	neutrons/cm²/s	same	—
Cross sections	cm⁻¹ or cm²	same	—
Plasma temperature T	keV	keV	—
Plasma energy W	J	MJ	× 10⁻⁶
Plasma density n	m⁻³	m⁻³	—
Boron concentration	ppm (mass)	ppm	—
Time	s	s, min, or hr (user toggle)	÷ 60, ÷ 3600
A unit consistency check is run at initialization: all user-supplied parameters are validated against expected ranges in internal units before the solver is launched. Out-of-range values produce a warning; values outside hard physical bounds (e.g., T_f < 0 K, n < 0) abort initialization with an error.
 
4.6.5 Solver Robustness
The following failure modes are explicitly handled:
Stiffness detection. If the RK45 solver reduces its stepsize below h_min = 10⁻¹² s without converging, it raises a stiffness flag and the integration is automatically restarted using the BDF solver from the last accepted state. This transition is logged.
Negative state values. Physical quantities n, Cᵢ, I, X, W, T must remain non-negative. If the solver produces a negative value for any of these states, the step is rejected, the stepsize is halved, and a floor of 0 is applied only after two consecutive rejections. Repeated flooring triggers a solver warning in the output log.
Reactivity excursion limiting. If |ρ| > 0.01 (1000 pcm, ~1.5β for U-235), the solver pauses, logs a superprompt-critical warning, and — depending on the simulation mode — either continues with a reduced maximum stepsize of 10⁻⁷ s or halts and returns the state at the point of excursion for user inspection.
Xenon oscillation resolution. The xenon/iodine subsystem (Eqs. 14–15) can develop spatial oscillations in a real core that the point model cannot capture. The platform flags any simulation in which the xenon reactivity worth |ρ_Xe| oscillates with period < 2 hr and amplitude > 50 pcm as potentially unphysical, and displays a caution note in the output.
Delay buffer underrun. If the adaptive solver requests Cᵢ(t − τ_loop) for a time earlier than the buffer window, the buffer returns the steady-state initial value and logs a buffer underrun warning. This can occur during the first τ_loop seconds of a transient and is expected behavior, not an error.
Helion temperature bounds. The Bosch-Hale parametrization (Eq. 18) is only valid for T ∈ [0.5, 190] keV. If the plasma temperature exits this range during integration, ⟨σv⟩ is clamped to the boundary value, a bounds warning is logged, and the simulation continues. Results during clamped intervals should be treated as extrapolations (R-06).


5. Requirements Specification
5.1 Functional Requirements
Platform-level:
•	FR-01: The platform shall provide independent simulation environments for PWR, Helion FRC, and MSR reactor types accessible from a unified dashboard interface.
•	FR-02: Each reactor twin shall support both real-time parameter control mode and predefined scenario simulation mode.
•	FR-03: The platform shall display a plain-language event log narrating physical system behavior in response to user inputs.
•	FR-04: Users shall be able to save and reload simulation states via HDF5 file export and import.
•	FR-05: The platform shall provide an accelerated time mode with user-selectable time multiplier (1×, 60×, 3600×) for slow transients including xenon dynamics.
PWR-specific:
•	FR-06: The PWR twin shall simulate neutron population dynamics using 6-group point kinetics with Keepin delayed neutron parameters for U-235.
•	FR-07: The PWR twin shall implement Doppler and moderator temperature feedback as independent reactivity components with physically grounded coefficients.
•	FR-08: The PWR twin shall model Xe-135 and I-135 dynamics including equilibrium buildup, xenon peak following shutdown, and neutron burnout at power.
•	FR-09: The PWR twin shall accept the following user inputs: control rod insertion depth (%), coolant inlet temperature (°C), coolant flow rate (% of nominal), boron concentration (ppm), and initial power level (% of rated).
•	FR-10: The PWR twin shall display: thermal power (MWth) vs time, fuel and coolant temperature vs time, reactivity breakdown by component (rod, Doppler, moderator, boron, xenon) vs time, and xenon/iodine concentration vs time.
Helion FRC-specific:
•	FR-11: The Helion twin shall compute plasma energy balance including fusion heating, Bremsstrahlung radiation loss, and conduction loss over a single pulse.
•	FR-12: The Helion twin shall compute D-He3 fusion reactivity using the Bosch-Hale parametrization.
•	FR-13: The Helion twin shall accept the following user inputs: injected plasma energy (MJ), magnetic compression ratio, initial plasma density (m⁻³), initial plasma temperature (keV), and pulse duration (μs).
•	FR-14: The Helion twin shall display: plasma temperature vs time, fusion power vs time, energy gain factor Q, power loss channel breakdown (radiation vs conduction), and a 2D ignition boundary heatmap across compression ratio × plasma density parameter space.
MSR-specific:
•	FR-15: The MSR twin shall implement modified point kinetics with precursor drift terms, computing β_eff,flow as a function of salt velocity.
•	FR-16: The MSR twin shall model precursor return with exponential decay during external loop transit.
•	FR-17: The MSR twin shall accept the following user inputs: salt flow rate (% of nominal), control rod position (%), fission product removal rate, and initial power level (% of rated).
•	FR-18: The MSR twin shall display: thermal power vs time, salt temperature vs time, β_eff,flow vs salt velocity, and reactivity component breakdown vs time.
5.2 Non-Functional Requirements
•	NFR-01: UI shall update within 500ms of any user parameter change on standard consumer hardware.
•	NFR-02: ODE solver shall complete each timestep integration in under 100ms without Numba acceleration; under 20ms with Numba on hot loops.
•	NFR-03: Platform shall be deployable as a web application accessible via public URL without user installation.
•	NFR-04: All reactor parameter inputs shall be bounded to physically plausible ranges with validation enforced at the Pydantic schema level before reaching the physics engine.
•	NFR-05: Simulation state files shall be human-inspectable (HDF5 with descriptive dataset names and units metadata).
•	NFR-06: Codebase shall maintain greater than 80% test coverage on physics engine modules.
5.3 Safety Requirements
•	SR-01: No user input combination shall cause the physics engine to produce an unhandled exception visible to the end user. All unphysical states shall return a defined error state with a plain-language explanation.
•	SR-02: Input bounds shall be derived from physical constraints, not arbitrary software limits. Each bound shall be documented with its physical justification in code comments.
•	SR-03: The platform shall display a persistent notice on all reactor twins stating that models are simplified analytical representations for educational purposes and do not represent licensed reactor safety analysis.
6. System Architecture

6.1 Data Flow
Real-time control mode: User adjusts parameter → Pydantic validates input bounds → Orchestrator updates reactor state → Physics engine integrates one timestep → State vector returned → Plotly traces updated → Event log narration generated → UI renders
Scenario mode: User selects scenario → Orchestrator loads scenario config from HDF5 → Physics engine integrates full scenario duration → Results stored to HDF5 → UI renders complete time-series → User can scrub through timeline
6.2 Shared Kinetics Core
The PWR and MSR share a common point kinetics implementation. The MSR engine subclasses or wraps the shared core, injecting precursor drift terms as additional ODE contributions. This ensures the kinetics mathematics is implemented and tested once, with MSR-specific modifications isolated and clearly identified. Switches cleanly between PWR, Helion, and MSR without state contamination.
6.3 Technology Stack
Layer	Tool	Purpose
IDE	Cursor Pro	Day-to-day implementation, autocomplete, multi-file editing
AI Agent	Claude Code Pro	Architecture, cross-file refactors, physics engine builds
Language	Python 3.11+	Primary implementation Language
UI Framework	Dash (Plotly)	React-compiled interactive dashboard
Charting	Plotly (built into Dash)	Time-series, heatmaps, parameter sweep plots
Styling	Dash Bootstrap Components	Responsive layout, professional theming
Physics Engine	NumPy + SciPy (solve_ivp)	ODE integration, all numerical simulation
Performance	Numba	JIT-compile hot physics loops
Data Validation	Pydantic	Reactor parameter schemas, input bounds
Data Storage	HDF5 via h5py	Save/load simulation runs
Delayed Neutron Data	IAEA Beta-Delayed Neutron DB	βeff, λᵢ, βᵢ group parameters
Fusion Reactivity Data	Bosch-Hale parametrization	D-He3 ⟨σv⟩
PWR Reference Data	IAEA PRIS Database	Real-world operating parameters
Physics Reference	Keepin 6-group DNP parameters	Precursor constants
Testing	pytest + pytest-benchmark	Unit tests, physics loop profiling
Dependency Management	Poetry	Reproducible environment
Linting	Ruff	Fast Python linting
Version Control	Git + GitHub	Source control, portfolio visibility
Deployment	Render or Railway	Free-tier Python/Dash hosting

7. Testing & Validation Strategy
7.1 Philosophy
Validation for a physics simulation has two distinct goals: verifying the software does what the code intends (software testing), and verifying the physics model produces results consistent with known physics (physics validation). Software correctness and physical correctness are both required and tested independently.
7.2 Software Testing
All physics engine modules shall achieve greater than 80% test coverage via pytest. Test categories:
•	Unit tests: Each equation implemented as a function is tested against hand-calculated values. Example: test_doppler_feedback() verifies that a 100°C fuel temperature rise with α_D = -2.5 pcm/°C returns exactly -250 pcm reactivity contribution.
•	Integration tests: Full ODE system integration tested for state conservation. Example: at steady state with zero external reactivity insertion, power shall remain constant within numerical tolerance across 1000 seconds of simulated time.
•	Boundary tests: All Pydantic-validated input bounds tested at minimum, maximum, and just outside limits.
•	Regression tests: Once a physically validated result is obtained, it is stored as a reference and tested against on every subsequent code change.
7.3 Physics Validation
PWR:
•	Steady-state criticality: at initial conditions with zero external reactivity, power shall be stable. Verified by running 1000 seconds with no inputs and confirming power drift < 0.1%.
•	Doppler self-regulation: a step reactivity insertion of +50 pcm shall produce a power excursion that stabilizes at a new equilibrium, not a runaway. The new equilibrium power shall be calculable analytically and matched within 5%.
•	Xenon equilibrium: after 50 hours simulated time at full power, xenon worth shall fall within the published range of -2500 to -3000 pcm for a commercial PWR.
•	Xenon peak: following a full-power shutdown, xenon concentration shall peak between 6 and 10 hours post-shutdown and return to near-zero by 40-50 hours. Peak xenon worth shall exceed equilibrium worth.
•	Prompt criticality verification: reactivity insertion of exactly +1$ (= +βeff) shall produce a divergent power excursion, not a controlled transient.
Helion FRC:
•	Energy conservation: integrated fusion energy output plus radiation and conduction losses shall equal input energy plus fusion yield within 1% across all simulated pulses.
•	Q boundary: at plasma conditions known from published FRC literature to be sub-ignition, Q shall be less than 1. At conditions expected to exceed ignition, Q shall exceed 1.
•	Temperature scaling: fusion power shall increase monotonically with temperature in the 10-100 keV range where D-He3 reactivity is increasing. Verified against tabulated Bosch-Hale values.
•	Bremsstrahlung dominance at low temperature: at plasma temperatures below ~30 keV, radiation losses shall exceed fusion power production, consistent with published D-He3 physics.
MSR:
•	β_eff,flow reduction: at zero salt flow, β_eff,flow shall equal static βeff. At increasing salt velocities, β_eff,flow shall decrease monotonically. Direction and approximate magnitude verified against MSRE analysis literature.
•	Pump trip transient: upon sudden reduction in salt flow rate, the decrease in precursor drift shall cause a calculable shift in reactor dynamics. Verified qualitatively against published MSR transient analyses.
•	Shared kinetics consistency: at zero salt velocity, MSR point kinetics results shall be numerically identical to PWR point kinetics results given identical input parameters. This validates the shared kinetics core implementation.
7.4 Benchmarking
Physics loop performance shall be benchmarked using pytest-benchmark before and after Numba optimization. Target: full ODE timestep integration in under 20ms per UI callback cycle on local hardware.
 
8. Milestone & Iterative Development Plan
Pre-Phase: Environment Setup
Duration: Week 1 Deliverables:
•	Poetry environment configured with all dependencies installed and locked
•	GitHub repository initialized with branch protection and README
•	ARCHITECTURE.md created in repository root covering layer separation, engine boundaries, shared kinetics core design decision, and 0DE state vectors. Updated at the end of each phase.
•	Dash boilerplate application running locally with placeholder reactor selector
•	IAEA delayed neutron database downloaded and parsed into project data directory
•	Keepin 6-group parameters loaded and unit-tested
•	Bosch-Hale D-He3 coefficients implemented and verified against published ⟨σv⟩ tables
•	Pydantic base schemas defined for all three reactor parameter sets
•	Render or Railway deployment pipeline connected to GitHub — deploy the boilerplate immediately so deployment is never a last-minute problem
Success Criteria: poetry run pytest passes on data loading tests. Dash app accessible at public URL showing placeholder UI.
 
Phase 1: PWR Twin
Duration: Weeks 2–5 This phase builds the platform, not just the PWR. Every architectural decision made here persists through Phases 2 and 3.
Week 2 — Kinetics Engine:
•	Implement 6-group point kinetics ODE system
•	Implement Xe-135/I-135 kinetics ODE system
•	Implement Doppler and moderator feedback calculators
•	Implement boron worth calculator
•	Implement reactivity balance summation
•	Unit test all components individually
•	Integration test: steady-state stability
Week 3 — Thermal Hydraulics + Full PWR System:
•	Implement lumped fuel and coolant temperature ODEs
•	Couple thermal-hydraulic module to kinetics engine via feedback loop
•	Implement full 11-state PWR ODE system
•	Apply Numba JIT to hot loops
•	Benchmark performance against NFR-02 target
•	Validate Doppler self-regulation and xenon equilibrium
Week 4 — PWR UI:
•	Build PWR control panel (Dash + DBC): rod, boron, flow rate, temperature inputs
•	Build real-time Plotly time-series plots: power, temperatures, reactivity breakdown, Xe/I concentrations
•	Implement accelerated time mode (time multiplier slider)
•	Implement plain-language event log
•	Implement HDF5 save/load for simulation states
Week 5 — PWR Scenarios + Validation:
•	Implement scenario mode with predefined PWR scenarios: rod ejection, boron dilution, xenon peak post-shutdown, load following
•	Complete physics validation checklist (Section 9.3)
•	Polish UI, fix performance issues
•	Deploy Phase 1 to production URL
•	Write Phase 1 retrospective: what architectural decisions need to change before Phase 2
Phase 1 Success Criteria: PWR twin accessible at public URL. All physics validation benchmarks passed. Xenon peak transient demonstrable. Rod ejection scenario shows Doppler self-regulation.
 
Phase 2: Helion FRC Twin
Duration: Weeks 6–9 This phase validates platform modularity. Helion's physics is fundamentally different — if the platform architecture accommodates it cleanly, Phase 3 is low risk.
Week 6 — Helion Physics Engine:
•	Implement plasma energy balance ODE
•	Implement Bosch-Hale reactivity lookup (already unit-tested from pre-phase)
•	Implement adiabatic compression heating calculator
•	Implement Bremsstrahlung radiation loss
•	Implement empirical confinement time scaling with documented uncertainty range
•	Unit and integration tests
Week 7 — Helion Scenarios + Parameter Sweep:
•	Implement pulse simulation (microsecond timescale, no time acceleration needed)
•	Implement 2D ignition boundary sweep across compression ratio × plasma density
•	Implement Q factor calculation
•	Validate energy conservation and temperature scaling
Week 8 — Helion UI:
•	Build Helion control panel: compression ratio, plasma density, temperature, pulse duration inputs
•	Build time-series plots: temperature, fusion power, loss channels
•	Build ignition boundary heatmap (most visually striking output in the platform)
•	Integrate into unified reactor selector with PWR
Week 9 — Integration + Polish:
•	End-to-end testing of PWR + Helion together in unified dashboard
•	Performance validation on deployed instance
•	Deploy Phase 2 to production
Phase 2 Success Criteria: Helion twin accessible at same public URL as PWR. Ignition boundary heatmap renders correctly showing Q > 1 region. Platform selector switches cleanly between PWR and Helion without state contamination.
 
Phase 3: MSR Twin
Duration: Weeks 10–12 This phase demonstrates the shared kinetics infrastructure was worth building correctly. MSR reuses the PWR kinetics core with drift modifications.
Week 10 — MSR Physics Engine:
•	Subclass/extend shared kinetics core with precursor drift terms
•	Implement precursor return delay with exponential decay
•	Implement β_eff,flow calculation
•	Implement unified salt thermal-hydraulic module
•	Implement temperature feedback (single salt coefficient)
•	Unit test drift modification against zero-flow equality with PWR results
Week 11 — MSR Scenarios + UI:
•	Implement MSR scenarios: pump trip, load following, comparison of β_eff,flow at different salt velocities
•	Build MSR control panel and plots
•	Integrate into unified dashboard
Week 12 — Final Integration + Portfolio Polish:
•	Three-reactor platform fully integrated and deployed
•	Complete physics validation for MSR (Section 9.3)
•	Full regression test suite passing across all three reactors
•	README written for GitHub: project purpose, physics summary, live demo link, tech stack
•	Engineering spec document finalized and linked from README
•	Final performance audit on deployed instance
Phase 3 Success Criteria: All three reactor twins accessible from unified dashboard at public URL. MSR β_eff,flow demonstrably decreases with salt velocity. Shared kinetics consistency test passes. Full physics validation checklist complete across all three reactors.


9. Glossary
Term	Definition
βeff	Effective delayed neutron fraction — the fraction of fission neutrons that are emitted with a delay. Makes reactor control possible. Typical PWR value ~0.0065.
β_eff,flow	Effective delayed neutron fraction reduced by precursor loss in a flowing-fuel MSR. Always less than static βeff.
Bosch-Hale parametrization	Published analytical fit for fusion reactivity ⟨σv⟩ as a function of plasma temperature for various fuel combinations including D-He3.
Doppler effect	Broadening of U-238 neutron absorption resonances with increasing fuel temperature, producing an inherent negative reactivity feedback. Also called Doppler broadening.
FRC	Field-Reversed Configuration — a compact plasma confinement geometry used by Helion Energy in which the magnetic field is reversed inside the plasma, creating a self-contained configuration.
HDF5	Hierarchical Data Format version 5 — a binary file format for storing large scientific datasets. Used here for simulation run storage.
Iodine pit	The period following reactor shutdown during which xenon buildup (from iodine decay) may prevent restart due to insufficient positive reactivity to overcome xenon worth.
Keepin parameters	Standard 6-group delayed neutron precursor data published by G.R. Keepin (1965). Tabulates decay constants λᵢ and group fractions βᵢ for major fissile isotopes.
Λ (prompt neutron lifetime)	Mean time between neutron birth and absorption causing fission. Typical PWR value ~10⁻⁵ seconds.
MSR	Molten Salt Reactor — a reactor design in which fissile material is dissolved in a molten fluoride or chloride salt serving as both fuel and coolant.
ODE	Ordinary Differential Equation — the mathematical form of all dynamic models in this platform.
Point kinetics	A zero-dimensional model of neutron population dynamics that treats the entire reactor core as a single point with uniform neutron flux.
Precursor drift	The transport of delayed neutron precursors out of the active core region by flowing salt in an MSR, reducing their contribution to reactor control.
PWR	Pressurized Water Reactor — a light water reactor design using pressurized water as both coolant and moderator. Most common commercial reactor type globally.
Q factor	Fusion energy gain factor — ratio of fusion energy produced to energy input required to heat the plasma. Q > 1 represents net energy gain.
Radau solver	An implicit Runge-Kutta method for stiff ODE systems. Used here via SciPy solve_ivp for all reactor physics integration.
Reactivity (ρ)	A measure of the departure of a reactor from criticality. ρ = 0 at steady state, positive during power increase, negative during shutdown.
Reduced-order model	A physics model that captures essential system behavior using simplified equations rather than full numerical simulation.
⟨σv⟩	Fusion reactivity — the temperature-dependent rate coefficient for fusion reactions, equal to the product of cross section and relative velocity averaged over the thermal velocity distribution.
Stiff ODE system	A system of ODEs with widely varying timescales that requires implicit numerical methods to solve efficiently. All three reactor models are stiff.
τ_E	Energy confinement time — the characteristic time for a plasma to lose its thermal energy. Central parameter in fusion power balance.
Xenon peak	The transient rise in Xe-135 concentration following reactor shutdown, caused by continued iodine decay in the absence of neutron flux burnout.
Xenon worth	The negative reactivity contribution of Xe-135 at a given concentration. Equilibrium xenon worth at full power in a commercial PWR is approximately -2500 to -3000 pcm.


10. References & Resources
Nuclear Data
•	IAEA Nuclear Data Section. Reference Database for Beta-Delayed Neutron Emission. https://www-nds.iaea.org/beta-delayed-neutron/
•	IAEA Power Reactor Information System (PRIS). https://pris.iaea.org/
•	Keepin, G.R. (1965). Physics of Nuclear Kinetics. Addison-Wesley. [Source of standard 6-group delayed neutron parameters]
•	Bosch, H.S. and Hale, G.M. (1992). "Improved formulas for fusion cross-sections and thermal reactivities." Nuclear Fusion 32(4), 611. [Source of D-He3 ⟨σv⟩ parametrization]
Textbooks
•	Lamarsh, J.R. and Baratta, A.J. Introduction to Nuclear Engineering. 3rd ed. Prentice Hall. [PWR and MSR kinetics reference]
•	Stacey, W.M. Nuclear Reactor Physics. 2nd ed. Wiley-VCH. [Point kinetics and feedback coefficient reference]
•	Glasstone, S. and Sesonske, A. Nuclear Reactor Engineering. 4th ed. [Thermal-hydraulic modeling reference]
MSR-Specific
•	Haubenreich, P.N. and Engel, J.R. (1970). "Experience with the Molten-Salt Reactor Experiment." Nuclear Applications and Technology 8(2). [MSRE operational data and precursor drift analysis]
•	Kerlin, T.W. et al. (1971). "Theoretical and Experimental Dynamic Analysis of the Molten Salt Reactor Experiment." Nuclear Applications and Technology 8(2).
FRC and Fusion
•	Tuszewski, M. (1988). "Field reversed configurations." Nuclear Fusion 28(11), 2033. [FRC physics reference]
•	Binderbauer, M.W. et al. (2015). "A high performance field-reversed configuration." Physics of Plasmas 22, 056110. [Modern FRC experimental results]
Software References
•	PyRK: Python for Reactor Kinetics. https://github.com/pyrk/pyrk [Reference implementation, not adopted]
•	SciPy documentation: scipy.integrate.solve_ivp. https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html
•	Dash documentation. https://dash.plotly.com/
•	Plotly Python documentation. https://plotly.com/python/



