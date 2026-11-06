# ARC V3A FLiBe CFD — assumptions and parameter log

This is a running log. The parameter tables give the **current** value of
everything the CFD case uses and where it comes from. The decision log at the
bottom records every change, newest last. The code's single source of truth
is the `CFG` block in [`make_case.py`](make_case.py); keep this file in step
with it.

**Source tags**

| tag | meaning |
|---|---|
| **PUB [ref]** | published value (reference below) |
| **USER** | given by the user in the task specification; no reference document was supplied |
| **OPENMC** | computed from the OpenMC run `ARC_v3A/run_v3a` |
| **DERIVED** | computed by `make_case.py` from other entries |
| **ASSUMPTION (user)** | modelling choice set by the user, not a published value |
| **ASSUMPTION (mine)** | my modelling choice; the user accepted it or confirmed it where noted |

**References**
- [H26] Hillesheim et al., J. Plasma Phys. 92 (2026) E69 (ARC V3A physics basis).
- [V3A] ARC V3A digital assets, doi:10.5281/zenodo.19498373.
- [RH17] Romatoski & Hu, Ann. Nucl. Energy 109 (2017), FLiBe properties.
- [K18] Kuang et al. 2018, ARC V0 FLiBe cooling scheme (Fig. 11).
- [C25] Carter et al., IAEA, Dec 2025 (CFS FLiBe CFD).

I have not seen any of these documents. Where a value is tagged PUB, the user
or the existing repo scripts quoted it from that reference.

---

## 1. Inputs and environment

| parameter | value | source / note |
|---|---|---|
| Geometry | `arc_v3a_paramak_model.build_profiles(ARCParameters(), magnets=False)` polygons, cm | USER (single source of geometry); the polygons come from [H26]/[V3A] per that script's header |
| OpenMC run folder | `ARC_v3A/run_v3a` | ASSUMPTION (mine): `run_test/` holds only inputs and lost-particle files, no post-processed output |
| OpenMC checks | TBR 1.208, total nuclear heating 1020.3 MW (1019.8 MW without magnets) | OPENMC. Accepted against "about 1.0–1.2 / about 1000 MW" (USER); the window coded is TBR 0.95–1.25 and heating 900–1100 MW (ASSUMPTION, mine) |
| OpenMC mesh vs cell tallies | heating identical; isotope mesh sums 0.08–0.24 % below the cell tallies | OPENMC; cause unknown, not rescaled |
| Python env | repo `.venv` (uv, Python 3.14, prompt "hot-flibe-summer") | ASSUMPTION (mine), accepted by user: no conda env of that name exists |
| OpenFOAM | v2606 (OpenFOAM.com), `/usr/lib/openfoam/openfoam2606/etc/bashrc` | USER. **Not installed on this machine.** Keywords were verified against the v2606 source (gitlab.com/openfoam/core/openfoam, tag OpenFOAM-v2606, commit 481094fd) |

## 2. Machine and geometry (checks)

| parameter | value | source / note |
|---|---|---|
| R0, a, κ, δ, TF coils | 4.62 m, 1.18 m, 1.8, 0.65, 18 | PUB [H26], quoted by user |
| Vessel build (plasma outward) | 5 mm W / 1 cm V / 4.5 cm FLiBe gap / 3 cm V; tank wall 2 cm | PUB [H26]/[V3A], quoted by user |
| FLiBe volumes (360°) | gap 27.83 m³, bulk 431.99 m³ | DERIVED (polygons, Pappus); user check values 27.8 / 432 |
| Midplane FLiBe | inboard bulk 3.12–3.33 m, outboard 5.91–7.20 m | DERIVED; matches user check values |
| Up-down symmetry | geometry symmetric; flow is not, so the full poloidal section is modelled | USER / DERIVED |

## 3. Operating point and heat budget

| parameter | value | source / note |
|---|---|---|
| Fusion power, neutron rate | 1133 MW, 4.0e20 n/s | PUB [H26], quoted by user |
| Nuclear heating, FLiBe gap / bulk | 309.44 / 499.15 MW | OPENMC (mesh fields, split by region; equal to cell tallies to < 0.001 %) |
| Nuclear heating, W / inner V / outer V / WC / tank | 68.75 / 42.00 / 88.40 / 11.76 / 0.30 MW | OPENMC |
| Magnet heating | excluded (cryogenics) | USER |
| Alpha heating | 227 MW | USER |
| RF/ohmic | 22 MW | USER gave "~22 MW"; using exactly 22 is ASSUMPTION (mine) |
| Psep | 120 MW to the divertors, 60 MW to each divertor circuit, uniform over that circuit's leg + foot wall | USER (V0 approach [K18]) |
| Core radiation | 227 + 22 − 120 = 129 MW, uniform on the main-chamber first wall, split inboard/outboard by first-wall area | USER. Using the plasma-facing (limiter) surface area for the split is ASSUMPTION (mine) |
| Total heat into FLiBe | 1268.80 MW | DERIVED |
| Inlet temperature | 800 K (FLiBe freezes at 732 K) | USER / [K18] |
| Total temperature rise | 100 K (outlet ≈ 900 K) | USER |
| Per-circuit temperature rise | 75 K | USER (Kuang 2018 channel limit) |

## 4. FLiBe properties

| parameter | value | source / note |
|---|---|---|
| Density | 2413 − 0.488 T kg/m³ (valid 800–1080 K) | PUB [RH17], quoted by user |
| Viscosity | 1.16e-4 exp(3755/T) Pa s (valid 873–1073 K, ±20 %) | PUB [RH17], quoted by user. **Extrapolated** to the 800 K inlet |
| Viscosity in OpenFOAM | degree-4 polynomial fitted over 780–1050 K: max error 0.24 %, rms 0.054 %; positive over 700–1200 K | DERIVED. The rule "lowest degree with max error ≤ 1 %" is ASSUMPTION (mine). Degree 5 gives 0.04 % |
| cp, k | 2386 J/kg/K, 1.1 W/m/K | PUB [RH17], quoted by user |
| Pr at 900 K | 16.3 | DERIVED (user: about 16) |
| molWeight | 33 | USER (0.67 LiF – 0.33 BeF2; no effect) |
| Thermo package | heRhoThermo / pureMixture / polynomial / hPolynomial / icoPolynomial / sensibleEnthalpy | USER; the combination was confirmed as instantiated in v2606 `rhoThermos.C` |

## 5. Cooling layout (V0 scheme on V3A)

| parameter | value | source / note |
|---|---|---|
| Scheme | ARC V0 [K18] Fig. 11 applied to V3A: 4 vessel circuits + bypass + 1 hot outlet | ASSUMPTION (user) |
| Junctions | (3.68, ±2.10) m, (4.18, ±2.35) m | USER; each falls exactly on a LIMITER_M vertex (0 cm snap) |
| Baffles | zero thickness, straight across the gap along the corner bisector (joining the matching corners of the gap's two walls) | ASSUMPTION (mine), confirmed by user 2026-11-05. At the lower inboard junction the baffle continues through the outer V wall, between the two openings |
| Circuit paths | inboard: top → bottom of inboard wall; upper divertor: inboard base → foot → outboard base; outboard: top → bottom through the midplane; lower divertor: outboard base → foot → inboard base | USER [K18] |
| Inlet patches | on the gap's tank-side face at each circuit's start, beginning at the baffle | USER (position); "beginning at the baffle" is ASSUMPTION (mine) |
| Inlet velocity | 2 m/s (slot area = 2πR·w) | USER (V0 pipes about 2 m/s) |
| Openings | outer V wall removed from the baffle back along the circuit's last edge; length sized for 2 m/s on the bulk-side face | USER (opening); the bulk-side sizing rule and the 2 m/s value are ASSUMPTION (mine), confirmed 2026-11-05. Reason: the wall is wedge-shaped at the junctions, so the bulk side is 1.7 cm (inboard) or 2.6 cm (outboard) shorter than the gap side |
| Bypass flow | m_total − Σ circuits = 1331 kg/s ("balance") | USER. V0's fixed 0.19 m³/s is a config option, not used |
| Bypass position | tank top, starting 4 cm outboard of the WC shield tip (4.11, 4.37) m along the tank-top edge, R = 4.154 m; discharges about 18° from vertical, downward | ASSUMPTION (mine; the user left the exact spot to me), confirmed 2026-11-05 |
| Outlet | tank top, R 4.90–5.20 m, across the apex (5.05, 4.68) m | USER (position); the R range is ASSUMPTION (mine, carried over from the first spec) |
| Supply/return pipes in the tank | not modelled | USER (2D limitation) |

**Derived circuit values** (full machine):

| circuit | Q MW | ṁ kg/s | gap m³ | transit s | inlet cm | opening cm |
|---|---|---|---|---|---|---|
| Inboard main | 156.6 | 875 | 4.13 | 9.47 | 0.95 | 0.98 |
| Upper divertor | 91.8 | 513 | 7.47 | 29.2 | 0.56 | 0.48 |
| Outboard main | 373.5 | 2087 | 8.75 | 8.40 | 1.93 | 1.94 |
| Lower divertor | 91.5 | 511 | 7.47 | 29.3 | 0.47 | 0.57 |
| Bypass | — | 1331 | — | — | 1.26 | — |

Total ṁ is 5318 kg/s. The design mixed outlet is 900.0 K and the design
circuit exit is 875 K. The upper and lower divertor values differ by 0.3 %;
that is OpenMC statistical noise.

## 6. Boundary conditions

| parameter | value | source / note |
|---|---|---|
| Wall heat flux | v1: each wall total spread uniformly over its patch | USER |
| Gap, plasma side of circuit c (`fw_c`) | W + inner V heating in its segment + its surface heat | USER (per-circuit split by integrating the OpenMC mesh over each segment) |
| Gap, tank side of circuit c (`vvGap_c`) | ½ of the outer V wall heating in its segment | USER |
| Bulk side of the outer V wall (`vvBulk`) | ½ of the total outer V wall heating (one patch) | USER (split); one patch is ASSUMPTION (mine) |
| WC shield face (`shield`) / tank inner face (`tank`) | 11.76 / 0.30 MW | USER / OPENMC |
| Baffles, opening side walls | adiabatic, no-slip | ASSUMPTION (mine) |
| Outer V wall heating where an opening removes it | still counted in the outer V wall totals (OpenMC had V there) | ASSUMPTION (mine) |
| Heat-flux BC | `externalWallHeatFluxTemperature`, `mode power`, `Q` in W per patch (wedge share), `kappaMethod fluidThermo` | ASSUMPTION (mine), so the imposed power is exact; verified in v2606 source |
| Inlets | `flowRateInletVelocity` with `massFlowRate` (wedge share), `rhoInlet` 2022.6 kg/m³; T = 800 K | USER (mass flow, 800 K) |
| Inlet turbulence | 5 % intensity; mixing length 0.07·D_h with D_h = 2 × slot width | USER (5 %); the mixing-length rule is ASSUMPTION (mine) |
| Outlet | p_rgh `prghPressure`, p = 1e5 Pa; U `pressureInletOutletVelocity`; T, k, ε and scalars `inletOutlet` (backflow T 900 K, scalars 0) | USER (fixed p, inletOutlet); the values are ASSUMPTION (mine) |
| Front/back | wedge | USER |
| Gravity | (0 0 −9.81) m/s² | USER |

## 7. Turbulence and wall treatment

| parameter | value | source / note |
|---|---|---|
| Model | realizableKE | USER |
| ν_t wall function | `nutUSpaldingWallFunction` | ASSUMPTION (mine), accepted: gap y+ is about 20–35 with 7.8 cells across, below the 30–300 target |
| Thermal wall function | `compressible::alphatJayatillekeWallFunction`, Prt 0.85 | USER |
| k, ε walls | `kqRWallFunction`, `epsilonWallFunction` | ASSUMPTION (mine; standard pair) |
| Buoyancy production of k/ε | none (realizableKE has none; `buoyancyTurbSource` exists in v2606 but is off) | limitation; noted |
| MHD | none | USER (out of scope; [C25] ignored it too) |
| Initial fields | U 0, T 850 K, p = p_rgh = 1e5 Pa, k 1e-3 m²/s², ε 5e-5 m²/s³ | ASSUMPTION (mine) |

## 8. Mesh

| parameter | value | source / note |
|---|---|---|
| Route | Gmsh 2D (geo kernel, quads) → one-cell wedge extruded in Python → Gmsh 2.2 → `gmshToFoam` → `changeDictionary` → `checkMesh` | ASSUMPTION (mine). Justification: zero-thickness baffles need duplicated nodes. Extruding in Python stops Gmsh from merging the coincident baffle curves, and the mesh can be checked in Python before OpenFOAM |
| Wedge | 2° total, symmetric about y = 0 | USER (< 5°); the 2° value is ASSUMPTION (mine) |
| Gap cells | 0.6 cm, giving 7.8 cells across 4.5 cm | USER (≥ 6–8 across); value mine |
| Bulk cells | 1.5 cm at walls, growing to 4 cm over 25 cm | USER (3–5 cm bulk); wall size and growth are ASSUMPTION (mine) |
| Slots/openings | ≥ 5 cells across, smallest 0.08 cm, refined within w + 3 cm | ASSUMPTION (mine) |
| Fine mesh | 55,459 cells (55,367 hex, 92 prism); quality minSICN min 0.33 | DERIVED |
| Coarse (smoke) mesh | 1.1 cm gap, 3 / 8 cm bulk, 3 cells per slot: 15,926 cells | ASSUMPTION (mine) |
| Wedge volume factor | CFD volume × 360/θ = 0.999797 × polygon volume (planar wedge, sin θ/θ) | DERIVED. Mapping, BCs and post all use 360/θ, so totals are exact |

## 9. Activation products and age

| parameter | value | source / note |
|---|---|---|
| Isotopes | N-16, F-20, O-19, F-18 | USER |
| Half-lives | N-16 7.13 s, F-20 11.163 s, O-19 26.88 s, F-18 6586.2 s | PUB via `openmc.data.half_life()` (USER: use OpenMC's decay data) |
| Equation | div(φC) − ∇·((μt/Sct)∇C) = S − λρC | USER |
| Transported quantity | specific concentration C, atoms/kg (φ is a mass flux) | USER offered either; chosen by me |
| Sct; molecular diffusion | 0.85; neglected (`alphaD 0`, `alphaDt` 1/0.85) | USER |
| Decay | implicit sink via `fvm::Sp(−λρ, C)` in a coded fvOption | USER (implicit); implementation mine |
| Solve | once on the converged flow (frozen): `buoyantSimpleFoam -postProcess -dict system/scalarsDict -time <latest>`; up to 100 outer correctors, outer tolerance 1e-9 | ASSUMPTION (mine) |
| Inlet concentration | 0 for all isotopes (once-through) | ASSUMPTION (mine), accepted. F-18 would really recirculate |
| Mean age | OpenFOAM `age` FO, diffusion on: μ_eff = μ + μt (Sc = 1), not μt/Sct | USER (age FO). **Flagged 2026-11-05: v2606's `age` cannot use μt/Sct; awaiting your decision** |
| Age distribution | flux-weighted histogram of local mean age over the outlet + mass-weighted histogram over the tank | ASSUMPTION (mine), accepted. A true RTD needs a transient tracer |

## 10. Source mapping

| parameter | value | source / note |
|---|---|---|
| Algorithm | bin total = density × ring volume, spread over FLiBe cells with centres in the bin (by volume); a bin with no centre goes to the nearest cell | USER |
| Refinement | each bin is classified gap or bulk by polygon overlap (a 1 cm bin cannot touch both through the 3 cm wall); the nearest-cell fallback stays within that region. The gap pool includes the openings | ASSUMPTION (mine) |
| Cell centres/volumes | OpenFOAM `writeCellCentres` / `writeCellVolumes`; zones from `cellZones` | USER |
| Check | gap, bulk and total within 1 % of the `flibe_*` rows of `cell_totals.csv`, otherwise stop | USER |
| Offline test (Python-computed centres, shuffled order) | exact conservation (round-off); −0.28 % to 0.00 % vs cell tallies; 34–48 % of each total via the nearest-cell fallback | DERIVED. The fallback share is expected: 4 cm bulk cells cover about 16 bins each |

## 11. Numerics and run control

| parameter | value | source / note |
|---|---|---|
| Solver | buoyantSimpleFoam, steady, p_rgh | USER |
| Schemes | U `bounded Gauss linearUpwind grad(U)`; h, K, scalars, age `bounded Gauss limitedLinear 1`; k, ε `bounded Gauss upwind`; laplacian `Gauss linear limited corrected 0.33` | ASSUMPTION (mine) |
| Relaxation | p_rgh 0.7, U 0.3, h 0.3, k/ε 0.5, scalars and age 1 | ASSUMPTION (mine); U, h, p_rgh as in the v2606 `buoyantCavity` tutorial |
| Residual control | p_rgh 1e-5, U 1e-6, h 1e-7, k/ε 1e-5 | ASSUMPTION (mine) |
| Iterations | fine 5000 (write every 500), coarse 300 (every 100) | ASSUMPTION (mine) |
| Processes | fine 8 (scotch), coarse 1 | ASSUMPTION (mine); the machine has 32 threads |
| Coded fvOptions | need `wmake` and headers (`openfoam2606-dev` or `-default`) | note |

---

## Decision log

| date | entry |
|---|---|
| 2026-11-05 | Read the scripts and run folders. OpenMC run `run_v3a` accepted (TBR 1.208, 1020 MW). Found that OpenFOAM is not installed and there is no conda env; the user accepted my defaults (use `.venv`, 249 MW surface heat, Spalding wall function, age histograms, once-through scalars, root `.gitignore`, no `git rm --cached`) |
| 2026-11-05 | User replaced the flow layout with the V0 scheme [K18] (4 circuits + bypass + top outlet) and the surface-heat split (Psep 120 MW to the divertors, rest uniform on the main chamber) |
| 2026-11-05 | Opening length redefined on the bulk-side face, because the wedge-shaped wall corners left the small circuits with no bulk-side opening at 2 m/s |
| 2026-11-05 | User confirmed the junctions, baffles, inlets, openings, bypass and outlet from `circuits_v3a.png` |
| 2026-11-05 | Verified the v2606 keywords in the source: `externalWallHeatFluxTemperature mode power`, `scalarTransport` (D = alphaD·μ + alphaDt·μt; field read from the current time; outer tolerance default 1 → set 1e-9), `age` (diffusivity μ_eff, Sc = 1), `scalarCodedSource` with `codeAddSupRho`, the polynomial thermo combination, `flowRateInletVelocity massFlowRate`, `prghPressure p`, gmshToFoam physical names → patches and cellZones, and that the `-dict` file merges into controlDict |
| 2026-11-05 | Wrote the mesh, case, mapping and post-processing. Fine mesh 55,459 cells. Mapping and post tested offline on synthetic cases; nothing has been run in OpenFOAM yet |
| 2026-11-05 | Open item: `age` diffusivity (μ + μt) differs from the isotopes' μt/Sct. Options: keep it; turn diffusion off; or compute age with `scalarTransport` (source ρ, D = μt/Sct) for consistency |
