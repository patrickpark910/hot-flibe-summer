# ARC V3A FLiBe blanket CFD (OpenFOAM v2606)

2D axisymmetric, steady, buoyant RANS model of the FLiBe in the ARC V3A
blanket tank and the 4.5 cm vessel cooling gap. It is one-way coupled to the
OpenMC run in `ARC_v3A/run_v3a`, which supplies the nuclear heating and the
F-19 activation-product sources.

The model transports four short-lived activation products (N-16, F-20, O-19,
F-18) with radioactive decay, plus the mean age of the fluid. The main
question: what fraction of each isotope leaves the tank before it decays, and
what activity reaches the outlet.

The method follows CFS (Carter et al., IAEA, Dec 2025): Fluent 2D
axisymmetric, k-epsilon, temperature-dependent FLiBe, buoyancy, and a decaying
user-defined scalar. Their slides are qualitative and use the V2A geometry, so
comparisons are about flow patterns only.

Every parameter, with its source (published, user specification, OpenMC,
derived or ASSUMPTION), is in [`assumptions_log.md`](assumptions_log.md). All
assumptions live in the single `CFG` block at the top of
[`make_case.py`](make_case.py).

## Files

| file | what |
|---|---|
| `make_case.py` | geometry, cooling circuits, heat loads and flows, Gmsh mesh, OpenFOAM case, source mapping |
| `foam_io.py` | small ASCII OpenFOAM reader/writer used by the two scripts |
| `post_cfd.py` | balances, isotope results, age, y+, PNG maps |
| `circuits_v3a.png` | the four vessel circuits, bypass and outlet (written by `make_case.py`) |
| `case_v3a/` | the OpenFOAM case (generated) |
| `assumptions_log.md` | running log of every parameter and decision |

## Requirements

- **Python steps:** the repo's `.venv` (uv, Python 3.14; it has gmsh, shapely,
  numpy, scipy, matplotlib, cadquery, paramak and openmc). There is no conda env
  named `hot-flibe-summer`.
- **OpenFOAM steps:** OpenFOAM.com v2606, sourced from
  `/usr/lib/openfoam/openfoam2606/etc/bashrc`. `Allrun` sources it itself and
  refuses to run if a conda env or Python venv is active.
  - The heat source and the isotope decay are coded fvOptions, so they are
    compiled at run time. That needs `wmake` and the OpenFOAM headers
    (`openfoam2606-dev`, or the `openfoam2606-default` meta-package).
  - Installed here as apt `openfoam2606-default` (2606.0~rc2). The gmsh wheel
    in `.venv` also needs the system `libGLU.so.1` (`sudo apt install libglu1-mesa`).

## How to run

Keep two shells: one with the Python env, one with OpenFOAM (no conda/venv).

```bash
# 1. Python env: geometry, flows, mesh, case files (smoke-test size first)
.venv/bin/python cfd/make_case.py build --coarse

# 2. OpenFOAM shell: convert and check the mesh, write cell centres/volumes
cd cfd/case_v3a && ./Allrun mesh

# 3. Python env: map OpenMC heating and isotope production onto the FLiBe cells
.venv/bin/python cfd/make_case.py map

# 4. OpenFOAM shell: flow (buoyantSimpleFoam; mean age at its end), then frozen-flow isotopes
./Allrun solve

# 5. Python env: report and maps (case_v3a/postProcessing/post_cfd/)
.venv/bin/python cfd/post_cfd.py
```

**Smoke test vs. full run:**
- `--coarse` gives about 16k cells and 300 iterations, run serially.
- Without `--coarse`: about 55k cells (7.8 cells across the gap), 5000
  iterations, 8 MPI processes. Run the smoke test first. Ask before any run
  expected to take more than 30 minutes.
- `./Allrun flow` and `./Allrun scalars` can also be run separately.
- While they run, a status line (iteration, residuals, cell T max, outlet T, time
  left) prints every 10 s; `PROGRESS_EVERY=30 ./Allrun solve` changes the interval.
  The full solver output is in `log.buoyantSimpleFoam` (`.scalars` for the scalars).
- `./Allclean` removes OpenFOAM outputs and keeps `v3a.msh`, the 0/ base
  fields, `constant/` and `system/`.

`python cfd/make_case.py circuits` only redoes the circuit table and plot.

## What the model does

**Domain.** FLiBe only, taken from the shapely polygons of
`arc_v3a_paramak_model.build_profiles()`: the vessel gap plus the bulk tank.
- The gap is split into four independent circuits by zero-thickness baffles at
  the junctions (3.68, ±2.10) m and (4.18, ±2.35) m. This is the ARC V0 scheme
  (Kuang et al. 2018, Fig. 11).
- Each circuit has a mass-flow inlet (800 K) on its tank-side face at its start.
  It discharges through an opening in the outer V wall at its end.
- A bypass enters downward from the tank top in the upper inboard region.
- One hot outlet sits at the top of the tank.

**Flows.**
- Each circuit carries ṁ_i = Q_i / (cp · 75 K). Q_i is the OpenMC heating of
  the W, the inner V, the gap FLiBe and half the outer V within its segment,
  plus its share of the plasma surface heat.
- The bypass takes the rest of the total ṁ = Q / (cp · 100 K).
- `make_case.py` prints every circuit's flow, heat, exit temperature and
  transit time.

**Mesh.**
- Gmsh 2D quads are extruded in Python into a one-cell 2° wedge, symmetric
  about y = 0, and written as Gmsh 2.2 for `gmshToFoam`.
- The extrusion is done in Python, not by Gmsh, so that the duplicated baffle
  curves stay separate.
- The mesh is checked in Python before OpenFOAM sees it: cell orientation,
  every edge internal or on exactly one patch, and zone volumes.

**Energy.**
- `buoyantSimpleFoam` with `heRhoThermo` and FLiBe properties from Romatoski &
  Hu 2017: polynomial ρ and cp, and μ(T) as a `tabulatedTransport` table of the
  Arrhenius law (700 K–1e5 K, interpolation error ≤ 0.1 % below 1400 K). A
  polynomial μ diverges outside its fit range, and the wall functions see
  wall-face temperatures well above the cell temperatures.
- Volumetric FLiBe heating is a nonuniform field, `0/qFlibe`, added to h
  through a coded fvOption.
- Each wall gets its OpenMC total plus surface heat through
  `externalWallHeatFluxTemperature` in `mode power`, so the imposed watts are
  exact. Psep = 120 MW goes to the divertor legs and feet (60 MW each); the
  remaining 129 MW goes uniformly on the main-chamber wall.

**Isotopes.**
- Solved once on the converged, frozen flow:
  `buoyantSimpleFoam -postProcess -dict system/scalarsDict`.
- `scalarTransport` solves div(φC) − ∇·((μt/Sct)∇C) = S − λρC, where C is in
  atoms/kg, because φ is a mass flux.
- S is read from `0/S_<iso>` and added explicitly; decay is an implicit sink.
- Half-lives come from `openmc.data.half_life()`.

**Mean age.** OpenFOAM's `age` function object, with diffusion on. It runs once at the
end of the flow run (`executeControl onEnd` in `controlDict`), not in the scalars step:
in `buoyantSimpleFoam -postProcess` v2606 builds it before `phi` exists and silently
drops it. It is saved only if the last iteration is a write time, so keep `endTime` a
multiple of `writeInterval` (`./Allrun flow` warns otherwise).

**Source mapping.**
- Each OpenMC bin's FLiBe total is spread over the FLiBe cells whose centres lie
  in the bin, weighted by volume.
- A bin with no cell centre gives its total to the nearest cell of the same
  FLiBe region (gap or bulk).
- Totals are checked against the `flibe_*` rows of `cell_totals.csv`; the step
  stops if any is off by more than 1 %.

## Checks built in

- **OpenMC run:** TBR and total heating must be in range, otherwise the scripts stop.
- **Region heating:** the OpenMC mesh heating split by region must equal the
  per-volume tallies (it does, to < 0.001 %).
- **Mesh:** orientation, topology, and zone volumes vs. polygon volumes. The
  ratio is 0.99980, the planar-wedge factor sin θ/θ. All totals use the same
  360/θ factor, so they stay exact.
- **`checkMesh`:** `Allrun mesh` stops if it does not report "Mesh OK".
- **Mapping:** within 1 % of the cell tallies. The offline test gives −0.28 % to
  0.00 %, which is the known OpenMC mesh-vs-cell difference for the isotopes.
- **Post-processing:**
  - mass in = out;
  - m cp ΔT vs. imposed heat (1–2 %);
  - production = decay + outflow per isotope (1 %);
  - flow-weighted outlet age vs. M/ṁ;
  - y+ per wall patch.

## Known limitations

- **2D axisymmetric.** In reality the supply and return pipes cross the tank
  (V0: Inconel 718 pipes, about 1.5 m² in total). They aren't modelled: in 2D
  they would become solid rings blocking the tank. Every inlet is a boundary
  patch that stands for a pipe arriving from out of plane.
- **Gap channels.** The gap is an open 4.5 cm annulus; the real design has
  discrete channels (V0 divertors used 12 mm swirl tubes). Gap velocities and
  wall heat transfer are unrepresentative, but transit time (volume / flow) is
  preserved, and that is what controls in-vessel N-16 decay.
- **No MHD.** A 5–11 T field would damp turbulence and change mixing and heat
  transfer. Carter et al. also ignored it.
- **No conjugate heat transfer (v1).** Wall heat goes straight into the FLiBe
  as a uniform flux per wall. Wall temperatures are FLiBe-side values. Baffles
  and the side walls of the openings are adiabatic.
- **No buoyancy production of turbulence.** `realizableKE` in OpenFOAM has no
  buoyancy production term for k and ε (`buoyancyTurbSource` exists in v2606 but is
  not used).
- **Age diffusivity.** The `age` function object's diffusion uses
  μ_eff = μ + μt (Sc = Sct = 1). The isotopes use μt/0.85.
- **Viscosity extrapolation.** The viscosity correlation is valid from 873 K;
  the 800 K inlet is extrapolated (±20 % stated uncertainty).
- **Once-through scalars.** Inlet concentrations are zero. F-18 (110 min)
  would really recirculate through the external loop.
- **Wall-function y+.** With 7.8 cells across the gap, y+ there is about 20–35.
  `nutUSpaldingWallFunction` is used because it is valid in the buffer layer.
- **Mean age vs. RTD.** The age reported is the local mean age. A true
  residence-time distribution at the outlet needs a transient tracer.
