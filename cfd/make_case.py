#!/usr/bin/env python3
"""
ARC V3A FLiBe blanket: OpenFOAM case builder (2D axisymmetric, steady, buoyant RANS)
====================================================================================

One-way coupled to the OpenMC run in ARC_v3A/run_v3a (heating and F-19
activation-product sources). Geometry comes only from the shapely R-Z polygons
of ARC_v3A/arc_v3a_paramak_model.py (build_profiles), which is not modified.

Cooling layout: the ARC V0 scheme (Kuang et al. 2018, Fig. 11) applied to V3A
(ASSUMPTION). The 4.5 cm vessel gap is split by zero-thickness baffles into four
independent circuits (inboard main chamber, upper divertor, outboard main
chamber, lower divertor). Each circuit has a mass-flow inlet on the gap's
tank-side face at its start and discharges into the bulk tank through an
opening in the outer V wall at its end. A bypass stream enters the upper
inboard bulk; one hot outlet sits at the top of the tank.

Subcommands (run in the "hot-flibe-summer" Python environment, not OpenFOAM's)
-----------
    python cfd/make_case.py circuits
        Builds the circuits, integrates the OpenMC heating over each gap
        segment, sizes every flow, slot and opening, prints the table and
        writes cfd/circuits_v3a.png. Needs no OpenFOAM.

    python cfd/make_case.py build [--coarse]
        Everything in 'circuits', plus: 2D Gmsh mesh of the FLiBe domain,
        extruded here into a one-cell wedge and written as Gmsh 2.2 (v3a.msh),
        checked in Python; viscosity table; and the whole OpenFOAM case
        cfd/case_v3a/ (0/, constant/, system/, Allrun, Allclean,
        case_summary.json). --coarse makes the smoke-test mesh and run length.

    (OpenFOAM shell)  cd cfd/case_v3a && ./Allrun mesh

    python cfd/make_case.py map
        Maps the OpenMC heating and isotope sources conservatively onto the
        FLiBe cells (needs 0/C and 0/V from './Allrun mesh') and writes
        0/qFlibe and 0/S_<isotope>. Stops if any total is off by > 1 %.

    (OpenFOAM shell)  ./Allrun solve       # flow (+ mean age at its end), then frozen-flow isotopes
    python cfd/post_cfd.py

Every modelling choice that is not a published value is marked ASSUMPTION
and lives in the CONFIG block below. cfd/assumptions_log.md lists every
parameter with its source.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import shapely
from shapely.geometry import LineString, Polygon
from shapely.geometry.polygon import orient

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO / "ARC_v3A"))
sys.path.insert(0, str(HERE))
import foam_io  # noqa: E402

# =============================================================================
# CONFIG: every input and modelling choice. Lengths in metres unless noted.
# =============================================================================
CFG = dict(
    # ---- inputs -------------------------------------------------------------
    # ASSUMPTION: run_v3a is the post-processed run (run_test/ holds no outputs)
    openmc_run=REPO / "ARC_v3A" / "run_v3a",
    case_dir=HERE / "case_v3a",
    circuits_png=HERE / "circuits_v3a.png",
    # ASSUMPTION: acceptance window for "TBR about 1.0-1.2, heating about 1000 MW"
    openmc_tbr_range=(0.95, 1.25),
    openmc_heating_MW_range=(900.0, 1100.0),

    # ---- operating point (flattop, steady) ----------------------------------
    P_alpha_MW=227.0,          # alpha heating, all deposited on the vessel
    P_aux_MW=22.0,             # ASSUMPTION: "~22 MW RF/ohmic"
    P_sep_MW=120.0,            # to the divertors, split equally (60 MW each)
    # remaining surface heat (P_alpha + P_aux - P_sep) is core radiation, uniform
    # on the main-chamber first wall, split inboard/outboard by first-wall area
    T_in=800.0,                # K, every inlet
    dT_total=100.0,            # K, sets total flow -> outlet ~900 K
    dT_circuit=75.0,           # K, per vessel circuit (Kuang 2018 channel limit)

    # ---- FLiBe properties, Romatoski & Hu 2017 (T in K) ---------------------
    cp=2386.0,                 # J/kg/K
    kappa=1.1,                 # W/m/K
    rho_coeffs=(2413.0, -0.488),            # kg/m3, valid 800-1080 K
    mu_arrhenius=(1.16e-4, 3755.0),         # Pa s = A exp(B/T), valid 873-1073 K
    # OpenFOAM gets mu(T) as a tabulatedTransport table of the Arrhenius law itself:
    # dT spacing over [T0, T1], then geometric spacing to T_top. The law falls
    # monotonically to A > 0, so it is safe at any T. T_top must exceed any face T:
    # v2606 nonUniformTable reads past the end of the table above its last point.
    # (A polynomial fit diverges outside its range, which made the first-wall faces
    # run away through the wall functions; see assumptions_log.md section 4.)
    mu_table=dict(T0=700.0, T1=2000.0, dT=10.0, T_top=1.0e5, ratio=1.1),
    molWeight=33.0,            # 0.67 LiF - 0.33 BeF2; no effect here

    # ---- cooling circuits (V0 scheme on V3A: ASSUMPTION) ---------------------
    # Junctions are LIMITER_M vertices; each baffle runs along the mitre
    # bisector at that vertex, i.e. it joins the matching corners of the
    # offset rings, straight across the gap.
    junctions={
        "upper_inboard": (3.68, 2.10),     # top of inboard main-chamber wall
        "upper_outboard": (4.18, 2.35),    # outer divertor baffle tip, upper X-point
        "lower_outboard": (4.18, -2.35),
        "lower_inboard": (3.68, -2.10),
    },
    junction_snap_tol=0.02,    # m, a junction must be within this of a limiter vertex
    # circuit: start junction -> end junction, going round the vessel through `via`
    circuits=[
        dict(name="inboard", label="Inboard main chamber", kind="main",
             start="upper_inboard", end="lower_inboard", via=(3.42, 0.0)),
        dict(name="udiv", label="Upper divertor", kind="divertor",
             start="upper_inboard", end="upper_outboard", via=(5.04, 4.09)),
        dict(name="outboard", label="Outboard main chamber", kind="main",
             start="upper_outboard", end="lower_outboard", via=(5.82, 0.0)),
        dict(name="ldiv", label="Lower divertor", kind="divertor",
             start="lower_outboard", end="lower_inboard", via=(5.04, -4.09)),
    ],
    inlet_velocity=2.0,        # m/s; slot width = m_dot / (rho * v * 2 pi R)
    # ASSUMPTION: each opening in the outer V wall is sized for this mean velocity
    opening_velocity=2.0,
    # bypass: "balance" -> m_total - sum(circuits) (outlet ~ T_in + dT_total);
    #         "fixed"   -> V0's fixed 0.19 m3/s (hotter outlet)
    bypass_mode="balance",
    bypass_fixed_m3_s=0.19,
    # ASSUMPTION: bypass slot on the tank top just outboard of the WC shield tip,
    # discharging downward (wall normal ~18 deg from vertical) into the upper
    # inboard bulk. Edge given by two TANK_INNER_M vertices; slot starts
    # bypass_offset along it from the first vertex.
    bypass_edge=((4.11, 4.37), (5.05, 4.68)),
    bypass_offset=0.04,
    # ASSUMPTION: hot outlet = tank top between these radii, across the apex (5.05, 4.68)
    outlet_R=(4.90, 5.20),

    # ---- wall heat (v1: each wall total spread uniformly over its patch) ----
    # gap plasma side of circuit c: W + inner V heating in its segment + its
    #   surface heat; gap tank side of c: half the outer V wall heating in its
    #   segment; bulk side of the outer V wall: the other half (one patch);
    #   WC shield heating on its FLiBe face; tank heating on its inner face.
    # Baffles and opening side walls are adiabatic (ASSUMPTION).

    # ---- mesh (cm). User spec: ~1 cm and >= 6-8 cells across the gap, 3-5 cm bulk
    wedge_deg=2.0,             # ASSUMPTION: total wedge angle (< 5 deg), symmetric about y = 0
    mesh_levels=dict(
        fine=dict(h_gap=0.6,          # 7.5 cells across the 4.5 cm gap
                  h_bulk_wall=1.5,    # ASSUMPTION: bulk cells at walls
                  h_bulk=4.0,         # bulk core (user: 3-5 cm)
                  bulk_grow=25.0,     # ASSUMPTION: distance over which bulk cells grow
                  slot_cells=5,       # ASSUMPTION: cells across each slot / opening
                  h_slot_min=0.08,    # ASSUMPTION: smallest cell (cm)
                  slot_grow=3.0),     # ASSUMPTION: refinement radius around slots
        coarse=dict(h_gap=1.1, h_bulk_wall=3.0, h_bulk=8.0, bulk_grow=30.0,
                    slot_cells=3, h_slot_min=0.15, slot_grow=4.0),
    ),

    # ---- turbulence and wall treatment -----------------------------------------
    turbulence_model="realizableKE",
    nut_wall="nutUSpaldingWallFunction",   # ASSUMPTION: valid y+ ~ 1-300 (gap y+ ~ 20-35)
    Prt=0.85,
    Sct=0.85,
    inlet_intensity=0.05,
    mixing_length_factor=0.07,  # ASSUMPTION: L = 0.07 * D_h, with D_h = 2 * slot width
    k_init=1e-3,               # ASSUMPTION: initial fields
    epsilon_init=5e-5,
    T_init=850.0,
    T_backflow=900.0,          # ASSUMPTION: T of any backflow at the outlet
    p_out=1e5,                 # Pa, fixed (absolute) pressure at the outlet

    # ---- isotopes (half-lives from openmc.data.half_life) ---------------------
    isotopes=("N16", "F20", "O19", "F18"),
    # ASSUMPTION: inlet concentration 0 for all (once-through; F-18 would really
    # recirculate through the external loop)

    # ---- numerics / run ---------------------------------------------------------
    end_iterations=dict(fine=5000, coarse=300),
    write_interval=dict(fine=500, coarse=100),
    n_procs=dict(fine=8, coarse=1),    # 1 = serial
    # ASSUMPTION: safeguards (numerics only). rhoMin/rhoMax in SIMPLE are rho at these
    # limits; with the old mu polynomial the first-wall FACE temperatures (> 4945 K,
    # where rho(T) <= 0) gave rho = 0 and a divide by zero on iteration 1.
    # limitTemperature clips CELL T. post_cfd.py reports both: the limiter status and
    # the wall faces whose rho is clamped. Neither should be active at convergence.
    T_limits=(732.0, 1400.0),  # K: FLiBe freezing point / well above any expected T
    scalar_nCorr=100,          # outer correctors of the frozen-flow scalar solves
    scalar_tolerance=1e-9,
    age_diffusion=True,        # OpenFOAM age FO: diffusivity muEff = mu + mut (Sc = 1)
    openfoam_bashrc="/usr/lib/openfoam/openfoam2606/etc/bashrc",
)


# =============================================================================
# FLiBe properties
# =============================================================================
def rho_flibe(T):
    a, b = CFG["rho_coeffs"]
    return a + b * T


# =============================================================================
# Geometry
# =============================================================================
def _unit(v):
    return v / np.linalg.norm(v)


def _mitre_ring(V: np.ndarray, d: float) -> np.ndarray:
    """Outward mitred offset (distance d) of the CCW vertex ring V (n, 2)."""
    nxt, prv = np.roll(V, -1, axis=0), np.roll(V, 1, axis=0)
    def normals(a, b):                       # outward (right-hand) normal of a CCW edge a->b
        e = b - a
        n = np.stack([e[:, 1], -e[:, 0]], axis=1)
        return n / np.linalg.norm(n, axis=1, keepdims=True)
    n1, n2 = normals(prv, V), normals(V, nxt)
    k = 1.0 + np.einsum("ij,ij->i", n1, n2)
    return V + d * (n1 + n2) / k[:, None]


def _ring_area_m2(pts_cm: np.ndarray) -> float:
    """Full 360 deg surface area of an R-Z polyline (cm in, m2 out)."""
    c = np.asarray(pts_cm) / 100.0
    dl = np.hypot(*np.diff(c, axis=0).T)
    return float((2.0 * math.pi * 0.5 * (c[1:, 0] + c[:-1, 0]) * dl).sum())


def _ring_volume_m3(poly: Polygon) -> float:
    """Full 360 deg volume of an R-Z polygon (Pappus; cm in, m3 out)."""
    return 2.0 * math.pi * poly.centroid.x * poly.area * 1e-6


def build_geometry() -> dict:
    """Vessel layers, FLiBe regions, circuits, inlets, openings, bypass, outlet (cm)."""
    import arc_v3a_paramak_model as am

    p = am.ARCParameters()
    parts = {pt.name: pt for pt in am.build_profiles(p, magnets=False)}
    limiter = orient(parts["first_wall"].cuts[0], 1.0)          # CCW
    V = np.asarray(limiter.exterior.coords)[:-1]
    n = len(V)

    # offset depths from the plasma-facing surface (cm)
    d_fw = p.first_wall_thickness
    d_v3 = d_fw + p.vv_inner_wall_thickness
    d_gap = d_v3 + p.vv_channel_thickness
    d_v5 = d_gap + p.vv_outer_wall_thickness
    rings = {0.0: V}
    for d, name in ((d_fw, "first_wall"), (d_v3, "vv_inner_wall"),
                    (d_gap, "vv_cooling_channel"), (d_v5, "vv_outer_wall")):
        rings[d] = _mitre_ring(V, d)
        mismatch = Polygon(rings[d]).symmetric_difference(parts[name].outer).area
        if mismatch > 1e-6 * parts[name].outer.area:
            raise SystemExit(f"mitre ring at {d} cm does not reproduce {name} (diff {mismatch:.3g} cm2)")
    rings[0.5 * (d_v3 + d_gap)] = _mitre_ring(V, 0.5 * (d_v3 + d_gap))   # gap centreline

    # junctions -> limiter vertex indices
    jidx = {}
    for jname, (r, z) in CFG["junctions"].items():
        dist = np.hypot(V[:, 0] - 100 * r, V[:, 1] - 100 * z)
        k = int(dist.argmin())
        if dist[k] > 100 * CFG["junction_snap_tol"]:
            raise SystemExit(f"junction {jname} ({r}, {z}) is {dist[k]:.1f} cm from the nearest limiter vertex")
        jidx[jname] = k

    def ccw_normal(i, j):
        """Outward unit normal of the limiter edge between adjacent vertices i, j."""
        a, b = (i, j) if (i + 1) % n == j else (j, i)
        e = V[b] - V[a]
        return _unit(np.array([e[1], -e[0]]))

    regions = {name: parts[name].region for name in
               ("first_wall", "vv_inner_wall", "vv_cooling_channel", "vv_outer_wall",
                "flibe_blanket", "wc_shield", "blanket_tank", "plasma")}
    rho_in = rho_flibe(CFG["T_in"])

    circuits = []
    for c in CFG["circuits"]:
        s, e = jidx[c["start"]], jidx[c["end"]]
        v = int(np.hypot(V[:, 0] - 100 * c["via"][0], V[:, 1] - 100 * c["via"][1]).argmin())
        fwd = [(s + k) % n for k in range((e - s) % n + 1)]
        bwd = [(s - k) % n for k in range((s - e) % n + 1)]
        idx = fwd if v in fwd else bwd
        if v not in idx or v in (s, e):
            raise SystemExit(f"circuit {c['name']}: via point is not between its junctions")
        seg = Polygon(np.vstack([rings[0.0][idx], rings[d_v5][idx][::-1]]))
        gap = Polygon(np.vstack([rings[d_v3][idx], rings[d_gap][idx][::-1]]))
        for nm, g in (("segment", seg), ("gap", gap)):
            if not g.is_valid:
                raise SystemExit(f"circuit {c['name']}: {nm} polygon is invalid")
        circuits.append(dict(c, idx=idx, segment=seg, gap=gap,
                             fw_line=rings[0.0][idx], plasma_face=rings[d_v3][idx],
                             tank_face=rings[d_gap][idx], centreline=rings[0.5 * (d_v3 + d_gap)][idx]))

    # checks: circuits tile the gap exactly
    gap_all = regions["vv_cooling_channel"]
    tiled = shapely.union_all([c["gap"] for c in circuits])
    if abs(tiled.area - gap_all.area) > 1e-6 * gap_all.area or \
            abs(sum(c["gap"].area for c in circuits) - gap_all.area) > 1e-6 * gap_all.area:
        raise SystemExit("circuit gap segments do not tile the vessel gap")

    geo = dict(parts=parts, regions=regions, V=V, rings=rings, jidx=jidx,
               depths=dict(fw=d_fw, v3=d_v3, gap=d_gap, v5=d_v5),
               t_v5=p.vv_outer_wall_thickness, circuits=circuits,
               ccw_normal=ccw_normal, rho_in=rho_in)
    return geo


def size_openings(geo: dict, flows: dict):
    """Inlet slot on the tank-side face at each circuit start and an opening
    through the outer V wall at each end, sized from the circuit flows."""
    rings, d = geo["rings"], geo["depths"]
    v_in, v_op = CFG["inlet_velocity"], CFG["opening_velocity"]
    for c in geo["circuits"]:
        mdot = flows[c["name"]]["mdot"]
        # inlet: from the baffle corner along the first tank-side edge
        i0, i1 = c["idx"][0], c["idx"][1]
        P0, P1 = rings[d["gap"]][i0], rings[d["gap"]][i1]
        e = _unit(P1 - P0)
        w = 1.0
        for _ in range(20):
            R_mid = (P0 + 0.5 * w * e)[0] / 100.0
            w = 100.0 * mdot / (geo["rho_in"] * v_in * 2.0 * math.pi * R_mid)
        if w > 0.8 * np.linalg.norm(P1 - P0):
            raise SystemExit(f"{c['name']}: inlet slot {w:.1f} cm does not fit on its edge")
        c["inlet"] = np.array([P0, P0 + w * e])
        c["inlet_width_cm"] = w
        c["inlet_R_m"] = R_mid
        # opening: removes the outer V wall from the baffle (mitre bisector) back
        # along the circuit's last edge. ASSUMPTION: its length L is the clear
        # length on the bulk-side face (the narrowest section, because the
        # bisector side is slanted), sized for opening_velocity.
        j1, j0 = c["idx"][-1], c["idx"][-2]
        m6, m9 = rings[d["gap"]][j1], rings[d["v5"]][j1]
        e = _unit(rings[d["v5"]][j0] - m9)
        nrm = geo["ccw_normal"](j0, j1)
        rho_out = rho_flibe(CFG["T_in"] + CFG["dT_circuit"])
        L = 1.0
        for _ in range(20):
            R_mid = (m9 + 0.5 * L * e)[0] / 100.0
            L = 100.0 * mdot / (rho_out * v_op * 2.0 * math.pi * R_mid)
        if L > 0.8 * np.linalg.norm(rings[d["v5"]][j0] - m9):
            raise SystemExit(f"{c['name']}: opening {L:.1f} cm does not fit on its edge")
        Q9 = m9 + L * e
        Q6 = Q9 - geo["t_v5"] * nrm
        edge6 = LineString([m6, rings[d["gap"]][j0]])
        if edge6.distance(shapely.Point(Q6)) > 1e-6:
            raise SystemExit(f"{c['name']}: opening does not land on the gap's tank-side face")
        c["opening"] = Polygon([m6, Q6, Q9, m9])
        c["Q6"], c["Q9"] = Q6, Q9
        c["opening_len_cm"] = L
        c["opening_gap_side_cm"] = float(np.linalg.norm(Q6 - m6))
        c["opening_R_m"] = R_mid


def bypass_and_outlet(geo: dict, mdot_bypass: float):
    """Bypass slot on the tank top (downward discharge) and the hot outlet."""
    A, B = (100.0 * np.array(q) for q in CFG["bypass_edge"])
    e = _unit(B - A)
    P0 = A + 100.0 * CFG["bypass_offset"] * e
    w = 1.0
    for _ in range(20):
        R_mid = (P0 + 0.5 * w * e)[0] / 100.0
        w = 100.0 * mdot_bypass / (geo["rho_in"] * CFG["inlet_velocity"] * 2.0 * math.pi * R_mid) \
            if mdot_bypass > 0 else 0.0
    geo["bypass"] = np.array([P0, P0 + w * e])
    geo["bypass_width_cm"] = w
    geo["bypass_R_m"] = R_mid
    # outlet polyline across the tank apex
    tank = np.asarray(orient(geo["regions"]["flibe_blanket"], 1.0).exterior.coords)
    top = [k for k in range(len(tank) - 1) if tank[k][1] > 400 and tank[k + 1][1] > 400]
    r0, r1 = (100.0 * r for r in CFG["outlet_R"])
    pts = []
    for k in top:
        a, b = tank[k], tank[k + 1]
        for r in (r0, r1):
            if min(a[0], b[0]) <= r <= max(a[0], b[0]) and a[0] != b[0]:
                pts.append(a + (r - a[0]) / (b[0] - a[0]) * (b - a))
    apex = tank[np.argmax(tank[:, 1])]
    if len(pts) != 2 or not (r0 < apex[0] < r1):
        raise SystemExit("outlet radii must straddle the tank apex on the tank top")
    pts.sort(key=lambda q: q[0])
    geo["outlet"] = np.array([pts[0], apex, pts[1]])


# =============================================================================
# OpenMC data
# =============================================================================
def load_openmc() -> dict:
    run = Path(CFG["openmc_run"])
    npz_path, csv_path = run / "source_fields.npz", run / "cell_totals.csv"
    for f in (npz_path, csv_path):
        if not f.exists():
            raise SystemExit(f"{f} not found: run 'arc_v3a_openmc.py post' first")
    d = dict(np.load(npz_path))
    need = ["r_edges_cm", "z_edges_cm", "bin_ring_volume_m3", "neutron_rate", "heating_flibe_W_m3",
            "heating_tungsten_W_m3", "heating_vanadium_W_m3", "heating_tungsten_carbide_W_m3",
            "heating_ss316l_W_m3"] + [f"{i}_generation_per_m3_s" for i in ("N16", "F18", "O19", "F20")]
    missing = [k for k in need if k not in d]
    if missing:
        raise SystemExit(f"source_fields.npz lacks {missing}: re-run 'post' with the current script")
    rows = list(csv.DictReader(open(csv_path)))
    cells = {r["label"]: r for r in rows}
    tbr = sum(float(r["H3_per_s"] or 0) for r in rows) / float(d["neutron_rate"])
    heat = sum(float(r["heating_MW"] or 0) for r in rows)
    lo, hi = CFG["openmc_tbr_range"]
    hlo, hhi = CFG["openmc_heating_MW_range"]
    print(f"OpenMC run {run}: TBR {tbr:.3f}, total nuclear heating {heat:.1f} MW")
    if not (lo <= tbr <= hi and hlo <= heat <= hhi):
        raise SystemExit("STOP: TBR or total heating outside the expected range; the run may not "
                         "have been post-processed with the current arc_v3a_openmc.py")
    return dict(f=d, cells=cells, tbr=tbr, heat_MW=heat)


def _bin_grid(f):
    re_, ze = f["r_edges_cm"], f["z_edges_cm"]
    return 0.5 * (re_[1:] + re_[:-1]), 0.5 * (ze[1:] + ze[:-1]), np.diff(re_), np.diff(ze)


def classify_bins(f, mask, regions: dict):
    """Label each True bin of `mask` with one of `regions` (dict name -> polygon):
    the region holding the bin centre; else the one with the largest overlap;
    else the nearest. Returns (i, j, label index, method counts)."""
    rc, zc, dr, dz = _bin_grid(f)
    i, j = np.nonzero(mask)
    x, y = rc[i], zc[j]
    names = list(regions)
    lab = np.full(len(i), -1)
    for k, nm in enumerate(names):
        lab[(lab == -1) & shapely.contains_xy(regions[nm], x, y)] = k
    counts = dict(centre=int((lab >= 0).sum()), overlap=0, nearest=0)
    rest = np.nonzero(lab == -1)[0]
    if rest.size:
        bx = shapely.box(x[rest] - dr[i[rest]] / 2, y[rest] - dz[j[rest]] / 2,
                         x[rest] + dr[i[rest]] / 2, y[rest] + dz[j[rest]] / 2)
        ov = np.stack([shapely.area(shapely.intersection(bx, regions[nm])) for nm in names], axis=1)
        pts = shapely.points(x[rest], y[rest])
        dist = np.stack([shapely.distance(regions[nm], pts) for nm in names], axis=1)
        has = ov.max(axis=1) > 0
        lab[rest] = np.where(has, ov.argmax(axis=1), dist.argmin(axis=1))
        counts["overlap"], counts["nearest"] = int(has.sum()), int((~has).sum())
    return i, j, lab, counts


def circuit_heat_loads(geo: dict, om: dict) -> dict:
    """Integrate the OpenMC mesh heating over each circuit's vessel segment (W)."""
    f, R = om["f"], geo["regions"]
    V = f["bin_ring_volume_m3"]
    segs = {c["name"]: c["segment"] for c in geo["circuits"]}
    snames = list(segs)
    out = {nm: dict(W=0.0, V3=0.0, V5=0.0, flibe_gap=0.0) for nm in snames}
    tot = {}

    def per_segment(field, mask, key):
        i, j, lab, _ = classify_bins(f, mask, segs)
        T = field[i, j] * V[i, j]
        for k, nm in enumerate(snames):
            out[nm][key] += float(T[lab == k].sum())

    W = f["heating_tungsten_W_m3"]
    per_segment(W, W > 0, "W")
    Vn = f["heating_vanadium_W_m3"]
    i, j, lab, cnt_v = classify_bins(f, Vn > 0, {"V3": R["vv_inner_wall"], "V5": R["vv_outer_wall"]})
    for k, key in enumerate(("V3", "V5")):
        m = np.zeros_like(Vn, dtype=bool)
        m[i[lab == k], j[lab == k]] = True
        per_segment(Vn, m, key)
    Fl = f["heating_flibe_W_m3"]
    i, j, lab, cnt_f = classify_bins(f, Fl > 0, {"gap": R["vv_cooling_channel"], "bulk": R["flibe_blanket"]})
    gmask = np.zeros_like(Fl, dtype=bool)
    gmask[i[lab == 0], j[lab == 0]] = True
    per_segment(Fl, gmask, "flibe_gap")
    tot["flibe_bulk"] = float((Fl[i[lab == 1], j[lab == 1]] * V[i[lab == 1], j[lab == 1]]).sum())
    tot["WC"] = float((f["heating_tungsten_carbide_W_m3"] * V).sum())
    tot["SS"] = float((f["heating_ss316l_W_m3"] * V).sum())
    for key in ("W", "V3", "V5", "flibe_gap"):
        tot[key] = sum(out[nm][key] for nm in snames)

    # check against the per-volume tallies
    ref = {"W": "tungsten_2", "V3": "vanadium_3", "V5": "vanadium_5", "flibe_gap": "flibe_vessel_channel",
           "flibe_bulk": "flibe_bulk_blanket", "WC": "tungsten_carbide_7", "SS": "ss316l_8"}
    print("\nOpenMC mesh heating by region vs cell tallies (MW, full machine)")
    for key, label in ref.items():
        cell = float(om["cells"][label]["heating_MW"])
        print(f"  {key:10s} mesh {tot[key] * 1e-6:9.3f}   cell {label:22s} {cell:9.3f}   "
              f"diff {100 * (tot[key] * 1e-6 / cell - 1):+.3f} %")
    print(f"  bins classified by centre/overlap/nearest: vanadium {cnt_v}, FLiBe {cnt_f}")
    return dict(per_circuit=out, totals=tot)


# =============================================================================
# Flows
# =============================================================================
def circuit_flows(geo: dict, loads: dict) -> dict:
    cp, dTc = CFG["cp"], CFG["dT_circuit"]
    P_rad = CFG["P_alpha_MW"] + CFG["P_aux_MW"] - CFG["P_sep_MW"]
    divs = [c for c in geo["circuits"] if c["kind"] == "divertor"]
    mains = [c for c in geo["circuits"] if c["kind"] == "main"]
    A_main = {c["name"]: _ring_area_m2(c["fw_line"]) for c in mains}
    flows = {}
    for c in geo["circuits"]:
        L = loads["per_circuit"][c["name"]]
        if c["kind"] == "divertor":
            P_s = 1e6 * CFG["P_sep_MW"] / len(divs)
        else:
            P_s = 1e6 * P_rad * A_main[c["name"]] / sum(A_main.values())
        Q_nuc = L["W"] + L["V3"] + L["flibe_gap"] + 0.5 * L["V5"]
        Q = Q_nuc + P_s
        mdot = Q / (cp * dTc)
        T_exit = CFG["T_in"] + Q / (mdot * cp)
        rho_m = rho_flibe(CFG["T_in"] + 0.5 * dTc)
        vol = _ring_volume_m3(c["gap"])
        flows[c["name"]] = dict(Q_nuc=Q_nuc, P_surf=P_s, Q=Q, mdot=mdot, T_exit=T_exit, V_gap=vol,
                                transit=vol / (mdot / rho_m), A_fw=_ring_area_m2(c["fw_line"]))
    t = loads["totals"]
    Q_total = (t["W"] + t["V3"] + t["V5"] + t["flibe_gap"] + t["flibe_bulk"] + t["WC"] + t["SS"]
               + 1e6 * (CFG["P_alpha_MW"] + CFG["P_aux_MW"]))
    m_total_target = Q_total / (cp * CFG["dT_total"])
    m_circ = sum(v["mdot"] for v in flows.values())
    note = ""
    if CFG["bypass_mode"] == "fixed":
        m_bp = CFG["bypass_fixed_m3_s"] * geo["rho_in"]
    else:
        m_bp = m_total_target - m_circ
        if m_bp < 0:
            note = f"bypass came out negative ({m_bp:.0f} kg/s); set to zero"
            m_bp = 0.0
    m_total = m_circ + m_bp
    return dict(circuits=flows, Q_total=Q_total, m_total=m_total, m_bypass=m_bp, m_circ=m_circ,
                T_out=CFG["T_in"] + Q_total / (m_total * cp), note=note)


def print_flow_table(geo, loads, fl):
    print("\nCooling circuits (full machine)")
    hdr = (f"  {'circuit':22s} {'W':>6s} {'V_in':>6s} {'gapF':>6s} {'V_out/2':>7s} {'surf':>6s} {'Q':>7s}"
           f" {'m_dot':>7s} {'V_dot':>6s} {'T_exit':>6s} {'V_gap':>6s} {'transit':>7s} {'inlet':>6s} {'opening':>7s}")
    print(hdr)
    print(f"  {'':22s} {'MW':>6s} {'MW':>6s} {'MW':>6s} {'MW':>7s} {'MW':>6s} {'MW':>7s}"
          f" {'kg/s':>7s} {'m3/s':>6s} {'K':>6s} {'m3':>6s} {'s':>7s} {'cm':>6s} {'cm':>7s}")
    for c in geo["circuits"]:
        L, F = loads["per_circuit"][c["name"]], fl["circuits"][c["name"]]
        rho_m = rho_flibe(CFG["T_in"] + 0.5 * CFG["dT_circuit"])
        print(f"  {c['label']:22s} {L['W'] / 1e6:6.1f} {L['V3'] / 1e6:6.1f} {L['flibe_gap'] / 1e6:6.1f}"
              f" {0.5 * L['V5'] / 1e6:7.1f} {F['P_surf'] / 1e6:6.1f} {F['Q'] / 1e6:7.1f} {F['mdot']:7.0f}"
              f" {F['mdot'] / rho_m:6.3f} {F['T_exit']:6.1f} {F['V_gap']:6.2f} {F['transit']:7.2f}"
              f" {c['inlet_width_cm']:6.2f} {c['opening_len_cm']:7.2f}")
    print(f"  {'Bypass':22s} {'':>6s} {'':>6s} {'':>6s} {'':>7s} {'':>6s} {'':>7s} {fl['m_bypass']:7.0f}"
          f" {fl['m_bypass'] / geo['rho_in']:6.3f} {'':>6s} {'':>6s} {'':>7s} {geo['bypass_width_cm']:6.2f}")
    t = loads["totals"]
    print(f"\n  heat into FLiBe: {fl['Q_total'] / 1e6:.1f} MW = nuclear {(fl['Q_total'] / 1e6 - CFG['P_alpha_MW'] - CFG['P_aux_MW']):.1f}"
          f" + surface {CFG['P_alpha_MW'] + CFG['P_aux_MW']:.0f} (Psep {CFG['P_sep_MW']:.0f} to divertors,"
          f" {CFG['P_alpha_MW'] + CFG['P_aux_MW'] - CFG['P_sep_MW']:.0f} radiation on main chamber)")
    print(f"  bulk-side heat: FLiBe {t['flibe_bulk'] / 1e6:.1f} + outer V wall/2 {0.5 * t['V5'] / 1e6:.1f}"
          f" + WC shield {t['WC'] / 1e6:.2f} + tank {t['SS'] / 1e6:.2f} MW")
    print(f"  circuits {fl['m_circ']:.0f} kg/s + bypass {fl['m_bypass']:.0f} kg/s ({CFG['bypass_mode']})"
          f" = {fl['m_total']:.0f} kg/s; mixed outlet {fl['T_out']:.1f} K")
    if fl["note"]:
        print(f"  NOTE: {fl['note']}")
    print(f"  bypass slot: R {geo['bypass_R_m']:.3f} m on the tank top; outlet: R {CFG['outlet_R'][0]:.2f}"
          f"-{CFG['outlet_R'][1]:.2f} m across the apex")
    print("  junction snap (limiter vertex used):")
    for jn, k in geo["jidx"].items():
        print(f"    {jn:15s} requested {CFG['junctions'][jn]}  ->  ({geo['V'][k][0] / 100:.3f}, {geo['V'][k][1] / 100:.3f})")


# =============================================================================
# Plot
# =============================================================================
SHORT = dict(inboard="inboard", udiv="upper div.", outboard="outboard", ldiv="lower div.")
COLORS = dict(main_in="#2a78d6", main_out="#1baf7a", divertor="#eb6834",   # categorical slots 1-3
              solid="#b9b8b2", wall="#7d7c76", flibe="#e7edf3", ink="#0b0b0b", ink2="#52514e")


def _ccolor(c):
    if c["kind"] == "divertor":
        return COLORS["divertor"]
    return COLORS["main_in"] if c["name"] == "inboard" else COLORS["main_out"]


def _patch(ax, geom, **kw):
    from matplotlib.patches import PathPatch
    from matplotlib.path import Path as MPath
    polys = [geom] if geom.geom_type == "Polygon" else list(geom.geoms)
    verts, codes = [], []
    for poly in polys:
        for ring in [poly.exterior, *poly.interiors]:
            xy = np.asarray(ring.coords) / 100.0
            verts += xy.tolist()
            codes += [MPath.MOVETO] + [MPath.LINETO] * (len(xy) - 2) + [MPath.CLOSEPOLY]
    ax.add_patch(PathPatch(MPath(verts, codes), **kw))


def _arrows_along(ax, pts_cm, color, fracs, lw=1.6, size=0.12):
    line = LineString(pts_cm / 100.0)
    for s in fracs:
        a = np.array(line.interpolate(s * line.length - 0.5 * size).coords[0])
        b = np.array(line.interpolate(s * line.length + 0.5 * size).coords[0])
        ax.annotate("", xy=b, xytext=a,
                    arrowprops=dict(arrowstyle="-|>,head_width=0.35,head_length=0.7", color=color, lw=lw))


def plot_circuits(geo: dict, fl: dict, path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    R, rings, d = geo["regions"], geo["rings"], geo["depths"]
    fig = plt.figure(figsize=(15, 10), facecolor="white")
    gs = fig.add_gridspec(2, 3, width_ratios=[1.35, 1, 1], wspace=0.25, hspace=0.22)
    ax = fig.add_subplot(gs[:, 0])
    zooms = [fig.add_subplot(gs[r, c]) for r in range(2) for c in (1, 2)]

    def base(a):
        _patch(a, R["blanket_tank"], facecolor=COLORS["wall"], edgecolor="none")
        _patch(a, R["wc_shield"], facecolor=COLORS["solid"], edgecolor="none")
        _patch(a, R["flibe_blanket"], facecolor=COLORS["flibe"], edgecolor="none")
        for nm in ("first_wall", "vv_inner_wall", "vv_outer_wall"):
            _patch(a, R[nm], facecolor=COLORS["wall"], edgecolor="none")
        for c in geo["circuits"]:
            _patch(a, c["gap"], facecolor=_ccolor(c), edgecolor="none")
            if "opening" in c:
                _patch(a, c["opening"], facecolor=_ccolor(c), edgecolor="none")
        for jn, k in geo["jidx"].items():                         # baffles
            both_end = sum(c["end"] == jn for c in geo["circuits"]) == 2
            outer = rings[d["v5"]][k] if both_end else rings[d["gap"]][k]
            a.plot(np.array([rings[d["v3"]][k][0], outer[0]]) / 100,
                   np.array([rings[d["v3"]][k][1], outer[1]]) / 100,
                   color=COLORS["ink"], lw=2.2, solid_capstyle="butt", zorder=5)
        a.set_aspect("equal")
        a.tick_params(labelsize=8, colors=COLORS["ink2"])
        for s in a.spines.values():
            s.set_color("#c3c2b7")

    # ---- overview -------------------------------------------------------------
    base(ax)
    for c in geo["circuits"]:
        ax.plot(*(c["centreline"] / 100).T, color=_ccolor(c), lw=3.2, solid_capstyle="butt")
        _arrows_along(ax, c["centreline"], _ccolor(c), (0.2, 0.5, 0.8), lw=2.0, size=0.14)
        ax.plot(*(c["inlet"].mean(axis=0) / 100), marker="v", ms=8, color=COLORS["ink"], zorder=6)
        oc = np.asarray(c["opening"].centroid.coords[0]) / 100
        ax.plot(*oc, marker="o", ms=7, mfc="white", mec=COLORS["ink"], mew=1.5, zorder=6)
    bp = geo["bypass"].mean(axis=0) / 100
    ax.annotate("", xy=bp + np.array([0.08, -0.32]), xytext=bp,
                arrowprops=dict(arrowstyle="-|>", color=COLORS["ink"], lw=1.6))
    ax.plot(*bp, marker="v", ms=8, color=COLORS["ink"], zorder=6)
    ax.text(bp[0] - 0.05, bp[1] + 0.12, f"bypass in\n{fl['m_bypass']:.0f} kg/s", ha="right", va="bottom",
            fontsize=8, color=COLORS["ink"])
    ol = geo["outlet"] / 100
    ax.plot(*ol.T, color=COLORS["ink"], lw=4, solid_capstyle="butt", zorder=6)
    ax.annotate(f"hot outlet\n{fl['m_total']:.0f} kg/s, {fl['T_out']:.0f} K", xy=ol[1], xytext=(ol[1][0] + 0.75, ol[1][1] + 0.05),
                fontsize=8, color=COLORS["ink"], arrowprops=dict(arrowstyle="-", color=COLORS["ink2"], lw=0.8))
    # direct circuit labels
    lab_pos = {"inboard": (3.62, 0.35, "left"), "outboard": (6.05, -0.35, "left"),
               "udiv": (5.85, 3.35, "left"), "ldiv": (5.85, -3.35, "left")}
    for c in geo["circuits"]:
        F = fl["circuits"][c["name"]]
        x, y, ha = lab_pos[c["name"]]
        ax.text(x, y, f"{c['label']}\n{F['mdot']:.0f} kg/s, {F['Q'] / 1e6:.0f} MW\ntransit {F['transit']:.1f} s",
                ha=ha, va="center", fontsize=8.5, color=COLORS["ink"],
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.8, pad=1.5))
    for jn, k in geo["jidx"].items():
        p = geo["V"][k] / 100
        ax.annotate(f"({p[0]:.2f}, {p[1]:+.2f})", xy=p, xytext=(p[0] + 0.12, p[1] + (0.1 if p[1] > 0 else -0.1)),
                    fontsize=7.5, color=COLORS["ink2"], ha="left", va="center")
    ax.set_xlim(1.2, 7.6)
    ax.set_ylim(-4.95, 5.25)
    ax.set_xlabel("R (m)", fontsize=9, color=COLORS["ink2"])
    ax.set_ylabel("Z (m)", fontsize=9, color=COLORS["ink2"])
    ax.set_title("ARC V3A vessel-gap cooling circuits (V0 scheme, ASSUMPTION)", fontsize=10, color=COLORS["ink"],
                 loc="left")
    handles = [Line2D([], [], color=COLORS["main_in"], lw=4, label="Inboard main chamber"),
               Line2D([], [], color=COLORS["main_out"], lw=4, label="Outboard main chamber"),
               Line2D([], [], color=COLORS["divertor"], lw=4, label="Upper / lower divertor"),
               Line2D([], [], color=COLORS["ink"], lw=2.2, label="baffle (zero thickness)"),
               Line2D([], [], color=COLORS["ink"], marker="v", ls="none", ms=7, label="inlet, 800 K"),
               Line2D([], [], mfc="white", mec=COLORS["ink"], marker="o", ls="none", ms=7, mew=1.5,
                      label="opening to bulk")]
    ax.legend(handles=handles, loc="lower left", fontsize=7.5, frameon=False)

    # ---- junction zooms ------------------------------------------------------
    half = 0.12
    box_kw = dict(facecolor="white", edgecolor="none", alpha=0.85, pad=1.0)
    for a, (jn, k) in zip(zooms, geo["jidx"].items()):
        base(a)
        p = geo["V"][k] / 100
        ctr = rings[d["gap"]][k] / 100
        for c in geo["circuits"]:
            col = _ccolor(c)
            if c["start"] == jn:                                   # inlet
                P = c["inlet"] / 100
                a.plot(*P.T, color=COLORS["ink"], lw=4, solid_capstyle="butt", zorder=6)
                mid = P.mean(axis=0)
                nrm = -geo["ccw_normal"](c["idx"][0], c["idx"][1])   # into the gap
                a.annotate("", xy=mid + 0.012 * nrm, xytext=mid - 0.022 * nrm,
                           arrowprops=dict(arrowstyle="-|>", color=COLORS["ink"], lw=1.4), zorder=7)
                a.text(*(mid - 0.045 * nrm), f"{SHORT[c['name']]} inlet\n{c['inlet_width_cm']:.2f} cm",
                       fontsize=7, color=COLORS["ink"], ha="center", va="center", bbox=box_kw, zorder=8)
            if c["end"] == jn:                                     # opening
                oc = np.asarray(c["opening"].centroid.coords[0]) / 100
                nrm = geo["ccw_normal"](c["idx"][-2], c["idx"][-1])
                a.annotate("", xy=oc + 0.04 * nrm, xytext=oc - 0.01 * nrm,
                           arrowprops=dict(arrowstyle="-|>", color=col, lw=1.6), zorder=7)
                a.text(*(oc + 0.06 * nrm), f"{SHORT[c['name']]} opening\n{c['opening_len_cm']:.2f} cm",
                       fontsize=7, color=COLORS["ink"], ha="center", va="center", bbox=box_kw, zorder=8)
            seg = c["centreline"] / 100
            near = np.hypot(*(seg - p).T) < 0.9 * half
            if near.any():
                pts = c["centreline"][np.nonzero(near)[0].min(): np.nonzero(near)[0].max() + 2]
                if len(pts) >= 2:
                    _arrows_along(a, pts, col, (0.6,), lw=1.4, size=0.04)
        a.plot(*p, marker="+", ms=8, color=COLORS["ink2"])
        a.set_xlim(ctr[0] - half, ctr[0] + half)
        a.set_ylim(ctr[1] - half, ctr[1] + half)
        a.set_title(f"{jn.replace('_', ' ')} junction  ({p[0]:.2f}, {p[1]:+.2f}) m", fontsize=9,
                    color=COLORS["ink"], loc="left")
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"\nwrote {path}")


# =============================================================================
# Viscosity table and decay data
# =============================================================================
def viscosity_table() -> dict:
    """mu = A exp(B/T) as (T, mu) points for OpenFOAM tabulatedTransport (linear
    interpolation). Reports the interpolation error evaluated the way OpenFOAM does."""
    A, B = CFG["mu_arrhenius"]
    t = CFG["mu_table"]
    T = list(np.arange(t["T0"], t["T1"] + 0.5 * t["dT"], t["dT"]))
    while T[-1] < t["T_top"]:
        T.append(min(T[-1] * t["ratio"], t["T_top"]))
    T = np.array(T)
    mu = A * np.exp(B / T)
    out = dict(points=[(float(x), float(y)) for x, y in zip(T, mu)], n=len(T))
    for key, (lo, hi) in (("max_err_700_1400K", (700.0, 1400.0)), ("max_err_all", (T[0], T[-1]))):
        Te = np.linspace(lo, hi, 200001)
        out[key] = float(np.abs(np.interp(Te, T, mu) / (A * np.exp(B / Te)) - 1.0).max())
    return out


def decay_data() -> dict:
    import openmc.data
    out = {}
    for iso in CFG["isotopes"]:
        t = float(openmc.data.half_life(iso))
        out[iso] = dict(half_life_s=t, lam=math.log(2.0) / t)
    return out


# =============================================================================
# Fluid-domain topology (cm) for Gmsh
# =============================================================================
def _on_segment(p, a, b, tol=1e-6):
    ab = b - a
    t = float((p - a) @ ab / (ab @ ab))
    return (np.linalg.norm(a + t * ab - p) < tol and -1e-12 < t < 1 + 1e-12), t


def _insert(ring, pts, tol=1e-6):
    """Insert points lying on the edges of a closed vertex ring."""
    ring = [np.asarray(q, float) for q in ring]
    for p in pts:
        p = np.asarray(p, float)
        if any(np.linalg.norm(p - q) < tol for q in ring):
            continue
        for k in range(len(ring)):
            on, t = _on_segment(p, ring[k], ring[(k + 1) % len(ring)], tol)
            if on and 0.0 < t < 1.0:
                ring.insert(k + 1, p)
                break
        else:
            raise SystemExit(f"point {p} does not lie on the ring")
    return ring


def domain_loops(geo: dict) -> list:
    """Surfaces as loops of vertices with one label per edge (edge k joins vertex
    k and k+1): a patch name, 'internal' (conformal interface) or 'baffles'
    (zero-thickness wall: each side gets its own curve, so nodes are duplicated)."""
    rings, d = geo["rings"], geo["depths"]
    V3, G, R9 = rings[d["v3"]], rings[d["gap"]], rings[d["v5"]]
    ends = {}
    for c in geo["circuits"]:
        ends.setdefault(c["end"], []).append(c["name"])
    surfaces = []
    for c in geo["circuits"]:
        idx, nm = c["idx"], c["name"]
        n = len(idx)
        if n < 3:
            raise SystemExit(f"circuit {nm} spans fewer than two limiter edges")
        verts = [V3[i] for i in idx] + [G[idx[-1]], c["Q6"]] + [G[i] for i in idx[-2:0:-1]] \
            + [c["inlet"][1], G[idx[0]]]
        labels = [f"fw_{nm}"] * (n - 1) + ["baffles", "internal"] + [f"vvGap_{nm}"] * (n - 1) \
            + [f"inlet_{nm}", "baffles"]
        assert len(verts) == len(labels) == 2 * n + 2
        surfaces.append(dict(name=f"gap_{nm}", zone=f"gap_{nm}", loops=[(verts, labels)]))
        side = "baffles" if len(ends[c["end"]]) == 2 else "openingSides"
        surfaces.append(dict(name=f"open_{nm}", zone=f"gap_{nm}",
                             loops=[([G[idx[-1]], c["Q6"], c["Q9"], R9[idx[-1]]],
                                     ["internal", "openingSides", "internal", side])]))
    # bulk: exterior (tank + shield face) and the outer V-wall face as a hole
    ext = np.asarray(orient(geo["regions"]["flibe_blanket"], 1.0).exterior.coords)[:-1]
    extra = list(geo["outlet"][[0, 2]])
    if geo["bypass_width_cm"] > 0:
        extra += list(geo["bypass"])
    ext = _insert(ext, extra)
    shield_b = geo["regions"]["wc_shield"].exterior
    out_line = LineString(geo["outlet"])
    byp_line = LineString(geo["bypass"]) if geo["bypass_width_cm"] > 0 else None
    ext_lab = []
    for k in range(len(ext)):
        a, b = ext[k], ext[(k + 1) % len(ext)]
        mid = shapely.Point(0.5 * (a + b))
        if byp_line is not None and byp_line.distance(mid) < 1e-6:
            ext_lab.append("inlet_bypass")
        elif out_line.distance(mid) < 1e-6:
            ext_lab.append("outlet")
        elif shield_b.distance(mid) < 1e-6:
            ext_lab.append("shield")
        else:
            ext_lab.append("tank")
    hole = _insert(R9, [c["Q9"] for c in geo["circuits"]])
    iface = [(c["Q9"], R9[c["idx"][-1]]) for c in geo["circuits"]]
    hole_lab = []
    for k in range(len(hole)):
        a, b = hole[k], hole[(k + 1) % len(hole)]
        is_if = any((np.allclose(a, p) and np.allclose(b, q)) or (np.allclose(a, q) and np.allclose(b, p))
                    for p, q in iface)
        hole_lab.append("internal" if is_if else "vvBulk")
    surfaces.append(dict(name="bulk", zone="bulk", loops=[(ext, ext_lab), (hole, hole_lab)]))
    return surfaces


# =============================================================================
# Mesh: Gmsh 2D -> one-cell wedge -> Gmsh 2.2 file for gmshToFoam
# =============================================================================
def mesh_2d(geo: dict, level: str) -> dict:
    import gmsh

    lv = CFG["mesh_levels"][level]
    surfaces = domain_loops(geo)
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add("arc_v3a_flibe")
    gmsh.option.setNumber("Geometry.AutoCoherence", 0)      # keep the duplicated baffle curves
    pts, lines, line_label, line_surfs = {}, {}, {}, {}

    def P(p):
        key = (round(float(p[0]), 6), round(float(p[1]), 6))
        if key not in pts:
            pts[key] = gmsh.model.geo.addPoint(p[0] / 100.0, 0.0, p[1] / 100.0)   # x = R, z = Z (m)
        return key

    surf_tag = {}
    for s in surfaces:
        loop_tags = []
        for verts, labels in s["loops"]:
            ctags = []
            for k, lab in enumerate(labels):
                ka, kb = P(verts[k]), P(verts[(k + 1) % len(verts)])
                if ka == kb:
                    raise SystemExit(f"{s['name']}: zero-length edge at {ka}")
                if lab == "baffles":
                    t, sign = gmsh.model.geo.addLine(pts[ka], pts[kb]), 1
                    line_label[t] = lab
                else:
                    key = (min(ka, kb), max(ka, kb))
                    if key not in lines:
                        lines[key] = (gmsh.model.geo.addLine(pts[ka], pts[kb]), ka)
                        line_label[lines[key][0]] = lab
                    t, start = lines[key]
                    if line_label[t] != lab:
                        raise SystemExit(f"edge {key} labelled both {line_label[t]} and {lab}")
                    sign = 1 if start == ka else -1
                line_surfs.setdefault(t, []).append(s["name"])
                ctags.append(sign * t)
            loop_tags.append(gmsh.model.geo.addCurveLoop(ctags))
        surf_tag[s["name"]] = gmsh.model.geo.addPlaneSurface(loop_tags)
    gmsh.model.geo.synchronize()
    for t, lab in line_label.items():
        if len(line_surfs[t]) != (2 if lab == "internal" else 1):
            raise SystemExit(f"curve {t} ({lab}) bounds {line_surfs[t]}")

    def curve(pa, pb):
        ka, kb = P(pa), P(pb)
        return lines[(min(ka, kb), max(ka, kb))][0]

    # ---- size fields ----------------------------------------------------------
    F = gmsh.model.mesh.field
    fields = []
    gap_surfs = [surf_tag[s["name"]] for s in surfaces if s["zone"] != "bulk"]
    gap_curves = sorted({t for t, ss in line_surfs.items() if any(x != "bulk" for x in ss)})
    f = F.add("MathEval")
    F.setString(f, "F", f"{lv['h_gap'] / 100.0}")
    r = F.add("Restrict")
    F.setNumber(r, "InField", f)
    F.setNumbers(r, "SurfacesList", gap_surfs)
    F.setNumbers(r, "CurvesList", gap_curves)
    fields.append(r)
    wall_curves = [t for t, lab in line_label.items() if lab in ("vvBulk", "shield", "tank", "outlet", "inlet_bypass")]
    d = F.add("Distance")
    F.setNumbers(d, "CurvesList", wall_curves)
    F.setNumber(d, "Sampling", 400)
    th = F.add("Threshold")
    F.setNumber(th, "InField", d)
    F.setNumber(th, "SizeMin", lv["h_bulk_wall"] / 100.0)
    F.setNumber(th, "SizeMax", lv["h_bulk"] / 100.0)
    F.setNumber(th, "DistMin", 0.0)
    F.setNumber(th, "DistMax", lv["bulk_grow"] / 100.0)
    fields.append(th)
    rings, dd = geo["rings"], geo["depths"]
    slots = []                                       # (curve, width cm, label)
    for c in geo["circuits"]:
        e = c["idx"][-1]
        slots.append((curve(c["inlet"][1], rings[dd["gap"]][c["idx"][0]]), c["inlet_width_cm"], f"inlet_{c['name']}"))
        slots.append((curve(rings[dd["gap"]][e], c["Q6"]), c["opening_len_cm"], f"opening_{c['name']} (gap side)"))
        slots.append((curve(c["Q9"], rings[dd["v5"]][e]), c["opening_len_cm"], f"opening_{c['name']} (bulk side)"))
    if geo["bypass_width_cm"] > 0:
        slots.append((curve(*geo["bypass"]), geo["bypass_width_cm"], "inlet_bypass"))
    for t, w, _ in slots:
        h = max(w / lv["slot_cells"], lv["h_slot_min"])
        d = F.add("Distance")
        F.setNumbers(d, "CurvesList", [t])
        F.setNumber(d, "Sampling", 100)
        th = F.add("Threshold")
        F.setNumber(th, "InField", d)
        F.setNumber(th, "SizeMin", h / 100.0)
        F.setNumber(th, "SizeMax", lv["h_bulk"] / 100.0)
        F.setNumber(th, "DistMin", w / 100.0)
        F.setNumber(th, "DistMax", (w + lv["slot_grow"]) / 100.0)
        F.setNumber(th, "StopAtDistMax", 1)
        fields.append(th)
    m = F.add("Min")
    F.setNumbers(m, "FieldsList", fields)
    F.setAsBackgroundMesh(m)
    for k, v in {"Mesh.MeshSizeExtendFromBoundary": 0, "Mesh.MeshSizeFromPoints": 0,
                 "Mesh.MeshSizeFromCurvature": 0, "Mesh.Algorithm": 8, "Mesh.RecombineAll": 1,
                 "Mesh.RecombinationAlgorithm": 1}.items():
        gmsh.option.setNumber(k, v)
    gmsh.model.mesh.generate(2)

    # ---- extract ----------------------------------------------------------------
    ntags, xyz, _ = gmsh.model.mesh.getNodes()
    ntags = np.asarray(ntags, dtype=np.int64)
    lut = np.full(ntags.max() + 1, -1, dtype=np.int64)
    lut[ntags] = np.arange(len(ntags))
    xyz = np.asarray(xyz).reshape(-1, 3)
    RZ = xyz[:, [0, 2]]
    elems, all_etags = [], []
    for s in surfaces:
        types, etags, enodes = gmsh.model.mesh.getElements(2, surf_tag[s["name"]])
        for ty, et, en in zip(types, etags, enodes):
            nn = {2: 3, 3: 4}.get(int(ty))
            if nn is None:
                raise SystemExit(f"unexpected 2D element type {ty}")
            elems.append((s["zone"], lut[np.asarray(en, dtype=np.int64)].reshape(-1, nn)))
            all_etags.append(np.asarray(et))
    qual = np.asarray(gmsh.model.mesh.getElementQualities(np.concatenate(all_etags), "minSICN"))
    patches = {}
    for t, lab in line_label.items():
        if lab == "internal":
            continue
        _, _, enodes = gmsh.model.mesh.getElements(1, t)
        patches.setdefault(lab, []).append(lut[np.asarray(enodes[0], dtype=np.int64)].reshape(-1, 2))
    slot_cells = []
    for t, w, lab in slots:
        _, _, enodes = gmsh.model.mesh.getElements(1, t)
        slot_cells.append((lab, w, len(enodes[0]) // 2))
    gmsh.finalize()
    return dict(RZ=RZ, elems=elems, patches={k: np.vstack(v) for k, v in patches.items()},
                quality=qual, slot_cells=slot_cells, level=level)


def _signed_area(RZ, conn):
    x, y = RZ[conn, 0], RZ[conn, 1]
    return 0.5 * (x * np.roll(y, -1, axis=1) - np.roll(x, -1, axis=1) * y).sum(axis=1)


def _centroid_R(RZ, conn):
    x, y = RZ[conn, 0], RZ[conn, 1]
    x1, y1 = np.roll(x, -1, axis=1), np.roll(y, -1, axis=1)
    cr = x * y1 - x1 * y
    return ((x + x1) * cr).sum(axis=1) / (3.0 * cr.sum(axis=1))


def write_wedge_msh(m2: dict, geo: dict, path: Path) -> dict:
    """Rotate the 2D mesh to +/- wedge/2 about Z, make hexes/prisms, check it, write Gmsh 2.2."""
    th = math.radians(CFG["wedge_deg"])
    a = 0.5 * th
    RZ = m2["RZ"]
    N = len(RZ)
    front = np.column_stack([RZ[:, 0] * math.cos(a), -RZ[:, 0] * math.sin(a), RZ[:, 1]])
    back = np.column_stack([RZ[:, 0] * math.cos(a), RZ[:, 0] * math.sin(a), RZ[:, 1]])
    nodes = np.vstack([front, back])

    # bottom face (front plane) must be clockwise in (R, Z) so its normal points +phi
    elems = []
    for zone, conn in m2["elems"]:
        A = _signed_area(RZ, conn)
        conn = np.where((A > 0)[:, None], conn[:, ::-1], conn)
        elems.append((zone, conn))

    # ---- checks ---------------------------------------------------------------
    report = {}
    for zone, conn in elems:
        bot, top = nodes[conn], nodes[conn + N]
        tp = np.einsum("ij,ij->i", np.cross(bot[:, 1] - bot[:, 0], bot[:, -1] - bot[:, 0]), top[:, 0] - bot[:, 0])
        if (tp <= 0).any():
            raise SystemExit(f"{zone}: {int((tp <= 0).sum())} inverted cells")
    vol_zone = {}
    for zone, conn in elems:
        V = math.sin(th) * np.abs(_signed_area(RZ, conn)) * _centroid_R(RZ, conn)   # exact planar-wedge volume
        vol_zone[zone] = vol_zone.get(zone, 0.0) + float(V.sum())
    k = 360.0 / CFG["wedge_deg"]
    ref = {"bulk": _ring_volume_m3(geo["regions"]["flibe_blanket"])}
    for c in geo["circuits"]:
        ref[f"gap_{c['name']}"] = _ring_volume_m3(c["gap"]) + _ring_volume_m3(c["opening"])
    report["zone_volumes_m3"] = {z: dict(cfd_x360=vol_zone[z] * k, polygon=ref[z]) for z in ref}
    # every element edge must be internal (2 uses) or on exactly one patch edge
    edges = np.vstack([np.sort(np.column_stack([c[:, i], c[:, (i + 1) % c.shape[1]]]), axis=1)
                       for _, c in elems for i in range(c.shape[1])])
    uniq, cnt = np.unique(edges, axis=0, return_counts=True)
    if (cnt > 2).any():
        raise SystemExit("edge shared by more than two cells")
    bnd = {tuple(e) for e in uniq[cnt == 1]}
    pe = {}
    for lab, ed in m2["patches"].items():
        for e in np.sort(ed, axis=1):
            pe[tuple(e)] = lab
    if bnd != set(pe):
        raise SystemExit(f"boundary edges without a patch: {len(bnd - set(pe))}, "
                         f"patch edges that are internal: {len(set(pe) - bnd)}")
    report["n_cells"] = int(sum(len(c) for _, c in elems))
    report["n_hex"] = int(sum(len(c) for _, c in elems if c.shape[1] == 4))
    report["n_prism"] = report["n_cells"] - report["n_hex"]
    report["min_quality_minSICN"] = float(m2["quality"].min())
    report["q01_quality_minSICN"] = float(np.percentile(m2["quality"], 1))
    for zone, conn in elems:
        if zone.startswith("gap_"):
            h = np.sqrt(np.abs(_signed_area(RZ, conn)))
            report.setdefault("gap_cell_size_cm", []).extend((100 * h).tolist())
    hg = np.asarray(report.pop("gap_cell_size_cm"))
    report["gap_cell_size_cm_median"] = float(np.median(hg))
    report["cells_across_gap_est"] = float((geo["depths"]["gap"] - geo["depths"]["v3"]) / np.median(hg))
    report["slot_cells"] = m2["slot_cells"]

    # ---- write Gmsh 2.2 ---------------------------------------------------------
    patch_names = ["front", "back"] + sorted(m2["patches"])
    zones = sorted({z for z, _ in elems})
    ptag = {nm: i + 1 for i, nm in enumerate(patch_names)}
    ztag = {nm: 1001 + i for i, nm in enumerate(zones)}
    out = ["$MeshFormat\n2.2 0 8\n$EndMeshFormat\n$PhysicalNames\n",
           f"{len(patch_names) + len(zones)}\n"]
    out += [f'2 {ptag[nm]} "{nm}"\n' for nm in patch_names]
    out += [f'3 {ztag[nm]} "{nm}"\n' for nm in zones]
    out.append(f"$EndPhysicalNames\n$Nodes\n{2 * N}\n")
    ids = np.arange(1, 2 * N + 1)
    out.append("".join(f"{i} {x:.12g} {y:.12g} {z:.12g}\n" for i, (x, y, z) in zip(ids, nodes)))
    out.append("$EndNodes\n")
    rows = []

    def add(etype, tag, conn1):                  # conn1: 1-based node ids
        for r in conn1:
            rows.append(f"{etype} 2 {tag} {tag} " + " ".join(map(str, r)))

    for zone, conn in elems:
        etype = 5 if conn.shape[1] == 4 else 6
        add(etype, ztag[zone], np.hstack([conn + 1, conn + N + 1]))
    for zone, conn in elems:
        etype = 3 if conn.shape[1] == 4 else 2
        add(etype, ptag["front"], conn + 1)
        add(etype, ptag["back"], conn + N + 1)
    for lab, ed in m2["patches"].items():
        add(3, ptag[lab], np.column_stack([ed[:, 0] + 1, ed[:, 1] + 1, ed[:, 1] + N + 1, ed[:, 0] + N + 1]))
    out.append(f"$Elements\n{len(rows)}\n")
    out.append("".join(f"{i + 1} {r}\n" for i, r in enumerate(rows)))
    out.append("$EndElements\n")
    path.write_text("".join(out))
    report["patches"] = patch_names
    report["zones"] = zones
    return report


def print_mesh_report(rep: dict):
    print(f"\nMesh: {rep['n_cells']} cells ({rep['n_hex']} hex, {rep['n_prism']} prism), "
          f"quality minSICN min {rep['min_quality_minSICN']:.3f}, 1st pct {rep['q01_quality_minSICN']:.3f}")
    print(f"  gap cells: median size {rep['gap_cell_size_cm_median']:.2f} cm -> "
          f"~{rep['cells_across_gap_est']:.1f} cells across the gap")
    for z, v in rep["zone_volumes_m3"].items():
        print(f"  zone {z:14s} CFD x360/wedge {v['cfd_x360']:9.4f} m3  polygon {v['polygon']:9.4f} m3  "
              f"ratio {v['cfd_x360'] / v['polygon']:.6f}")
    print("  cells across slots / openings:")
    for lab, w, n in rep["slot_cells"]:
        print(f"    {lab:30s} {w:5.2f} cm  {n:3d} cells")


# =============================================================================
# OpenFOAM case
# =============================================================================
def patch_table(geo, loads, fl) -> dict:
    """Every boundary patch with its role and full-machine flow (kg/s) or heat (W)."""
    t, per = loads["totals"], loads["per_circuit"]
    P = {"front": dict(role="wedge"), "back": dict(role="wedge")}
    for c in geo["circuits"]:
        P[f"inlet_{c['name']}"] = dict(role="inlet", mdot_full=fl["circuits"][c["name"]]["mdot"],
                                       width_m=c["inlet_width_cm"] / 100.0)
    if fl["m_bypass"] > 0:
        P["inlet_bypass"] = dict(role="inlet", mdot_full=fl["m_bypass"], width_m=geo["bypass_width_cm"] / 100.0)
    P["outlet"] = dict(role="outlet")
    for c in geo["circuits"]:
        L, F = per[c["name"]], fl["circuits"][c["name"]]
        P[f"fw_{c['name']}"] = dict(role="wallHeat", Q_full=L["W"] + L["V3"] + F["P_surf"])
        P[f"vvGap_{c['name']}"] = dict(role="wallHeat", Q_full=0.5 * L["V5"])
    P["vvBulk"] = dict(role="wallHeat", Q_full=0.5 * t["V5"])
    P["shield"] = dict(role="wallHeat", Q_full=t["WC"])
    P["tank"] = dict(role="wallHeat", Q_full=t["SS"])
    P["baffles"] = dict(role="wallAdiabatic")
    P["openingSides"] = dict(role="wallAdiabatic")
    k = CFG["wedge_deg"] / 360.0
    for p in P.values():
        if "Q_full" in p:
            p["Q"] = p["Q_full"] * k
        if "mdot_full" in p:
            p["mdot"] = p["mdot_full"] * k
    walls = sum(p["Q_full"] for p in P.values() if "Q_full" in p)
    vol = t["flibe_gap"] + t["flibe_bulk"]
    if abs(walls + vol - fl["Q_total"]) > 1e-9 * fl["Q_total"]:
        raise SystemExit("wall + volumetric heat does not add up to the total")
    return P


def _render_field(cls, name, dims, internal, bcs, note=""):
    s = foam_io.header(cls, name, "0")
    if note:
        s += "// " + note.replace("\n", "\n// ") + "\n\n"
    s += f"dimensions      {dims};\n\ninternalField   {internal};\n\nboundaryField\n{{\n"
    for p, lines in bcs.items():
        s += f"    {p}\n    {{\n" + "".join(f"        {ln}\n" for ln in lines) + "    }\n"
    return s + "}\n"


def write_fields(case: Path, P: dict):
    T_in, rho_in = CFG["T_in"], rho_flibe(CFG["T_in"])
    k0, e0, T0, pout = CFG["k_init"], CFG["epsilon_init"], CFG["T_init"], CFG["p_out"]

    def make(rule):
        return {p: (["type            wedge;"] if info["role"] == "wedge" else rule(p, info)) for p, info in P.items()}

    def U(p, i):
        if i["role"] == "inlet":
            return ["type            flowRateInletVelocity;",
                    f"massFlowRate    constant {i['mdot']:.10g};   // kg/s in the wedge; "
                    f"{i['mdot_full']:.1f} kg/s full machine",
                    "rho             rho;", f"rhoInlet        {rho_in:.6g};", "value           uniform (0 0 0);"]
        if i["role"] == "outlet":
            return ["type            pressureInletOutletVelocity;", "value           uniform (0 0 0);"]
        return ["type            noSlip;"]

    def T(p, i):
        if i["role"] == "inlet":
            return ["type            fixedValue;", f"value           uniform {T_in};"]
        if i["role"] == "outlet":
            return ["type            inletOutlet;", f"inletValue      uniform {CFG['T_backflow']};",
                    f"value           uniform {CFG['T_backflow']};"]
        if i["role"] == "wallHeat":
            return ["type            externalWallHeatFluxTemperature;", "mode            power;",
                    f"Q               constant {i['Q']:.10g};   // W in the wedge; "
                    f"{i['Q_full'] / 1e6:.4f} MW full machine",
                    "kappaMethod     fluidThermo;", f"value           uniform {T0};"]
        return ["type            zeroGradient;"]

    def prgh(p, i):
        if i["role"] == "outlet":
            return ["type            prghPressure;", "rho             rho;", f"p               uniform {pout};",
                    f"value           uniform {pout};"]
        return ["type            fixedFluxPressure;", f"value           uniform {pout};"]

    def p_(p, i):
        return ["type            calculated;", f"value           uniform {pout};"]

    def kf(p, i):
        if i["role"] == "inlet":
            return ["type            turbulentIntensityKineticEnergyInlet;",
                    f"intensity       {CFG['inlet_intensity']};", f"value           uniform {k0};"]
        if i["role"] == "outlet":
            return ["type            inletOutlet;", f"inletValue      uniform {k0};", f"value           uniform {k0};"]
        return ["type            kqRWallFunction;", f"value           uniform {k0};"]

    def ef(p, i):
        if i["role"] == "inlet":
            L = CFG["mixing_length_factor"] * 2.0 * i["width_m"]
            return ["type            turbulentMixingLengthDissipationRateInlet;",
                    f"mixingLength    {L:.6g};", f"value           uniform {e0};"]
        if i["role"] == "outlet":
            return ["type            inletOutlet;", f"inletValue      uniform {e0};", f"value           uniform {e0};"]
        return ["type            epsilonWallFunction;", f"value           uniform {e0};"]

    def nut(p, i):
        if i["role"].startswith("wall"):
            return [f"type            {CFG['nut_wall']};", "value           uniform 0;"]
        return ["type            calculated;", "value           uniform 0;"]

    def alphat(p, i):
        if i["role"].startswith("wall"):
            return ["type            compressible::alphatJayatillekeWallFunction;",
                    f"Prt             {CFG['Prt']};", "value           uniform 0;"]
        return ["type            calculated;", "value           uniform 0;"]

    def scal(p, i):
        if i["role"] == "inlet":
            return ["type            fixedValue;", "value           uniform 0;"]
        if i["role"] == "outlet":
            return ["type            inletOutlet;", "inletValue      uniform 0;", "value           uniform 0;"]
        return ["type            zeroGradient;"]

    z = case / "0"
    z.mkdir(parents=True, exist_ok=True)
    (z / "U").write_text(_render_field("volVectorField", "U", "[0 1 -1 0 0 0 0]", "uniform (0 0 0)", make(U)))
    (z / "T").write_text(_render_field("volScalarField", "T", "[0 0 0 1 0 0 0]", f"uniform {T0}", make(T)))
    (z / "p_rgh").write_text(_render_field("volScalarField", "p_rgh", "[1 -1 -2 0 0 0 0]", f"uniform {pout}", make(prgh)))
    (z / "p").write_text(_render_field("volScalarField", "p", "[1 -1 -2 0 0 0 0]", f"uniform {pout}", make(p_)))
    (z / "k").write_text(_render_field("volScalarField", "k", "[0 2 -2 0 0 0 0]", f"uniform {k0}", make(kf)))
    (z / "epsilon").write_text(_render_field("volScalarField", "epsilon", "[0 2 -3 0 0 0 0]", f"uniform {e0}", make(ef)))
    (z / "nut").write_text(_render_field("volScalarField", "nut", "[0 2 -1 0 0 0 0]", "uniform 0", make(nut)))
    (z / "alphat").write_text(_render_field("volScalarField", "alphat", "[1 -1 -1 0 0 0 0]", "uniform 0", make(alphat)))
    for iso in CFG["isotopes"]:
        (z / f"C_{iso}").write_text(_render_field(
            "volScalarField", f"C_{iso}", "[-1 0 0 0 0 0 0]", "uniform 0", make(scal),
            note=f"{iso} specific concentration (atoms per kg of FLiBe). Solved on the frozen flow by the\n"
                 f"scalarTransport function object in system/scalarsDict (./Allrun scalars)."))


_CODED = r"""            // Explicit volumetric source read once from 0/<sourceField>, plus an
            // implicit decay sink -lambda*rho*psi (phi is a mass flux, so psi is per kg).
            const word srcName(coeffs_.get<word>("sourceField"));
            const scalar lambda(coeffs_.get<scalar>("lambda"));
            if (!mesh_.foundObject<volScalarField>(srcName))
            {
                regIOobject::store
                (
                    new volScalarField
                    (
                        IOobject
                        (
                            srcName, fileName("0"), mesh_,
                            IOobject::MUST_READ, IOobject::NO_WRITE
                        ),
                        mesh_
                    )
                );
            }
            const volScalarField& S = mesh_.lookupObject<volScalarField>(srcName);
            eqn += S();
            if (lambda > 0)
            {
                eqn -= fvm::Sp(dimensionedScalar(dimless/dimTime, lambda)*rho, eqn.psi());
            }"""


def _coded_source(entry: str, field: str, source: str, lam: float, indent: str = "") -> str:
    body = (f"{entry}\n{{\n    type            scalarCodedSource;\n    selectionMode   all;\n"
            f"    fields          ({field});\n    name            flibeVolumeSource;\n"
            f"    sourceField     {source};\n    lambda          {lam:.10g};   // 1/s\n"
            "    codeInclude\n    #{\n        #include \"fvm.H\"\n    #};\n"
            "    codeAddSupRho\n    #{\n" + _CODED + "\n    #};\n"
            # v2606 CodedFvSource::read() makes codeCorrect and codeConstrain mandatory
            "    codeCorrect\n    #{\n    #};\n"
            "    codeConstrain\n    #{\n    #};\n}\n")
    return "".join(indent + ln + "\n" if ln else "\n" for ln in body.splitlines())


def write_case(case: Path, geo, loads, fl, P, mu_tab, decay, level, mesh_rep):
    (case / "constant").mkdir(parents=True, exist_ok=True)
    (case / "system").mkdir(parents=True, exist_ok=True)
    write_fields(case, P)
    H = foam_io.header
    A_mu, B_mu = CFG["mu_arrhenius"]
    mu_pts = mu_tab["points"]
    th = 1.0 / CFG["Sct"]
    c8 = lambda c: "(" + " ".join(f"{x:.12g}" for x in (list(c) + [0.0] * 8)[:8]) + ")"
    # ---- constant ----
    (case / "constant" / "g").write_text(
        H("uniformDimensionedVectorField", "g", "constant") +
        "dimensions      [0 1 -2 0 0 0 0];\nvalue           (0 0 -9.81);\n")
    (case / "constant" / "thermophysicalProperties").write_text(
        H("dictionary", "thermophysicalProperties", "constant") +
        f"// FLiBe, Romatoski & Hu 2017 (T in K). mu: table of {A_mu:g} exp({B_mu:g}/T), {mu_tab['n']} points,\n"
        f"// {mu_pts[0][0]:g}-{mu_pts[-1][0]:g} K; linear-interpolation error {100 * mu_tab['max_err_700_1400K']:.3f} % "
        f"(700-1400 K), {100 * mu_tab['max_err_all']:.3f} % (whole table).\n"
        "// The table must cover every face T: v2606 nonUniformTable reads past its end above the last point.\n\n"
        "thermoType\n{\n    type            heRhoThermo;\n    mixture         pureMixture;\n"
        "    transport       tabulated;\n    thermo          hPolynomial;\n"
        "    equationOfState icoPolynomial;\n    specie          specie;\n    energy          sensibleEnthalpy;\n}\n\n"
        "mixture\n{\n    specie\n    {\n        molWeight       " + f"{CFG['molWeight']};\n    }}\n"
        "    equationOfState\n    {\n        rhoCoeffs<8>    " + c8(CFG["rho_coeffs"]) + ";\n    }\n"
        "    thermodynamics\n    {\n        Hf              0;\n        Sf              0;\n"
        "        CpCoeffs<8>     " + c8([CFG["cp"]]) + ";\n    }\n"
        "    transport\n    {\n        mu\n        (\n"
        + "".join(f"            ({T_:.10g} {m_:.10g})\n" for T_, m_ in mu_pts) +
        "        );\n        kappa\n        (\n"
        f"            ({mu_pts[0][0]:g} {CFG['kappa']:g})\n            ({mu_pts[-1][0]:g} {CFG['kappa']:g})\n"
        "        );\n    }\n}\n")
    (case / "constant" / "turbulenceProperties").write_text(
        H("dictionary", "turbulenceProperties", "constant") +
        "simulationType  RAS;\n\nRAS\n{\n" + f"    RASModel        {CFG['turbulence_model']};\n"
        "    turbulence      on;\n    printCoeffs     on;\n}\n")
    (case / "constant" / "fvOptions").write_text(
        H("dictionary", "fvOptions", "constant") +
        "// OpenMC nuclear heating in the FLiBe (W/m3), mapped conservatively by make_case.py map\n\n" +
        _coded_source("flibeHeating", "h", "qFlibe", 0.0) +
        "\n// Startup safeguard (numerics only): clip T while the flow develops from rest.\n"
        "// Must clip no cells at convergence (post_cfd.py checks postProcessing/limitT).\n"
        f"limitT\n{{\n    type            limitTemperature;\n    selectionMode   all;\n"
        f"    min             {CFG['T_limits'][0]:g};\n    max             {CFG['T_limits'][1]:g};\n"
        "    writeToFile     true;\n}\n")
    # ---- system ----
    n_end, n_wr = CFG["end_iterations"][level], CFG["write_interval"][level]
    fo = ("functions\n{\n"
          "    solverInfo\n    {\n        type            solverInfo;\n        libs            (utilityFunctionObjects);\n"
          "        fields          (U p_rgh h k epsilon);\n    }\n"
          "    yPlus\n    {\n        type            yPlus;\n        libs            (fieldFunctionObjects);\n"
          "        writeControl    writeTime;\n    }\n"
          "    outletFlow\n    {\n        type            surfaceFieldValue;\n        libs            (fieldFunctionObjects);\n"
          "        regionType      patch;\n        name            outlet;\n        operation       sum;\n"
          "        fields          (phi);\n        writeFields     false;\n        writeControl    timeStep;\n"
          "        writeInterval   10;\n    }\n"
          "    outletT\n    {\n        type            surfaceFieldValue;\n        libs            (fieldFunctionObjects);\n"
          "        regionType      patch;\n        name            outlet;\n        operation       weightedAverage;\n"
          "        weightField     phi;\n        fields          (T);\n        writeFields     false;\n"
          "        writeControl    timeStep;\n        writeInterval   10;\n    }\n"
          "    TminMax\n    {\n        type            fieldMinMax;\n        libs            (fieldFunctionObjects);\n"
          "        fields          (T U);\n        writeControl    timeStep;\n        writeInterval   10;\n    }\n"
          # Mean age on the final flow field. It must run inside the flow run: in
          # 'solver -postProcess' the age FO is built before phi exists, fails its phi
          # lookup and is silently dropped (v2606 age::read). onEnd also fires when
          # residualControl stops the run early (writeAndEnd moves endTime). Written only
          # if the final iteration is a write time, so it never makes a time folder that
          # lacks the flow fields.
          "    age\n    {\n        type            age;\n        libs            (fieldFunctionObjects);\n"
          "        phi             phi;\n        rho             rho;\n"
          f"        nCorr           {CFG['scalar_nCorr']};\n        tolerance       {CFG['scalar_tolerance']};\n"
          f"        diffusion       {'on' if CFG['age_diffusion'] else 'off'};   // muEff = mu + mut\n"
          "        executeControl  onEnd;\n        writeControl    writeTime;\n    }\n}\n")
    (case / "system" / "controlDict").write_text(
        H("dictionary", "controlDict", "system") +
        "application     buoyantSimpleFoam;\nstartFrom       latestTime;\nstartTime       0;\n"
        f"stopAt          endTime;\nendTime         {n_end};\ndeltaT          1;\n"
        f"writeControl    timeStep;\nwriteInterval   {n_wr};\npurgeWrite      2;\nwriteFormat     ascii;\n"
        "writePrecision  12;\nwriteCompression off;\ntimeFormat      general;\ntimePrecision   6;\n"
        "runTimeModifiable true;\n\n" + fo)
    div_sc = "".join(f"    div(phi,C_{iso}) bounded Gauss limitedLinear 1;\n" for iso in CFG["isotopes"])
    (case / "system" / "fvSchemes").write_text(
        H("dictionary", "fvSchemes", "system") +
        "ddtSchemes\n{\n    default         steadyState;\n}\n\n"
        "gradSchemes\n{\n    default         Gauss linear;\n    grad(U)         cellLimited Gauss linear 1;\n}\n\n"
        "divSchemes\n{\n    default         none;\n    div(phi,U)      bounded Gauss linearUpwind grad(U);\n"
        "    div(phi,h)      bounded Gauss limitedLinear 1;\n    div(phi,K)      bounded Gauss limitedLinear 1;\n"
        "    div(phi,k)      bounded Gauss upwind;\n    div(phi,epsilon) bounded Gauss upwind;\n"
        "    div(((rho*nuEff)*dev2(T(grad(U))))) Gauss linear;\n" + div_sc +
        "    div(phi,age)    bounded Gauss limitedLinear 1;\n}\n\n"
        "laplacianSchemes\n{\n    default         Gauss linear limited corrected 0.33;\n}\n\n"
        "interpolationSchemes\n{\n    default         linear;\n}\n\n"
        "snGradSchemes\n{\n    default         limited corrected 0.33;\n}\n\n"
        "wallDist\n{\n    method          meshWave;\n}\n")
    (case / "system" / "fvSolution").write_text(
        H("dictionary", "fvSolution", "system") +
        "solvers\n{\n    p_rgh\n    {\n        solver          GAMG;\n        smoother        DICGaussSeidel;\n"
        "        tolerance       1e-8;\n        relTol          0.01;\n    }\n"
        "    \"(U|h|k|epsilon)\"\n    {\n        solver          PBiCGStab;\n        preconditioner  DILU;\n"
        "        tolerance       1e-10;\n        relTol          0.1;\n    }\n"
        "    \"(C_.*|age)\"\n    {\n        solver          PBiCGStab;\n        preconditioner  DILU;\n"
        "        tolerance       1e-14;\n        relTol          1e-6;\n        maxIter         20000;\n    }\n}\n\n"
        "SIMPLE\n{\n    momentumPredictor yes;\n    nNonOrthogonalCorrectors 0;\n    pRefCell        0;\n"
        f"    pRefValue       {CFG['p_out']};\n"
        f"    rhoMin          {rho_flibe(CFG['T_limits'][1]):.6g};   // rho at T_limits (startup safeguard)\n"
        f"    rhoMax          {rho_flibe(CFG['T_limits'][0]):.6g};\n"
        "    residualControl\n    {\n        p_rgh           1e-5;\n"
        "        U               1e-6;\n        h               1e-7;\n        \"(k|epsilon)\"   1e-5;\n    }\n}\n\n"
        "relaxationFactors\n{\n    fields\n    {\n        rho             1.0;\n        p_rgh           0.7;\n    }\n"
        "    equations\n    {\n        U               0.3;\n        h               0.3;\n"
        "        \"(k|epsilon)\"   0.5;\n        \"C_.*\"           1;\n        age             1;\n    }\n}\n")
    (case / "system" / "decomposeParDict").write_text(
        H("dictionary", "decomposeParDict", "system") +
        f"numberOfSubdomains {max(CFG['n_procs'][level], 1)};\n\nmethod          scotch;\n")
    bnd = "".join(f"    {p}\n    {{\n        type            {'wedge' if i['role'] == 'wedge' else ('wall' if i['role'].startswith('wall') else 'patch')};\n"
                  + ("        inGroups        (wall);\n" if i["role"].startswith("wall") else "") + "    }\n"
                  for p, i in P.items())
    (case / "system" / "changeDictionaryDict").write_text(
        H("dictionary", "changeDictionaryDict", "system") + "boundary\n{\n" + bnd + "}\n")
    sc = ""
    for iso in CFG["isotopes"]:
        lam = decay[iso]["lam"]
        sc += (f"    C_{iso}\n    {{\n        // {iso}: half-life {decay[iso]['half_life_s']:.6g} s (openmc.data)\n"
               "        type            scalarTransport;\n        libs            (solverFunctionObjects);\n"
               f"        field           C_{iso};\n        phi             phi;\n        rho             rho;\n"
               f"        alphaD          0;            // molecular diffusion neglected\n"
               f"        alphaDt         {th:.10g};   // 1/Sct, Sct = {CFG['Sct']}: D = mut/Sct\n"
               f"        nCorr           {CFG['scalar_nCorr']};\n        tolerance       {CFG['scalar_tolerance']};\n"
               "        bounded01       false;\n        resetOnStartUp  false;\n        fvOptions\n        {\n"
               + _coded_source(f"source_{iso}", f"C_{iso}", f"S_{iso}", lam, indent="            ")
               + "        }\n    }\n")
    (case / "system" / "scalarsDict").write_text(
        H("dictionary", "scalarsDict", "system") +
        "// Frozen-flow isotope transport, run once on the converged flow (mean age is\n"
        "// computed at the end of the flow run: 'age' in controlDict):\n"
        "//   buoyantSimpleFoam -postProcess -dict system/scalarsDict -time <latest>\n"
        "// Each isotope: div(phi C) - div((mut/Sct) grad C) = S - lambda rho C, C in atoms/kg.\n\n"
        "functions\n{\n" + sc + "}\n")
    write_scripts(case, level)
    summary = dict(level=level, wedge_deg=CFG["wedge_deg"], T_in=CFG["T_in"], cp=CFG["cp"],
                   rho_coeffs=CFG["rho_coeffs"], mu_table={k: v for k, v in mu_tab.items() if k != "points"}, T_limits=list(CFG["T_limits"]), decay=decay, Sct=CFG["Sct"],
                   openmc_run=str(CFG["openmc_run"]), patches=P,
                   heat_W=dict(total=fl["Q_total"], **{k: v for k, v in loads["totals"].items()},
                               surface=1e6 * (CFG["P_alpha_MW"] + CFG["P_aux_MW"])),
                   flows=dict(m_total=fl["m_total"], m_bypass=fl["m_bypass"], T_out_target=fl["T_out"],
                              circuits=fl["circuits"]),
                   mesh=mesh_rep, isotopes=list(CFG["isotopes"]))
    (case / "case_summary.json").write_text(json.dumps(summary, indent=1, default=float))


# Progress lines for ./Allrun flow / scalars. The solver still writes only to its log;
# a watcher reads that log every PROGRESS_EVERY seconds (default 10) and prints one line.
_ALLRUN_PROGRESS = r"""
# ---- progress: one status line every PROGRESS_EVERY s (default 10) from the log ----
status() {  # status <log> [endTime] [final]: last complete iteration (flow) or scalar solve
    { if [ -n "${2:-}" ]; then tail -n 600 "$1"; else cat "$1"; fi; } 2>/dev/null | awk -v end="${2:-}" -v final="${3:-}" '
        function hms(x) { x = int(x); return (x >= 3600 ? int(x / 3600) "h" : "") \
                          sprintf("%dm%02ds", int(x % 3600 / 60), x % 60) }
        function snap(f) { di = it; dX = xt; dT = tmax; dC = clip; dO = tout
                           split("", d); for (f in r) d[f] = r[f]
                           if (fi == "") { fi = di; fx = dX } }
        /^Time = / { if (it != "") snap(); it = $3 + 0; split("", r); clip = 0 }
        / Solving for / { f = $4; sub(/,$/, "", f); v = $8; sub(/,$/, "", v)
                          if (!(f in r)) r[f] = v + 0; nc = (f == sf) ? nc + 1 : 1; sf = f; sv = v + 0 }
        /^limitTemperature=/ { split($0, a, "LimitedCells="); clip += a[2] + 0
                               if (/Type=Upper/) { split($0, b, "UnlimitedTmax="); tmax = b[2] + 0 } }
        /^ExecutionTime = / { xt = $3 + 0 }
        /weightedAverage\(outlet\) of T/ { tout = $NF }
        END {
            if (final != "" && it != "") snap()
            if (end == "") { if (sf != "") printf "  scalars: %-6s outer corrector %3d, residual %.2e\n", sf, nc, sv; exit }
            if (di == "") exit
            u = d["Ux"]; if (d["Uy"] > u) u = d["Uy"]; if (d["Uz"] > u) u = d["Uz"]
            left = (di > fi && dX > fx) ? hms((end - di) * (dX - fx) / (di - fi)) : "?"
            printf "  iter %5d/%d (%3d%%)  res p_rgh %.1e U %.1e h %.1e k %.1e eps %.1e | cell Tmax %.0f K, clipped %d | T out %.1f K | %s elapsed, ~%s left\n",
                   di, end, 100 * di / end, d["p_rgh"], u, d["h"], d["k"], d["epsilon"], dT, dC, dO, hms(dX), left
        }'
}
progress() {  # progress <log> [endTime] &   (killed by the caller when the solver ends)
    local prev="" line
    while sleep "${PROGRESS_EVERY:-10}"; do
        line=$(status "$@")
        [ -n "$line" ] && [ "$line" != "$prev" ] && echo "$line" && prev=$line
    done
}
watch_start() { progress "$@" 2>/dev/null & WATCH_PID=$!; trap 'pkill -P $WATCH_PID 2>/dev/null; kill $WATCH_PID 2>/dev/null' EXIT; }
watch_stop() { pkill -P "$WATCH_PID" 2>/dev/null; kill "$WATCH_PID" 2>/dev/null; wait "$WATCH_PID" 2>/dev/null; trap - EXIT; }
"""

def write_scripts(case: Path, level: str):
    np_ = max(CFG["n_procs"][level], 1)
    allrun = f"""#!/bin/bash
#------------------------------------------------------------------------------
# ARC V3A FLiBe blanket - OpenFOAM steps (generated by cfd/make_case.py)
#   ./Allrun mesh      gmshToFoam, patch types, checkMesh, cell centres/volumes
#   (Python env)       python cfd/make_case.py map
#   ./Allrun flow      buoyantSimpleFoam (parallel when NP > 1); mean age at the end
#   ./Allrun scalars   frozen-flow isotope transport, latest time
#   ./Allrun solve     flow, then scalars
# Run in a shell where conda / the Python venv is NOT active.
#------------------------------------------------------------------------------
cd "${{0%/*}}" || exit 1
NP={np_}
OF_BASHRC={CFG['openfoam_bashrc']}

if [ -n "${{CONDA_PREFIX:-}}${{VIRTUAL_ENV:-}}" ]; then
    echo "Deactivate conda / the Python venv before running the OpenFOAM steps." >&2
    exit 1
fi
if [ -z "${{WM_PROJECT_DIR:-}}" ]; then
    . "$OF_BASHRC" || {{ echo "Cannot source $OF_BASHRC" >&2; exit 1; }}
fi
. "${{WM_PROJECT_DIR:?}}"/bin/tools/RunFunctions

# latest time folder that holds a saved flow field (phi), not just a function-object field
latestTime() {{ local t last=""; for t in $(foamListTimes -noZero 2>/dev/null); do [ -f "$t/phi" ] && last=$t; done; echo "$last"; }}
# on failure, show the end of the newest log (runApplication sends all output there)
fail() {{ log=$(ls -t log.* 2>/dev/null | head -n 1); echo "FAILED - last lines of $log:" >&2;
          [ -n "$log" ] && tail -n 30 "$log" >&2; exit 1; }}
{_ALLRUN_PROGRESS}
do_mesh() {{
    runApplication -o gmshToFoam v3a.msh || fail
    runApplication -o changeDictionary || fail
    runApplication -o checkMesh || fail
    grep -q "Mesh OK" log.checkMesh || {{ echo "checkMesh FAILED - see log.checkMesh" >&2; exit 1; }}
    runApplication -o -s cellCentres postProcess -func writeCellCentres -time 0 || fail
    runApplication -o -s cellVolumes postProcess -func writeCellVolumes -time 0 || fail
    echo "Mesh done. Next, in the Python env: python cfd/make_case.py map"
}}

do_flow() {{
    [ -f 0/qFlibe ] || {{ echo "0/qFlibe missing: run 'python cfd/make_case.py map' first" >&2; exit 1; }}
    end=$(foamDictionary system/controlDict -entry endTime -value)
    wi=$(foamDictionary system/controlDict -entry writeInterval -value)
    awk -v e="$end" -v w="$wi" 'BEGIN {{ exit !(e % w != 0) }}' && echo "  WARNING: endTime $end is not a multiple of writeInterval $wi:" \
        "the last iterations (and the mean age) will not be saved" >&2
    echo "  (progress every ${{PROGRESS_EVERY:-10}} s; full output in log.buoyantSimpleFoam)"
    [ "$NP" -gt 1 ] && {{ runApplication -o decomposePar -force || fail; }}
    watch_start log.buoyantSimpleFoam "$end"
    if [ "$NP" -gt 1 ]; then
        runParallel -o -np "$NP" buoyantSimpleFoam || {{ watch_stop; fail; }}
    else
        runApplication -o buoyantSimpleFoam || {{ watch_stop; fail; }}
    fi
    watch_stop
    status log.buoyantSimpleFoam "$end" final
    if grep -q "solution converged" log.buoyantSimpleFoam; then
        echo "  flow converged: $(grep -m1 'solution converged' log.buoyantSimpleFoam)"
    else
        echo "  flow stopped at endTime $end without meeting the residualControl targets"
    fi
    [ "$NP" -gt 1 ] && {{ runApplication -o reconstructPar -latestTime || fail; }}
    t=$(latestTime)
    if [ -f "$t/age" ]; then echo "  mean age written to $t/age"
    else echo "  WARNING: no $t/age - see the 'age' lines in log.buoyantSimpleFoam" >&2; fi
}}

do_scalars() {{
    t=$(latestTime)
    [ -n "$t" ] || {{ echo "no solution time directory found" >&2; exit 1; }}
    for f in 0/C_*; do cp "$f" "$t"/; done
    echo "  (progress every ${{PROGRESS_EVERY:-10}} s; full output in log.buoyantSimpleFoam.scalars)"
    watch_start log.buoyantSimpleFoam.scalars
    runApplication -o -s scalars buoyantSimpleFoam -postProcess -dict system/scalarsDict -time "$t" || {{ watch_stop; fail; }}
    watch_stop
    echo "Scalars done (time $t). Next, in the Python env: python cfd/post_cfd.py"
}}

case "${{1:-}}" in
    mesh)    do_mesh ;;
    flow)    do_flow ;;
    scalars) do_scalars ;;
    solve)   do_flow; do_scalars ;;
    *)       sed -n '3,8p' "$0"; exit 1 ;;
esac
"""
    allclean = """#!/bin/bash
# Remove OpenFOAM outputs; keeps v3a.msh, 0/ base fields, constant/ and system/.
cd "${0%/*}" || exit 1
rm -rf processor* postProcessing dynamicCode constant/polyMesh log.* ./*.foam
for d in */; do
    d=${d%/}
    case "$d" in 0|constant|system) ;; *) [[ "$d" =~ ^[0-9.eE+-]+$ ]] && rm -rf "$d" ;; esac
done
rm -f 0/C 0/Cx 0/Cy 0/Cz 0/V 0/qFlibe 0/S_*
"""
    for name, txt in (("Allrun", allrun), ("Allclean", allclean)):
        f = case / name
        f.write_text(txt)
        os.chmod(f, 0o755)


# =============================================================================
# Source mapping (OpenMC mesh -> FLiBe cells), conservative
# =============================================================================
def map_sources(case: Path) -> dict:
    from scipy.spatial import cKDTree

    summ = json.loads((case / "case_summary.json").read_text())
    om = load_openmc()
    geo = build_geometry()
    f, cells = om["f"], om["cells"]
    Cc = foam_io.read_field(case / "0" / "C")["internal"]
    Vc = np.asarray(foam_io.read_field(case / "0" / "V")["internal"], dtype=float)
    zones = foam_io.read_cell_zones(case / "constant" / "polyMesh" / "cellZones")
    bnd = foam_io.read_boundary(case / "constant" / "polyMesh" / "boundary")
    k = 360.0 / summ["wedge_deg"]
    nC = len(Vc)
    pool = np.full(nC, -1)
    for z, lab in zones.items():
        pool[lab] = 1 if z == "bulk" else 0
    if (pool < 0).any():
        raise SystemExit(f"{int((pool < 0).sum())} cells are in no zone")
    R, Z = np.hypot(Cc[:, 0], Cc[:, 1]), Cc[:, 2]
    re_, ze = f["r_edges_cm"] / 100.0, f["z_edges_cm"] / 100.0
    rc, zc = 0.5 * (re_[1:] + re_[:-1]), 0.5 * (ze[1:] + ze[:-1])
    nr, nz = len(rc), len(zc)
    ib, jb = np.searchsorted(re_, R) - 1, np.searchsorted(ze, Z) - 1
    inmesh = (ib >= 0) & (ib < nr) & (jb >= 0) & (jb < nz)
    flat = np.where(inmesh, ib * nz + jb, -1)
    fields = {"qFlibe": f["heating_flibe_W_m3"]}
    for iso in summ["isotopes"]:
        fields[f"S_{iso}"] = f[f"{iso}_generation_per_m3_s"]
    anyF = np.zeros((nr, nz), bool)
    for F_ in fields.values():
        anyF |= F_ > 0
    i, j, lab, cnt = classify_bins(f, anyF, {"gap": geo["regions"]["vv_cooling_channel"],
                                             "bulk": geo["regions"]["flibe_blanket"]})
    bin_pool = np.full(nr * nz, -1)
    bin_pool[i * nz + j] = lab
    Vring = f["bin_ring_volume_m3"].ravel()
    out, report = {}, {}
    for name, F_ in fields.items():
        T = (F_ * f["bin_ring_volume_m3"]).ravel()          # full-machine total per bin
        assigned = np.zeros(nC)
        fallback = 0.0
        for p in (0, 1):
            cp_ = np.nonzero((pool == p) & (flat >= 0))[0]
            vbin = np.bincount(flat[cp_], weights=Vc[cp_], minlength=nr * nz)
            bins = np.nonzero((bin_pool == p) & (T > 0))[0]
            has = vbin[bins] > 0
            # bins holding cell centres of this pool: share by cell volume
            share = np.zeros(nr * nz)
            share[bins[has]] = T[bins[has]] / vbin[bins[has]]
            assigned[cp_] += share[flat[cp_]] * Vc[cp_]
            # bins with no cell centre: nearest cell of the same pool
            lost = bins[~has]
            if lost.size:
                allp = np.nonzero(pool == p)[0]
                tree = cKDTree(np.column_stack([R[allp], Z[allp]]))
                _, nn = tree.query(np.column_stack([rc[lost // nz], zc[lost % nz]]))
                np.add.at(assigned, allp[nn], T[lost])
                fallback += float(T[lost].sum())
        out[name] = assigned / (Vc * k)
        mesh_tot = float(T[bin_pool >= 0].sum())
        report[name] = dict(gap=float(assigned[pool == 0].sum()), bulk=float(assigned[pool == 1].sum()),
                            total=float(assigned.sum()), mesh_total=mesh_tot, fallback=fallback)
        unclassified = float(T[bin_pool < 0].sum())
        if unclassified > 0:
            raise SystemExit(f"{name}: {unclassified:.3g} in bins not classified as FLiBe")

    # ---- checks vs per-volume tallies -------------------------------------------
    print("\nMapped sources (full machine) vs OpenMC cell tallies")
    ok = True
    for name, r in report.items():
        key = "heating_MW" if name == "qFlibe" else f"{name[2:]}_per_s"
        scale = 1e6 if name == "qFlibe" else 1.0
        cg = float(cells["flibe_vessel_channel"][key]) * scale
        cb = float(cells["flibe_bulk_blanket"][key]) * scale
        for part, ref in (("gap", cg), ("bulk", cb), ("total", cg + cb)):
            dev = r[part] / ref - 1.0
            flag = "" if abs(dev) <= 0.01 else "   <-- FAIL (> 1 %)"
            ok &= abs(dev) <= 0.01
            print(f"  {name:8s} {part:5s} mapped {r[part]:.6e}  cell tally {ref:.6e}  {100 * dev:+.3f} %{flag}")
        print(f"  {name:8s} mapped/mesh total {r['total'] / r['mesh_total'] - 1:+.2e};"
              f" via nearest-cell fallback {100 * r['fallback'] / r['total']:.2f} %")
    print(f"  bin classification (centre/overlap/nearest): {cnt}")
    if not ok:
        raise SystemExit("STOP: mapped totals differ from the cell tallies by more than 1 %")
    patches = {p: ("wedge" if v["type"] == "wedge" else "calculated") for p, v in bnd.items()}
    for name, vals in out.items():
        dims = "[1 -1 -3 0 0 0 0]" if name == "qFlibe" else "[0 -3 -1 0 0 0 0]"
        what = ("FLiBe nuclear heating (W/m3)" if name == "qFlibe"
                else f"{name[2:]} production (atoms/m3/s)")
        foam_io.write_scalar_field(case / "0" / name, name, vals, dims, patches,
                                   comment=f"{what}, mapped conservatively from {summ['openmc_run']}"
                                           f"/source_fields.npz by cfd/make_case.py map")
    summ["mapped"] = report
    (case / "case_summary.json").write_text(json.dumps(summ, indent=1, default=float))
    return report


# =============================================================================
def prepare(level: str | None = None):
    om = load_openmc()
    geo = build_geometry()
    R = geo["regions"]
    print(f"FLiBe volumes (360 deg): gap {_ring_volume_m3(R['vv_cooling_channel']):.2f} m3,"
          f" bulk {_ring_volume_m3(R['flibe_blanket']):.2f} m3; circuit gaps "
          + ", ".join(f"{c['name']} {_ring_volume_m3(c['gap']):.2f}" for c in geo["circuits"]))
    loads = circuit_heat_loads(geo, om)
    fl = circuit_flows(geo, loads)
    size_openings(geo, fl["circuits"])
    bypass_and_outlet(geo, fl["m_bypass"])
    print_flow_table(geo, loads, fl)
    return om, geo, loads, fl


def cmd_circuits(args):
    _, geo, _, fl = prepare()
    plot_circuits(geo, fl, Path(CFG["circuits_png"]))


def cmd_build(args):
    level = "coarse" if args.coarse else "fine"
    om, geo, loads, fl = prepare()
    plot_circuits(geo, fl, Path(CFG["circuits_png"]))
    mu = viscosity_table()
    print(f"\nViscosity table (tabulatedTransport): {mu['n']} points {mu['points'][0][0]:g}-{mu['points'][-1][0]:g} K, "
          f"interpolation error {100 * mu['max_err_700_1400K']:.3f} % (700-1400 K), "
          f"{100 * mu['max_err_all']:.3f} % (whole table)")
    A, B = CFG["mu_arrhenius"]
    print(f"  Pr at 900 K = {A * math.exp(B / 900.0) * CFG['cp'] / CFG['kappa']:.1f}")
    decay = decay_data()
    print("  half-lives (openmc.data): " + ", ".join(f"{i} {d['half_life_s']:.6g} s" for i, d in decay.items()))
    case = Path(CFG["case_dir"])
    case.mkdir(parents=True, exist_ok=True)
    m2 = mesh_2d(geo, level)
    rep = write_wedge_msh(m2, geo, case / "v3a.msh")
    print_mesh_report(rep)
    P = patch_table(geo, loads, fl)
    write_case(case, geo, loads, fl, P, mu, decay, level, rep)
    print(f"\nwrote {case}/ ({level}: {CFG['end_iterations'][level]} iterations, "
          f"{max(CFG['n_procs'][level], 1)} process(es)). Next, in an OpenFOAM shell: cd {case} && ./Allrun mesh")


def cmd_map(args):
    map_sources(Path(CFG["case_dir"]))
    print("wrote 0/qFlibe and 0/S_*. Next, in an OpenFOAM shell: ./Allrun solve")


def main():
    ap = argparse.ArgumentParser(description="ARC V3A FLiBe OpenFOAM case builder")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("circuits", help="circuits, heat loads, flows and a plot (no OpenFOAM)")
    b = sub.add_parser("build", help="mesh + full OpenFOAM case")
    b.add_argument("--coarse", action="store_true", help="smoke-test mesh and run length")
    sub.add_parser("map", help="map OpenMC sources onto the FLiBe cells (after ./Allrun mesh)")
    args = ap.parse_args()
    {"circuits": cmd_circuits, "build": cmd_build, "map": cmd_map}[args.cmd](args)


if __name__ == "__main__":
    main()
