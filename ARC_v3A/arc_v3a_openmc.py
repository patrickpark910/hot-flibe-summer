#!/usr/bin/env python3
"""
OpenMC fixed-source model for the ARC V3A DAGMC geometry
=========================================================

Turns the Paramak/cad_to_dagmc file (e.g. arc_v3a_deg10.h5m) into an OpenMC run
that produces the source terms for the 2D axisymmetric FLiBe CFD model:

    * nuclear heating per material on an R-Z mesh (W/m3)
    * F-19 activation rates on the same mesh: N-16 (n,a), F-18 (n,2n),
      O-19 (n,p), F-20 (n,gamma)  (atoms/m3/s)
    * tritium production in FLiBe (atoms/m3/s)
    * per-volume totals (heating, activation, TBR) for conservation checks

Subcommands
-----------
    python arc_v3a_openmc.py inspect --h5m arc_v3a_deg10.h5m
        Lists DAGMC volumes, materials, sector angle and faceted volumes. Needs h5py only.

    python arc_v3a_openmc.py build --h5m arc_v3a_deg10.h5m --out run_v3a
        Writes the OpenMC XML inputs plus model_info.json to ./run_v3a.

    python arc_v3a_openmc.py run --h5m arc_v3a_deg10.h5m --out run_v3a --threads 16
        Build, then run OpenMC.

    python arc_v3a_openmc.py post --out run_v3a
        Reads the last statepoint and writes CFD-ready fields
        (source_fields.npz), per-volume totals (cell_totals.csv) and a check plot.

Requirements
------------
    OpenMC built with DAGMC (conda-forge: openmc=*=dagmc*), h5py, numpy;
    openmc-plasma-source >= 0.9 for the default tokamak source (or use --source ring);
    matplotlib for the check plot. Set OPENMC_CROSS_SECTIONS to a library with
    photon data (e.g. ENDF/B-VIII.0 or FENDL-3.2 HDF5 from openmc.org).

Where the numbers come from
---------------------------
[H26]  Hillesheim et al., J. Plasma Phys. 92 (2026) E69. Table 2: R0 = 4.62 m,
       a = 1.18 m, kappa = 1.8, delta = 0.65, Pfus = 1.13 GW. Table 4: 4e20 n/s.
[Z26]  ARC V3A digital assets, doi:10.5281/zenodo.19498373: input.gacode_V3A
       (profiles) and geqdsk-ARCv3a (magnetic axis at R = 4.6996 m).
[B22]  Bae et al., Nucl. Fusion 62 (2022) 066016, Table 2: densities and the
       magnet mix (50 vol% Cu + 50 vol% SS316L, 8.48 g/cm3), 90% Li-6.
[RH17] Romatoski & Hu, Ann. Nucl. Energy 109 (2017): FLiBe density
       2413 - 0.488 T kg/m3 -> 1.974 g/cm3 at 900 K.
Other compositions (V-4Cr-4Ti, 316L) are standard nominal values; the V3A
materials themselves are placeholders [H26].
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
import sys

import numpy as np

EV_TO_J = 1.602176634e-19

# ---------------------------------------------------------------------------
# Operating point [H26]
# ---------------------------------------------------------------------------
FUSION_POWER_MW = 1133.0
NEUTRON_RATE = 4.0e20          # neutrons/s, full machine, flattop [H26 Table 4]

# Tokamak source for openmc-plasma-source (cm, m^-3, eV). Fausser-type H-mode
# profiles least-squares fitted to the D+T density and Ti of input.gacode_V3A
# [Z26]: density max error 4 %, Ti max error 22 % (at the far edge, where
# almost no neutrons are made). With this shape the profiles give ~1.19 GW,
# close to the 1.13 GW design value; tallies are normalised to NEUTRON_RATE
# anyway, so only the spatial and energy shape of this source is used.
V3A_PLASMA = dict(
    major_radius=462.0,
    minor_radius=118.07,          # rmin at the separatrix in input.gacode_V3A
    elongation=1.8,
    triangularity=0.65,
    mode="H",
    ion_density_centre=2.713e20,
    ion_density_peaking_factor=1.061,
    ion_density_pedestal=1.800e20,
    ion_density_separatrix=6.913e19,
    ion_temperature_centre=20106.0,
    ion_temperature_peaking_factor=1.547,
    ion_temperature_beta=1.626,
    ion_temperature_pedestal=5427.0,
    ion_temperature_separatrix=75.0,
    pedestal_radius=114.08,
    shafranov_factor=7.96,        # magnetic axis 469.96 cm - R0 462 cm [Z26]
)
MAGNETIC_AXIS_R_CM = 469.96       # used by the simple ring source

# F-19 activation channels tallied: OpenMC score -> product
F19_REACTIONS = {"(n,a)": "N16", "(n,2n)": "F18", "(n,p)": "O19", "(n,gamma)": "F20"}

# Materials whose heating is resolved on the mesh (all sit inside the tank)
MESH_HEATING_MATERIALS = ["flibe", "tungsten", "vanadium", "tungsten_carbide", "ss316l"]

# IDs for the CSG wrapper, far above anything in the DAGMC file
CSG_ID0 = 100000


# ---------------------------------------------------------------------------
# DAGMC file reader (h5py only)
# ---------------------------------------------------------------------------
def read_dagmc(h5m: str) -> dict:
    """Volumes, material tags, faceted volumes and extents from a DAGMC h5m."""
    import h5py

    with h5py.File(h5m, "r") as f:
        t = f["tstt"]
        coords = t["nodes/coordinates"][()]
        node0 = int(t["nodes/coordinates"].attrs["start_id"])
        conn = t["elements/Tri3/connectivity"][()].astype(np.int64)
        tri0 = int(t["elements/Tri3/connectivity"].attrs["start_id"])
        slist = t["sets/list"][()]
        set0 = int(t["sets/list"].attrs["start_id"])
        contents = t["sets/contents"][()].astype(np.int64)

        def decode(v):
            return bytes(v).split(b"\0")[0].decode()

        cat = {int(k): decode(v) for k, v in zip(t["tags/CATEGORY/id_list"][()], t["tags/CATEGORY/values"][()])}
        gid = {int(k): int(v) for k, v in zip(t["tags/GLOBAL_ID/id_list"][()], t["tags/GLOBAL_ID/values"][()])}
        names = {int(k): decode(v) for k, v in zip(t["tags/NAME/id_list"][()], t["tags/NAME/values"][()])}
        sense = {int(k): (int(v[0]), int(v[1])) for k, v in
                 zip(t["tags/GEOM_SENSE_2/id_list"][()], t["tags/GEOM_SENSE_2/values"][()])}

    def set_contents(h):
        i = h - set0
        start = slist[i - 1, 0] + 1 if i > 0 else 0
        c = contents[start:slist[i, 0] + 1]
        if c.size and slist[i, 3] & 0x8:            # ranged storage: (first, count) pairs
            return np.concatenate([np.arange(a, a + n) for a, n in c.reshape(-1, 2)])
        return c

    vols = sorted((h for h, c in cat.items() if c == "Volume"), key=lambda h: gid[h])
    vol_mat = {}
    for g in (h for h, c in cat.items() if c == "Group"):
        nm = names.get(g, "")
        if nm.startswith("mat:"):
            for h in set_contents(g):
                if cat.get(int(h)) == "Volume":
                    vol_mat[int(h)] = nm[4:]

    acc = {v: [0.0, []] for v in vols}
    for s in (h for h, c in cat.items() if c == "Surface"):
        h = set_contents(s)
        tris = h[(h >= tri0) & (h < tri0 + len(conn))] - tri0
        if tris.size == 0:
            continue
        tv = coords[conn[tris] - node0]
        sv = np.einsum("ij,ij->i", tv[:, 0], np.cross(tv[:, 1], tv[:, 2])).sum() / 6.0
        fw, rv = sense.get(s, (0, 0))
        for v, sign in ((fw, 1.0), (rv, -1.0)):
            if v in acc:
                acc[v][0] += sign * sv
                acc[v][1].append(tv.reshape(-1, 3))

    R_all = np.hypot(coords[:, 0], coords[:, 1])
    phi_all = np.degrees(np.arctan2(coords[:, 1], coords[:, 0]))
    sector = float(np.round(phi_all.max(), 3))
    if phi_all.min() < -1e-6 or sector > 359.9:
        sector = 360.0

    volumes = []
    for v in vols:
        p = np.vstack(acc[v][1]) if acc[v][1] else np.zeros((1, 3))
        R = np.hypot(p[:, 0], p[:, 1])
        volumes.append(dict(id=gid[v], material=vol_mat.get(v, "void"),
                            volume_cm3=acc[v][0],
                            r_min=float(R.min()), r_max=float(R.max()),
                            z_min=float(p[:, 2].min()), z_max=float(p[:, 2].max())))
    # label the FLiBe regions: the smaller FLiBe volume is the vessel channel
    flibe = sorted((v for v in volumes if v["material"] == "flibe"), key=lambda v: v["volume_cm3"])
    for v in volumes:
        v["label"] = f"{v['material']}_{v['id']}"
    if len(flibe) == 2:
        flibe[0]["label"], flibe[1]["label"] = "flibe_vessel_channel", "flibe_bulk_blanket"
    return dict(volumes=volumes, sector_deg=sector, r_max=float(R_all.max()),
                z_min=float(coords[:, 2].min()), z_max=float(coords[:, 2].max()))


def print_dagmc(info: dict):
    k = 360.0 / info["sector_deg"]
    print(f"sector: 0 - {info['sector_deg']:g} deg  (full-machine scale factor {k:g})")
    print(f"{'id':>4} {'label':24s} {'material':18s} {'V full (m3)':>12} {'R (m)':>13} {'Z (m)':>14}")
    for v in info["volumes"]:
        print(f"{v['id']:4d} {v['label']:24s} {v['material']:18s} {v['volume_cm3'] * 1e-6 * k:12.2f} "
              f"{v['r_min'] / 100:6.2f}-{v['r_max'] / 100:5.2f} {v['z_min'] / 100:6.2f}-{v['z_max'] / 100:5.2f}")
    fl = sum(v["volume_cm3"] for v in info["volumes"] if v["material"] == "flibe") * 1e-6 * k
    print(f"total FLiBe: {fl:.1f} m3")


def mesh_extent(info: dict, pad_cm: float):
    """R-Z box around FLiBe, vessel, shield and tank wall."""
    keep = [v for v in info["volumes"]
            if v["material"] in ("flibe", "tungsten", "vanadium", "tungsten_carbide")]
    r0 = min(v["r_min"] for v in keep) - pad_cm
    r1 = max(v["r_max"] for v in keep) + pad_cm
    z0 = min(v["z_min"] for v in keep) - pad_cm
    z1 = max(v["z_max"] for v in keep) + pad_cm
    return (max(r0, 0.0), r1, z0, z1)


# ---------------------------------------------------------------------------
# Model pieces (OpenMC imported lazily so 'inspect' works without it)
# ---------------------------------------------------------------------------
def make_materials(li6_percent: float, flibe_density: float):
    import openmc

    # Names must match the h5m tags exactly; IDs are left to OpenMC.
    def mat(name, density, temperature=None):
        m = openmc.Material(name=name)
        m.set_density("g/cm3", density)
        if temperature is not None:
            m.temperature = temperature
        return m

    plasma = mat("plasma", 5e-8)                           # [B22] convention
    plasma.add_nuclide("H2", 0.5)
    plasma.add_nuclide("H3", 0.5)

    tungsten = mat("tungsten", 19.3, 873.15)               # first wall
    tungsten.add_element("W", 1.0)

    vanadium = mat("vanadium", 6.11, 873.15)               # V-4Cr-4Ti, vessel walls
    for el, wo in (("V", 0.92), ("Cr", 0.04), ("Ti", 0.04)):
        vanadium.add_element(el, wo, percent_type="wo")

    flibe = mat("flibe", flibe_density, 900.0)             # [RH17] at 900 K
    flibe.add_elements_from_formula("Li2BeF4", enrichment=li6_percent,
                                    enrichment_target="Li6", enrichment_type="ao")

    wc = mat("tungsten_carbide", 15.63)                    # inboard shield
    wc.add_elements_from_formula("WC")

    ss = mat("ss316l", 8.0)                                # tank wall, CS structure
    for el, wo in (("Fe", 65.645), ("Cr", 17.0), ("Ni", 12.0), ("Mo", 2.5), ("Mn", 2.0),
                   ("Si", 0.75), ("P", 0.045), ("S", 0.03), ("C", 0.03)):
        ss.add_element(el, wo, percent_type="wo")

    cu = openmc.Material(name="copper_helper")
    cu.add_element("Cu", 1.0)
    cu.set_density("g/cm3", 8.96)
    mats = [plasma, tungsten, vanadium, flibe, wc, ss]
    for name in ("magnet", "tf_magnet"):                   # [B22]: 50/50 vol Cu + SS316L
        mats.append(openmc.Material.mix_materials([cu, ss], [0.5, 0.5], "vo", name=name))
    return {m.name: m for m in mats}


def make_geometry(h5m: str, info: dict, margin_cm: float = 50.0):
    import openmc

    dag = openmc.DAGMCUniverse(h5m, universe_id=CSG_ID0, auto_geom_ids=False)
    cyl = openmc.ZCylinder(r=info["r_max"] + margin_cm, boundary_type="vacuum", surface_id=CSG_ID0 + 1)
    zlo = openmc.ZPlane(z0=info["z_min"] - margin_cm, boundary_type="vacuum", surface_id=CSG_ID0 + 2)
    zhi = openmc.ZPlane(z0=info["z_max"] + margin_cm, boundary_type="vacuum", surface_id=CSG_ID0 + 3)
    region = -cyl & +zlo & -zhi
    sector = info["sector_deg"]
    if sector < 360.0:
        if sector > 180.0:
            raise SystemExit("sectors wider than 180 deg are not handled; use 360 or <= 180")
        a = math.radians(sector)
        p0 = openmc.YPlane(y0=0.0, boundary_type="reflective", surface_id=CSG_ID0 + 4)
        p1 = openmc.Plane(a=-math.sin(a), b=math.cos(a), c=0.0, d=0.0,
                          boundary_type="reflective", surface_id=CSG_ID0 + 5)
        region &= +p0 & -p1
    # The DAGMC sector faces coincide with the reflecting planes; OpenMC gives
    # the higher (CSG) level priority on coincident boundaries.
    cell = openmc.Cell(cell_id=CSG_ID0, fill=dag, region=region)
    root = openmc.Universe(universe_id=CSG_ID0 + 1, cells=[cell])
    return openmc.Geometry(root)


def make_source(kind: str, sector_deg: float):
    import openmc

    rot = math.radians(sector_deg)
    if kind == "tokamak":
        import inspect
        from openmc_plasma_source import tokamak_source

        if "rotation_angle" not in inspect.signature(tokamak_source).parameters:
            raise SystemExit("openmc-plasma-source is too old (needs >= 0.9: rotation_angle "
                             "argument). Upgrade it, or use --source ring.")
        ts = tokamak_source(**V3A_PLASMA, start_angle=0.0, rotation_angle=rot)
        full = 2 * math.pi / rot
        print(f"tokamak source: plasma volume {ts.plasma_volume * full:.0f} m3, "
              f"profile fusion power {ts.fusion_power * full:.0f} MW (full machine)")
        # openmc-plasma-source sets element strengths in neutrons/s. In fixed-source
        # mode OpenMC multiplies every tally by the total source strength, so scale
        # the total to 1 to keep tallies per source particle ('post' applies the
        # physical neutron rate).
        src = ts.source
        total = src.strength
        for s in src.sources:
            s.strength /= total
        return src
    if kind == "ring":   # 14.06 MeV isotropic ring at the magnetic axis, as in [B22]
        src = openmc.IndependentSource()
        src.space = openmc.stats.CylindricalIndependent(
            r=openmc.stats.Discrete([MAGNETIC_AXIS_R_CM], [1.0]),
            phi=openmc.stats.Uniform(0.0, rot),
            z=openmc.stats.Discrete([0.0], [1.0]))
        src.angle = openmc.stats.Isotropic()
        src.energy = openmc.stats.Discrete([14.06e6], [1.0])
        return src
    raise SystemExit(f"unknown source kind {kind!r}")


def make_tallies(mats: dict, info: dict, bin_cm: float, photons: bool):
    import openmc

    r0, r1, z0, z1 = mesh_extent(info, pad_cm=3.0)
    nr, nz = int(math.ceil((r1 - r0) / bin_cm)), int(math.ceil((z1 - z0) / bin_cm))
    mesh = openmc.CylindricalMesh(
        r_grid=np.linspace(r0, r0 + nr * bin_cm, nr + 1),
        z_grid=np.linspace(z0, z0 + nz * bin_cm, nz + 1),
        phi_grid=(0.0, math.radians(info["sector_deg"])),
        name="rz_mesh")      # no fixed ID: the tokamak source creates its own mesh
    print(f"R-Z mesh: {nr} x {nz} bins of {bin_cm:g} cm, R {r0:.0f}-{r0 + nr * bin_cm:.0f} cm, "
          f"Z {z0:.0f}-{z0 + nz * bin_cm:.0f} cm")
    mf = openmc.MeshFilter(mesh)
    heat = "heating" if photons else "heating-local"
    heat_mats = [mats[n] for n in MESH_HEATING_MATERIALS if n in mats]
    flibe_ids = [v["id"] for v in info["volumes"] if v["material"] == "flibe"]
    all_ids = [v["id"] for v in info["volumes"]]

    t_rx = openmc.Tally(name="mesh_F19_reactions")
    t_rx.filters, t_rx.nuclides, t_rx.scores = [mf], ["F19"], list(F19_REACTIONS)
    t_heat = openmc.Tally(name="mesh_heating")
    t_heat.filters, t_heat.scores = [mf, openmc.MaterialFilter(heat_mats)], [heat]
    t_h3 = openmc.Tally(name="mesh_tritium")
    t_h3.filters, t_h3.scores = [mf, openmc.MaterialFilter([mats["flibe"]])], ["H3-production"]
    t_cell = openmc.Tally(name="cell_totals")
    t_cell.filters, t_cell.scores = [openmc.CellFilter(all_ids)], [heat, "H3-production"]
    t_cell_rx = openmc.Tally(name="cell_F19_reactions")
    t_cell_rx.filters, t_cell_rx.nuclides = [openmc.CellFilter(flibe_ids)], ["F19"]
    t_cell_rx.scores = list(F19_REACTIONS)
    tallies = openmc.Tallies([t_rx, t_heat, t_h3, t_cell, t_cell_rx])
    return tallies, [m.name for m in heat_mats]


def build(args) -> "openmc.Model":
    import openmc

    info = read_dagmc(args.h5m)
    print_dagmc(info)
    mats = make_materials(args.li6, args.flibe_density)
    missing = sorted({v["material"] for v in info["volumes"]} - set(mats) - {"void"})
    if missing:
        raise SystemExit(f"h5m material tags with no OpenMC material: {missing}")

    settings = openmc.Settings()
    settings.run_mode = "fixed source"
    settings.batches = args.batches
    settings.particles = int(args.particles)
    settings.source = make_source(args.source, info["sector_deg"])
    settings.photon_transport = not args.neutron_only
    settings.electron_treatment = "ttb"
    # use the closest temperature the library has (fusion results are insensitive)
    settings.temperature = {"default": 293.6, "method": "nearest", "tolerance": 1000.0}
    settings.output = {"tallies": False}
    settings.seed = args.seed

    tallies, heat_names = make_tallies(mats, info, args.bin_cm, photons=not args.neutron_only)
    # absolute path: OpenMC runs inside --out, so a relative h5m path would break
    model = openmc.Model(geometry=make_geometry(os.path.abspath(args.h5m), info),
                         materials=openmc.Materials(mats.values()),
                         settings=settings, tallies=tallies)
    os.makedirs(args.out, exist_ok=True)
    model.export_to_xml(directory=args.out)
    meta = dict(h5m=os.path.abspath(args.h5m), sector_deg=info["sector_deg"],
                neutron_rate=args.neutron_rate, fusion_power_MW=FUSION_POWER_MW,
                photons=not args.neutron_only, source=args.source, li6_percent=args.li6,
                flibe_density=args.flibe_density, heating_materials=heat_names,
                volumes=info["volumes"])
    with open(os.path.join(args.out, "model_info.json"), "w") as fh:
        json.dump(meta, fh, indent=1)
    n_hist = args.batches * int(args.particles)
    print(f"wrote OpenMC inputs and model_info.json to {args.out}/  ({n_hist:.2e} histories)")
    if not os.environ.get("OPENMC_CROSS_SECTIONS"):
        print("warning: OPENMC_CROSS_SECTIONS is not set; OpenMC needs it to run")
    return model


# ---------------------------------------------------------------------------
# Post-processing
# ---------------------------------------------------------------------------
def total_source_strength(run_dir: str) -> float:
    """Sum of top-level source strengths, which OpenMC multiplies into every
    fixed-source tally. Read from the run's settings.xml (or model.xml)."""
    import xml.etree.ElementTree as ET

    path = os.path.join(run_dir, "settings.xml")
    if os.path.exists(path):
        settings = ET.parse(path).getroot()
    else:
        settings = ET.parse(os.path.join(run_dir, "model.xml")).getroot().find("settings")
    total = 0.0
    for src in settings.findall("source"):           # direct children only
        val = src.get("strength") or src.findtext("strength")
        total += float(val) if val is not None else 1.0
    return total if total > 0 else 1.0


def post(args):
    import openmc

    with open(os.path.join(args.out, "model_info.json")) as fh:
        meta = json.load(fh)
    sps = sorted(glob.glob(os.path.join(args.out, "statepoint.*.h5")))
    if not sps:
        raise SystemExit(f"no statepoint in {args.out}")
    sp = openmc.StatePoint(sps[-1])
    S_full = meta["neutron_rate"]                    # n/s, full machine
    k = 360.0 / meta["sector_deg"]
    W = total_source_strength(args.out)
    per_src = 1.0 / W                                # undo OpenMC's strength scaling
    print(f"{os.path.basename(sps[-1])}: {sp.n_batches} batches x {sp.n_particles} particles; "
          f"source strength in inputs {W:.4g} (divided out); "
          f"normalised to {S_full:.3e} n/s (full machine)")

    # --- mesh fields ---------------------------------------------------------
    t_rx = sp.get_tally(name="mesh_F19_reactions")
    mesh = t_rx.find_filter(openmc.MeshFilter).mesh
    V_m3 = mesh.volumes[:, 0, :] * 1e-6               # sector bin volume (nr, nz)
    S_sector = S_full / k                            # neutrons born in the sector
    out = dict(r_edges_cm=np.asarray(mesh.r_grid), z_edges_cm=np.asarray(mesh.z_grid),
               sector_deg=meta["sector_deg"], neutron_rate=S_full)

    def field(tally, mean_scale):
        m = tally.get_reshaped_data("mean", expand_dims=True)[:, 0, :]     # drop phi
        s = tally.get_reshaped_data("std_dev", expand_dims=True)[:, 0, :]
        with np.errstate(divide="ignore", invalid="ignore"):
            rel = np.where(m > 0, s / m, np.nan)
        return m * mean_scale, rel

    # per source particle -> per m3 per s in the sector bin: x S_sector / V
    scale = S_sector / V_m3                          # (nr, nz)
    rx, rx_rel = field(t_rx, per_src)                # (nr, nz, 1 nuclide, n scores)
    for i, score in enumerate(F19_REACTIONS):
        iso = F19_REACTIONS[score]
        out[f"{iso}_generation_per_m3_s"] = rx[:, :, 0, i] * scale
        out[f"{iso}_rel_err"] = rx_rel[:, :, 0, i]

    t_heat = sp.get_tally(name="mesh_heating")
    ht, ht_rel = field(t_heat, EV_TO_J * per_src)     # (nr, nz, n materials, 1, 1)
    for j, name in enumerate(meta["heating_materials"]):
        out[f"heating_{name}_W_m3"] = ht[:, :, j, 0, 0] * scale
        out[f"heating_{name}_rel_err"] = ht_rel[:, :, j, 0, 0]

    t_h3 = sp.get_tally(name="mesh_tritium")
    h3, h3_rel = field(t_h3, per_src)                 # (nr, nz, 1, 1, 1)
    out["H3_generation_per_m3_s"] = h3[:, :, 0, 0, 0] * scale
    out["H3_rel_err"] = h3_rel[:, :, 0, 0, 0]
    # bin-averaged densities: a bin cut by a wall holds FLiBe-only totals spread
    # over the whole bin. For conservative mapping use density x ring volume
    # (full-machine total per bin) and spread it over the FLiBe CFD cells in it.
    out["bin_ring_volume_m3"] = V_m3 * k
    np.savez_compressed(os.path.join(args.out, "source_fields.npz"), **out)

    # --- per-volume totals -------------------------------------------------
    labels = {v["id"]: v for v in meta["volumes"]}
    tc = sp.get_tally(name="cell_totals")
    ids = list(tc.find_filter(openmc.CellFilter).bins)
    m, s = tc.mean[:, 0, :] * per_src, tc.std_dev[:, 0, :] * per_src
    tcr = sp.get_tally(name="cell_F19_reactions")
    rx_ids = list(tcr.find_filter(openmc.CellFilter).bins)
    rows, tot_heat, tbr = [], 0.0, 0.0
    for i, cid in enumerate(ids):
        v = labels[int(cid)]
        row = dict(id=int(cid), label=v["label"], material=v["material"],
                   volume_m3=v["volume_cm3"] * 1e-6 * k,
                   heating_MW=m[i, 0] * S_full * EV_TO_J * 1e-6,
                   heating_rel_err=s[i, 0] / m[i, 0] if m[i, 0] > 0 else float("nan"),
                   H3_per_s=m[i, 1] * S_full)
        tot_heat += row["heating_MW"]
        tbr += m[i, 1]
        if cid in rx_ids:
            j = rx_ids.index(cid)
            for q, score in enumerate(F19_REACTIONS):
                row[f"{F19_REACTIONS[score]}_per_s"] = tcr.mean[j, 0, q] * per_src * S_full
        rows.append(row)
    keys = (["id", "label", "material", "volume_m3", "heating_MW", "heating_rel_err", "H3_per_s"]
            + [f"{iso}_per_s" for iso in F19_REACTIONS.values()])
    with open(os.path.join(args.out, "cell_totals.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys, restval="")
        w.writeheader()
        w.writerows(rows)

    print(f"\nTBR = {tbr:.4f}")
    if not 0.5 < tbr < 2.0:
        print("warning: TBR outside 0.5-2; check the source-strength normalisation")
    print(f"total nuclear heating = {tot_heat:.0f} MW  (H26 implies ~1000 MW: ~906 MW of "
          f"neutrons + ~100 MW from lithium reactions)")
    for r in rows:
        if r["material"] == "flibe":
            print(f"  {r['label']:22s} {r['heating_MW']:8.1f} MW  N-16 {r.get('N16_per_s', 0):.3e}/s  "
                  f"F-20 {r.get('F20_per_s', 0):.3e}/s  O-19 {r.get('O19_per_s', 0):.3e}/s  "
                  f"F-18 {r.get('F18_per_s', 0):.3e}/s")
    # mesh vs cell consistency (FLiBe heating)
    if "heating_flibe_W_m3" in out:
        mesh_flibe_MW = np.nansum(out["heating_flibe_W_m3"] * out["bin_ring_volume_m3"]) * 1e-6
        cell_flibe_MW = sum(r["heating_MW"] for r in rows if r["material"] == "flibe")
        print(f"FLiBe heating: mesh sum {mesh_flibe_MW:.1f} MW vs cell sum {cell_flibe_MW:.1f} MW "
              f"(should agree; a gap means the mesh misses part of the FLiBe)")
    n16 = out["N16_generation_per_m3_s"]
    sig = n16 > 0.01 * np.nanmax(n16)
    print(f"N-16 rel. error where generation > 1% of peak: median {np.nanmedian(out['N16_rel_err'][sig]):.3f}, "
          f"95th pct {np.nanpercentile(out['N16_rel_err'][sig], 95):.3f}")
    print(f"wrote source_fields.npz and cell_totals.csv to {args.out}/")
    plot(out, os.path.join(args.out, "source_fields.png"))


def plot(out: dict, filename: str):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.colors import LogNorm
    except ImportError:
        return
    r, z = out["r_edges_cm"] / 100, out["z_edges_cm"] / 100
    panels = [("heating_flibe_W_m3", "FLiBe heating (W/m3)"),
              ("N16_generation_per_m3_s", "N-16 generation (1/m3/s)"),
              ("F20_generation_per_m3_s", "F-20 generation (1/m3/s)"),
              ("N16_rel_err", "N-16 relative error")]
    fig, axs = plt.subplots(1, 4, figsize=(16, 6))
    for ax, (key, title) in zip(axs, panels):
        d = np.ma.masked_invalid(np.where(out[key] > 0, out[key], np.nan)).T
        norm = None if key.endswith("rel_err") else LogNorm(vmin=max(d.max() * 1e-4, d.min()), vmax=d.max())
        pc = ax.pcolormesh(r, z, d, norm=norm, shading="flat",
                           vmin=0 if norm is None else None, vmax=0.2 if norm is None else None)
        fig.colorbar(pc, ax=ax, shrink=0.8)
        ax.set_title(title)
        ax.set_aspect("equal")
        ax.set_xlabel("R (m)")
    axs[0].set_ylabel("Z (m)")
    fig.tight_layout()
    fig.savefig(filename, dpi=150)
    plt.close(fig)
    print(f"wrote {filename}")


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="ARC V3A OpenMC model from a DAGMC h5m")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p_ins = sub.add_parser("inspect", help="list DAGMC volumes (h5py only)")
    p_ins.add_argument("--h5m", required=True)
    for name in ("build", "run"):
        p = sub.add_parser(name)
        p.add_argument("--h5m", required=True)
        p.add_argument("--out", default="run_v3a", help="output directory")
        p.add_argument("--particles", type=float, default=1e5, help="particles per batch")
        p.add_argument("--batches", type=int, default=10)
        p.add_argument("--bin-cm", type=float, default=1.0, help="R-Z mesh bin size (cm)")
        p.add_argument("--source", choices=["tokamak", "ring"], default="tokamak")
        p.add_argument("--neutron-only", action="store_true",
                       help="no photon transport (faster; heating becomes heating-local)")
        p.add_argument("--li6", type=float, default=90.0, help="Li-6 enrichment, atom %%")
        p.add_argument("--flibe-density", type=float, default=1.974, help="g/cm3 (1.974 = 900 K)")
        p.add_argument("--neutron-rate", type=float, default=NEUTRON_RATE, help="n/s, full machine")
        p.add_argument("--seed", type=int, default=1)
        if name == "run":
            p.add_argument("--threads", type=int, default=None)
    p_post = sub.add_parser("post", help="statepoint -> CFD source fields")
    p_post.add_argument("--out", default="run_v3a")
    args = ap.parse_args()

    if args.cmd == "inspect":
        print_dagmc(read_dagmc(args.h5m))
    elif args.cmd == "build":
        build(args)
    elif args.cmd == "run":
        import openmc
        build(args)
        openmc.run(threads=args.threads, cwd=args.out)
    elif args.cmd == "post":
        post(args)


if __name__ == "__main__":
    main()
