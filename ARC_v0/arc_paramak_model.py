#!/usr/bin/env python3
"""
ARC fusion pilot plant: parametric CAD model built with Paramak (>= 0.10)
=========================================================================

Builds a neutronics-ready CAD model of the ARC tokamak (Sorbom et al. 2015,
with the long-leg X-point-target divertor of Kuang et al. 2018):

    plasma
    double-walled vacuum vessel (W first wall / Inconel / FLiBe channel / Be / Inconel)
        with upper and lower long-leg divertors ending in circular "feet"
    FLiBe immersion blanket and blanket tank
    TiH2 inboard TF shield
    ZrH2 L-shaped neutron shields around the superconducting divertor coils
    PF1-PF3 divertor coils (in pockets of the tank), PF4-PF5 vertical-field coils
    TR1-TR3 copper trim / vertical-stability coils inside the tank
    central solenoid and 18 TF coils

Every component is generated from 2D R-Z profiles (shapely) and turned into
3D with paramak.revolved_shape / paramak.plasma_simplified, then collected
in a paramak.Assembly. Adjacent layers share exactly the same profile, so
their surfaces coincide (good for DAGMC imprinting) and nothing overlaps.

Units: cm (paramak / OpenMC convention).

Where the numbers come from
---------------------------
Where the sources disagree, the most recent one wins: values given in
[Badalassi23] (FERMI) take precedence over [Bae22], [Kuang18] and [Aimetta22].

[Badalassi23] V. Badalassi et al., "FERMI: Fusion Energy Reactor Models
            Integrator", Fus. Sci. Technol. 79 (2023) 345. Sec. I.B: ARC-class
            tokamak with R0 = 3.3 m, tungsten-alloy first wall, Inconel 718
            vacuum vessel, 2 cm FLiBe cooling channel; Sec. II.B.3: FLiBe with
            90 % Li-6; Table II: FLiBe density 1940 kg/m3. FERMI gives no
            other dimensions in its text, so the rest of the build still comes
            from the papers below (FERMI's neutronics re-uses the Bae22 model).
[Bae22]     J.W. Bae, E. Peterson, J. Shimwell, "ARC reactor neutronics
            multi-code validation", Nucl. Fusion 62 (2022) 066016.
            Table 1 (radial build A-Q), Table 2 (materials/densities),
            Fig. 1 caption (blanket 188.7 mm inboard / 1163.2 mm outboard).
[Kuang18]   A.Q. Kuang et al., "Conceptual design study for heat exhaust
            management in the ARC fusion pilot plant", Fus. Eng. Des. 137
            (2018) 221. Plasma shape, Table 1 coil positions/sizes, 25 cm ZrH2
            PF shields, secondary X-point at R = 3.7 m; tank top and outboard
            tank wall and TF coil outline digitised from Fig. 1 (to scale,
            calibrated on the Table 1 coil positions); divertor-foot size
            from Fig. 7.
[Aimetta22] A. Aimetta et al., "Neutronic analysis of the fusion reactor ARC:
            Monte Carlo simulations with the Serpent code", Fus. Sci. Technol.
            78 (2022) 275. Table III: 617 cm between divertor centres.

Things that are modelling choices rather than published values are marked
ASSUMPTION in ARCParameters. The model is "consistent with the published
dimensions", not a copy of the CFS CAD used by Bae et al.

Requirements:  pip install paramak shapely scipy matplotlib
Optional:      pip install cad_to_dagmc      (for --h5m)

Examples:
    python arc_paramak_model.py                       # 90 deg sector -> STEP + PNG
    python arc_paramak_model.py --angle 360           # full torus
    python arc_paramak_model.py --h5m arc.h5m         # also write a DAGMC file
    python arc_paramak_model.py --no-magnets          # blanket/VV/shields only
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field

import cadquery as cq
import numpy as np
import paramak
from paramak.assemblies.assembly import Assembly
from scipy.interpolate import splev, splprep
from shapely import affinity
from shapely.geometry import LineString, Point, Polygon, box
from shapely.ops import unary_union


# --------------------------------------------------------------------------
# Parameters
# --------------------------------------------------------------------------
@dataclass
class ARCParameters:
    # ---- plasma [Kuang18 Sec. 3.2, Sorbom 2015] --------------------------
    major_radius: float = 330.0            # R0 = 3.3 m [Badalassi23 Sec. I.B]
    minor_radius: float = 110.0
    elongation: float = 1.8
    triangularity: float = 0.375

    # ---- vacuum vessel, from the plasma outwards [Bae22 Table 1 Q,P,O,N,M]
    # ASSUMPTION: 2 cm plasma-first-wall gap. With Bae's blanket thicknesses
    # this puts the TF inner leg at R = 134 cm (Kuang18 Fig. 1) and the
    # outboard tank wall at R = 565 cm (Aimetta22 Table III: 563 cm).
    plasma_wall_gap: float = 2.0
    first_wall_thickness: float = 0.1      # Q  tungsten alloy [Badalassi23]
    inner_vv_thickness: float = 1.0        # P  Inconel 718 [Badalassi23]
    flibe_channel_thickness: float = 2.0   # O  FLiBe, 2 cm [Badalassi23 Sec. I.B]
    multiplier_thickness: float = 1.0      # N  beryllium
    outer_vv_thickness: float = 3.0        # M  Inconel 718

    # ---- long-leg divertor -------------------------------------------------
    foot_centre_r: float = 370.0   # secondary X-point R = 3.7 m [Kuang18]
    foot_centre_z: float = 308.5   # 617 cm between divertor centres [Aimetta22]
    foot_radius: float = 25.0      # inner radius; ~0.25 m in Kuang18 Fig. 7
    leg_width: float = 20.0        # ASSUMPTION: inner width, ~0.2 m in Bae22 Fig. 1

    # ---- blanket and tank [Bae22] ----------------------------------------
    blanket_inboard_thickness: float = 18.87    # Fig. 1 caption
    blanket_outboard_thickness: float = 116.32  # Fig. 1 caption
    tank_wall_thickness: float = 3.0            # F
    tank_top_z: float = 378.5                   # FLiBe surface, Kuang18 Fig. 1
    tank_inboard_chamfer: float = 30.0          # 45 deg corners, Kuang18 Fig. 1
    tank_outboard_chamfer: float = 60.0         # keeps the tank under the TF coil
    # outboard tank wall (FLiBe side) as (Z, dR from midplane radius),
    # digitised from Kuang18 Fig. 1
    tank_outboard_wall: tuple = (
        (0.0, 0.0), (25.0, -1.5), (50.0, -6.0), (75.0, -12.0), (100.0, -20.0),
        (120.0, -28.0), (140.0, -35.0), (160.0, -40.0),
    )

    # ---- inboard TF shield [Bae22 J, K, L] ---------------------------------
    tank_to_tf_shield_gap: float = 3.0   # J
    tf_shield_thickness: float = 51.0    # K  TiH2
    tf_shield_to_tf_gap: float = 1.0     # L

    # ---- PF coil packaging [Bae22 A-E, G-I] --------------------------------
    pf_case_thickness: float = 1.0        # A, C, G  (SS316L)
    pf_shield_thickness: float = 25.0     # B, H     (ZrH2)  [also Kuang18 Sec. 4.2]
    pf_shield_to_tank_gap: float = 1.0    # D, I
    trim_case_thickness: float = 1.0      # E        (Inconel 718)

    # ---- coils [Kuang18 Table 1]: name -> (R, Z, width, height), upper half
    pf_coils: dict = field(default_factory=lambda: {
        "pf1": (180.0, 303.0, 20.0, 45.0),
        "pf2": (485.0, 270.0, 20.0, 45.0),
        "pf3": (485.0, 210.0, 20.0, 45.0),
        "pf4": (613.0, 278.0, 20.0, 20.0),
        "pf5": (657.0, 170.0, 20.0, 20.0),
    })
    # trim coils: Kuang18 gives r = 0.11 m / 0.04 m^2 copper; modelled as
    # 20 x 20 cm squares (same area) like the boxes in Bae22 Fig. 1
    trim_coils: dict = field(default_factory=lambda: {
        "tr1": (420.0, 285.0, 20.0, 20.0),
        "tr2": (359.0, 197.0, 20.0, 20.0),
        "tr3": (398.0, 170.0, 20.0, 20.0),
    })

    # ---- central solenoid (Kuang18 Fig. 1, digitised) -----------------------
    cs_inner_radius: float = 34.0
    cs_outer_radius: float = 67.0
    cs_half_height: float = 422.0

    # ---- TF coils -------------------------------------------------------------
    n_tf_coils: int = 18
    tf_thickness: float = 64.0          # inner leg 0.69-1.34 m in Kuang18 Fig. 1
    tf_toroidal_width: float = 23.0     # ASSUMPTION: fits 18 coils at R = 68 cm
    # inner surface of the TF "D" above the straight inner leg (R, Z), from
    # Kuang18 Fig. 1; the inner leg itself sits at the build-derived radius
    tf_inner_arch: tuple = (
        (140, 362), (155, 378), (180, 393), (210, 404), (250, 412), (290, 414),
        (325, 410), (355, 402), (380, 391), (405, 378), (430, 361), (455, 345),
        (480, 321), (500, 300), (530, 255), (553, 222), (566, 196), (574, 172),
        (581, 150), (587, 125), (593, 100), (598, 75), (601, 50), (602, 25),
        (602, 0),
    )
    tf_straight_leg_half_height: float = 340.0

    # ---- discretisation -------------------------------------------------------
    plasma_points: int = 360
    quad_segs: int = 12
    # max deviation of the polyline profiles from the exact offset curves;
    # must stay well below the 0.1 cm first-wall thickness
    simplify_tolerance: float = 0.02


# Material tags (DAGMC "mat:" names) and densities in g/cm3 [Bae22 Table 2],
# replaced by [Badalassi23] wherever FERMI gives a value
MATERIALS = {
    "plasma": 5e-8,
    "tungsten": 19.3,        # FERMI calls it a tungsten alloy but gives no composition
    "inconel718": 8.19,
    "flibe": 1.94,           # 1940 kg/m3 [Badalassi23 Table II] (Bae22: 1.924); 90 % Li-6
    "beryllium": 1.85,       # not listed in Bae22 Table 2; standard density
    "tih2": 3.75,
    "zrh2": 5.61,
    "copper": 8.96,
    "magnet": 8.48,          # 50 % copper-steel / 50 % SS316L (TF, CS, HTS PF coils)
    "ss316l": 8.0,
}

COLORS = {
    "plasma": (0.9, 0.35, 0.6), "tungsten": (0.2, 0.2, 0.2),
    "inconel718": (0.55, 0.55, 0.6), "flibe": (0.45, 0.75, 0.95),
    "beryllium": (0.85, 0.3, 0.25), "tih2": (0.65, 0.5, 0.85),
    "zrh2": (0.95, 0.2, 0.2), "copper": (0.85, 0.5, 0.2),
    "magnet": (0.95, 0.65, 0.2), "ss316l": (0.35, 0.35, 0.4),
}


# --------------------------------------------------------------------------
# 2D helpers
# --------------------------------------------------------------------------
def _mirror(geom):
    return affinity.scale(geom, xfact=1.0, yfact=-1.0, origin=(0.0, 0.0))


def _rect(r, z, w, h):
    return box(r - w / 2.0, z - h / 2.0, r + w / 2.0, z + h / 2.0)


def _mitre(geom, d):
    return geom.buffer(d, join_style="mitre", mitre_limit=10.0)


def _clean(geom, simplify=0.0):
    """Returns a single valid Polygon (raises if the boolean split it)."""
    geom = geom.buffer(0)
    if simplify > 0:
        geom = geom.simplify(simplify, preserve_topology=True)
    if geom.geom_type != "Polygon":
        raise ValueError(f"expected a single polygon, got {geom.geom_type}")
    return geom


@dataclass
class Part:
    """One volume: 3D = revolve(outer) - revolve(each cut)."""
    name: str
    material: str
    outer: Polygon
    cuts: list = field(default_factory=list)
    kind: str = "revolve"            # "revolve", "plasma" or "tf"

    @property
    def region(self):
        """The 2D R-Z footprint of the part."""
        geom = self.outer
        for c in self.cuts:
            geom = geom.difference(c)
        return geom


def plasma_polygon(p: ARCParameters) -> Polygon:
    """Same parametrisation as paramak.plasma_simplified."""
    t = np.linspace(0, 2 * np.pi, p.plasma_points, endpoint=False)
    r = p.major_radius + p.minor_radius * np.cos(t + p.triangularity * np.sin(t))
    z = p.elongation * p.minor_radius * np.sin(t)
    return Polygon(np.c_[r, z])


def vacuum_chamber(p: ARCParameters) -> Polygon:
    """Plasma + gap, plus straight divertor legs to circular feet (up/down)."""
    main = plasma_polygon(p).buffer(p.plasma_wall_gap, quad_segs=p.quad_segs)

    # primary X-point ~ top of the plasma (R0 - delta*a, kappa*a) = (2.89, 1.98) m
    x_point = np.array([p.major_radius - p.triangularity * p.minor_radius,
                        p.elongation * p.minor_radius])
    foot = np.array([p.foot_centre_r, p.foot_centre_z])
    direction = (foot - x_point) / np.linalg.norm(foot - x_point)
    leg_start = x_point - 20.0 * direction     # start inside the chamber
    leg = LineString([leg_start, foot]).buffer(p.leg_width / 2.0, cap_style="flat")
    foot_circle = Point(foot).buffer(p.foot_radius, quad_segs=2 * p.quad_segs)
    upper = unary_union([leg, foot_circle])
    return _clean(unary_union([main, upper, _mirror(upper)]), p.simplify_tolerance)


def _pf_stack(p: ARCParameters, name):
    """coil, case polygons for a coil in the upper half."""
    r, z, w, h = p.pf_coils[name]
    coil = _rect(r, z, w, h)
    case = _mitre(coil, p.pf_case_thickness)
    return coil, case


def build_profiles(p: ARCParameters, magnets: bool = True) -> tuple[list[Part], dict]:
    """Returns the 2D parts and the midplane radial build {component: (r_in, r_out)}."""
    parts: list[Part] = []
    t_sh, g_sh, t_tank = (p.pf_shield_thickness, p.pf_shield_to_tank_gap,
                          p.tank_wall_thickness)

    # ---- plasma ------------------------------------------------------------
    parts.append(Part("plasma", "plasma", plasma_polygon(p), kind="plasma"))

    # ---- vacuum vessel layers ----------------------------------------------
    chamber = vacuum_chamber(p)
    layers = [
        ("first_wall", "tungsten", p.first_wall_thickness),
        ("inner_vacuum_vessel", "inconel718", p.inner_vv_thickness),
        ("flibe_cooling_channel", "flibe", p.flibe_channel_thickness),
        ("beryllium_multiplier", "beryllium", p.multiplier_thickness),
        ("outer_vacuum_vessel", "inconel718", p.outer_vv_thickness),
    ]
    inner, depth = chamber, 0.0
    for name, mat, t in layers:
        depth += t
        outer = _clean(chamber.buffer(depth, quad_segs=p.quad_segs), p.simplify_tolerance)
        parts.append(Part(name, mat, outer, [inner]))
        inner = outer
    vv_envelope = inner

    # ---- radial positions at the midplane ------------------------------------
    vv_in = p.major_radius - p.minor_radius - p.plasma_wall_gap - depth
    vv_out = p.major_radius + p.minor_radius + p.plasma_wall_gap + depth
    r_tank_in = vv_in - p.blanket_inboard_thickness          # FLiBe boundary
    r_tank_out = vv_out + p.blanket_outboard_thickness
    r_tfs_out = r_tank_in - t_tank - p.tank_to_tf_shield_gap
    r_tfs_in = r_tfs_out - p.tf_shield_thickness
    r_tf_face = r_tfs_in - p.tf_shield_to_tf_gap

    # ---- PF coil pockets and ZrH2 shields (upper half, mirrored later) -------
    pf1_coil, pf1_case = _pf_stack(p, "pf1")
    c1 = pf1_case.bounds                                     # (rmin, zmin, rmax, zmax)
    pf1_shield = unary_union([
        box(r_tfs_in, c1[1] - t_sh, c1[2] + t_sh, c1[1]),   # plate under the coil
        box(c1[2], c1[1], c1[2] + t_sh, c1[3] + t_sh),      # plate on plasma side
    ])
    pf2_coil, pf2_case = _pf_stack(p, "pf2")
    pf3_coil, pf3_case = _pf_stack(p, "pf3")
    c23 = unary_union([pf2_case, pf3_case]).bounds

    # tank interior (FLiBe side of the tank wall), upper half
    r_p1 = pf1_shield.bounds[2] + g_sh + t_tank        # pocket wall next to PF1
    z_pb1 = pf1_shield.bounds[1] - g_sh - t_tank       # pocket floor under PF1
    r_p2 = c23[0] - t_sh - g_sh - t_tank               # pocket wall next to PF2/3
    z_pb2 = c23[1] - t_sh - g_sh - t_tank              # pocket floor under PF3
    r_c1 = r_p1 + p.tank_inboard_chamfer
    z_c1 = p.tank_top_z - p.tank_inboard_chamfer
    r_c2 = r_p2 - p.tank_outboard_chamfer
    z_c2 = p.tank_top_z - p.tank_outboard_chamfer

    wall_z, wall_dr = np.array(p.tank_outboard_wall).T
    r_ob_end = r_tank_out + np.interp(z_pb2, wall_z, wall_dr)
    outboard_wall = [(r_tank_out + dr, z) for z, dr in zip(wall_z, wall_dr) if z < z_pb2]
    tank_upper = Polygon(
        [(r_tank_in, 0.0), (r_tank_in, z_pb1), (r_p1, z_pb1), (r_p1, z_c1),
         (r_c1, p.tank_top_z), (r_c2, p.tank_top_z), (r_p2, z_c2), (r_p2, z_pb2),
         (r_ob_end, z_pb2)] + outboard_wall[::-1]
    )
    tank_interior = _clean(unary_union([tank_upper, _mirror(tank_upper)]))
    tank_outer = _clean(_mitre(tank_interior, t_tank))

    pf23_shield = unary_union([
        box(c23[0] - t_sh, c23[1] - t_sh, r_ob_end, c23[1]),        # floor plate
        box(c23[0] - t_sh, c23[1], c23[0], min(c23[3] + t_sh, z_c2)),  # wall plate
    ])

    # trim coils inside the tank
    trims = []
    for name, (r, z, w, h) in p.trim_coils.items():
        coil = _rect(r, z, w, h)
        trims.append((name, coil, _mitre(coil, p.trim_case_thickness)))

    # ---- blanket, tank, TF shield ----------------------------------------------
    trim_cases = [c for _, _, c in trims] + [_mirror(c) for _, _, c in trims]
    parts.append(Part("flibe_blanket", "flibe", tank_interior, [vv_envelope] + trim_cases))
    parts.append(Part("blanket_tank", "inconel718", tank_outer, [tank_interior]))
    z_tfs = pf1_shield.bounds[1] - g_sh
    parts.append(Part("tf_shield", "tih2", box(r_tfs_in, -z_tfs, r_tfs_out, z_tfs)))

    for side, f in (("upper", lambda g: g), ("lower", _mirror)):
        parts.append(Part(f"pf1_shield_{side}", "zrh2", _clean(f(pf1_shield))))
        parts.append(Part(f"pf23_shield_{side}", "zrh2", _clean(f(pf23_shield))))
        for name, coil, case in trims:
            parts.append(Part(f"{name}_{side}", "copper", f(coil)))
            parts.append(Part(f"{name}_case_{side}", "inconel718", f(case), [f(coil)]))

    # ---- superconducting coils --------------------------------------------------
    pf_names = ["pf1", "pf2", "pf3"] + (["pf4", "pf5"] if magnets else [])
    for side, f in (("upper", lambda g: g), ("lower", _mirror)):
        for name in pf_names:
            coil, case = _pf_stack(p, name)
            parts.append(Part(f"{name}_{side}", "magnet", f(coil)))
            parts.append(Part(f"{name}_case_{side}", "ss316l", f(case), [f(coil)]))

    if magnets:
        parts.append(Part("central_solenoid", "magnet",
                          box(p.cs_inner_radius, -p.cs_half_height,
                              p.cs_outer_radius, p.cs_half_height)))
        parts.append(tf_profile(p, r_tf_face))

    midplane = dict(
        tf_inner_leg=(r_tf_face - p.tf_thickness, r_tf_face),
        tf_shield=(r_tfs_in, r_tfs_out),
        tank_wall_inboard=(r_tank_in - t_tank, r_tank_in),
        blanket_inboard=(r_tank_in, vv_in),
        vacuum_vessel_inboard=(vv_in, vv_in + depth),
        plasma=(p.major_radius - p.minor_radius, p.major_radius + p.minor_radius),
        vacuum_vessel_outboard=(vv_out - depth, vv_out),
        blanket_outboard=(vv_out, r_tank_out),
        tank_wall_outboard=(r_tank_out, r_tank_out + t_tank),
    )
    return parts, midplane


def tf_profile(p: ARCParameters, r_face: float) -> Part:
    """D-shaped TF coil: straight inner leg at r_face plus a smoothed arch."""
    z_leg = p.tf_straight_leg_half_height
    # extra points below the leg end and below the midplane give vertical
    # tangents where the arch meets the straight leg and the mirrored half
    arch = np.array([(r_face, z_leg - 40.0), (r_face, z_leg)] + list(p.tf_inner_arch)
                    + [(p.tf_inner_arch[-2][0], -p.tf_inner_arch[-2][1])], float)
    tck, u = splprep(arch.T, s=len(arch) * 4.0)
    r, z = splev(np.linspace(u[1], u[-2], 240), tck)
    r = np.maximum(r, r_face)          # no dent inside the straight leg
    z = np.clip(z, 0.0, None)
    r[0], z[0], z[-1] = r_face, z_leg, 0.0
    upper = list(zip(r, z))
    lower = [(x, -y) for x, y in upper[::-1]][1:]
    d_inner = _clean(Polygon(upper + lower), 0.05)
    d_outer = _clean(d_inner.buffer(p.tf_thickness, quad_segs=p.quad_segs), 0.05)
    return Part("tf_coils", "magnet", d_outer, [d_inner], kind="tf")


# --------------------------------------------------------------------------
# 2D checks
# --------------------------------------------------------------------------
def check_profiles(parts: list[Part], tol: float = 1e-3) -> list[str]:
    """Pairwise overlap check of the R-Z footprints (cm^2)."""
    problems = []
    regions = [(pt.name, pt.region) for pt in parts]
    for i, (na, ga) in enumerate(regions):
        if not ga.is_valid:
            problems.append(f"{na}: invalid polygon")
        for nb, gb in regions[i + 1:]:
            if ga.intersects(gb):
                a = ga.intersection(gb).area
                if a > tol:
                    problems.append(f"{na} overlaps {nb} by {a:.3f} cm2")
    return problems


def clearance(parts: list[Part], name_a: str, name_b: str) -> float:
    ga = next(pt.region for pt in parts if pt.name == name_a)
    gb = next(pt.region for pt in parts if pt.name == name_b)
    return ga.distance(gb)


# --------------------------------------------------------------------------
# 3D
# --------------------------------------------------------------------------
def _revolve(poly: Polygon, angle: float, name: str) -> cq.Workplane:
    pts = [(x, z, "straight") for x, z in list(poly.exterior.coords)[:-1]]
    return paramak.revolved_shape(points=pts, rotation_angle=angle, name=name)


def _tf_coils(p: ARCParameters, part: Part, angle: float) -> cq.Workplane:
    outer = list(part.outer.exterior.coords)[:-1]
    inner = list(part.cuts[0].exterior.coords)[:-1]
    coil = (cq.Workplane("XZ").polyline(outer).close()
            .polyline(inner).close()
            .extrude(p.tf_toroidal_width / 2.0, both=True))
    # coils at 10 + 20k degrees: symmetric about 0 deg and any multiple of 10 deg,
    # so a sector with reflecting boundaries is valid for angle = 90, 180, ...
    pitch = 360.0 / p.n_tf_coils
    solids = []
    for k in range(p.n_tf_coils):
        solids.append(coil.rotate((0, 0, 0), (0, 0, 1), pitch / 2 + k * pitch).val())
    coils = cq.Workplane("XY").newObject([cq.Compound.makeCompound(solids)])
    if angle < 360.0:
        big = 2.0 * max(part.outer.bounds[2], part.outer.bounds[3])
        wedge = _revolve(box(0, -big, big, big), angle, "wedge")
        coils = coils.intersect(wedge)
    return coils


def build_assembly(p: ARCParameters, parts: list[Part], angle: float) -> Assembly:
    assembly = Assembly()
    for pt in parts:
        if pt.kind == "plasma":
            solid = paramak.plasma_simplified(
                major_radius=p.major_radius, minor_radius=p.minor_radius,
                elongation=p.elongation, triangularity=p.triangularity,
                rotation_angle=angle)
        elif pt.kind == "tf":
            solid = _tf_coils(p, pt, angle)
        else:
            solid = _revolve(pt.outer, angle, pt.name)
            for c in pt.cuts:
                solid = solid.cut(_revolve(c, angle, pt.name + "_cut"))
        assembly.add(solid, name=pt.name, color=cq.Color(*COLORS[pt.material]),
                     material=cq.Material(pt.material))
    return assembly


def volumes_m3(assembly: Assembly, angle: float) -> dict[str, float]:
    """Volume of each part scaled to the full 360 degree machine (m^3)."""
    out = {}
    for shape, name, *_ in assembly:
        leaf = name.split("/")[-1]
        v = sum(s.Volume() for s in shape.Solids()) if hasattr(shape, "Solids") else 0.0
        out[leaf] = v * 1e-6 * 360.0 / angle
    return out


# --------------------------------------------------------------------------
# Output helpers
# --------------------------------------------------------------------------
def plot_cross_section(parts: list[Part], filename: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import PathPatch
    from matplotlib.path import Path

    def patch(geom, color):
        polys = [geom] if geom.geom_type == "Polygon" else list(geom.geoms)
        verts, codes = [], []
        for poly in polys:
            for ring in [poly.exterior, *poly.interiors]:
                xy = np.asarray(ring.coords)
                verts += xy.tolist()
                codes += [Path.MOVETO] + [Path.LINETO] * (len(xy) - 2) + [Path.CLOSEPOLY]
        return PathPatch(Path(verts, codes), facecolor=color, edgecolor="k", lw=0.3)

    fig, ax = plt.subplots(figsize=(7, 9))
    seen = set()
    for pt in parts:
        ax.add_patch(patch(pt.region, COLORS[pt.material]))
        if pt.material not in seen:
            ax.plot([], [], "s", color=COLORS[pt.material], label=pt.material, ms=9)
            seen.add(pt.material)
    ax.set_xlim(0, 720); ax.set_ylim(-500, 500); ax.set_aspect("equal")
    ax.set_xlabel("R [cm]"); ax.set_ylabel("Z [cm]")
    ax.set_title("ARC V0 (2018) - poloidal slice - Paramak model")
    ax.legend(loc="upper right", fontsize=7, framealpha=0.9)
    ax.grid(alpha=0.25)
    fig.tight_layout(); fig.savefig(filename, dpi=200); plt.close(fig)


# published volumes (m^3) for comparison: Kuang18 Table 7 / Aimetta22 Table II (CAD)
REFERENCE_VOLUMES = {
    "plasma": (137, 147),
    "first_wall": (0.35, 0.32),
    "inner_vacuum_vessel": (3.50, 3.27),
    "flibe_cooling_channel": (7.04, 6.58),
    "beryllium_multiplier": (3.55, 3.31),
    "outer_vacuum_vessel": (10.7, 10.0),
    "flibe_blanket": (241, 304),
    "pf_shielding": (49.2, 43.5),
}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--angle", type=float, default=90.0,
                    help="toroidal extent in degrees (default 90, use 360 for full torus)")
    ap.add_argument("--step", default="arc_reactor.step", help="STEP output ('' to skip)")
    ap.add_argument("--h5m", default="", help="DAGMC h5m output (needs cad_to_dagmc)")
    ap.add_argument("--png", default="arc_cross_section.png", help="R-Z plot ('' to skip)")
    ap.add_argument("--pdf", default="arc_cross_section.pdf", help="R-Z plot ('' to skip)")
    ap.add_argument("--no-magnets", action="store_true",
                    help="leave out TF coils, central solenoid and PF4/PF5")
    ap.add_argument("--mesh-tolerance", type=float, default=0.1,
                    help="surface faceting tolerance for the h5m file [cm]")
    args = ap.parse_args()

    p = ARCParameters()
    parts, midplane = build_profiles(p, magnets=not args.no_magnets)

    print("Midplane radial build [cm]:")
    for k, (a, b) in midplane.items():
        print(f"  {k:24s} {a:8.2f} -> {b:8.2f}   ({b - a:6.2f})")

    problems = check_profiles(parts)
    if problems:
        raise SystemExit("Geometry overlaps found:\n  " + "\n  ".join(problems))
    print(f"2D check: {len(parts)} profiles, no overlaps")
    if not args.no_magnets:
        for a, b in [("tf_coils", "pf4_case_upper"), ("tf_coils", "pf5_case_upper"),
                     ("tf_coils", "pf23_shield_upper"), ("tf_coils", "blanket_tank"),
                     ("tf_coils", "tf_shield"), ("tf_coils", "central_solenoid")]:
            print(f"  clearance {a} - {b}: {clearance(parts, a, b):.1f} cm")

    if args.png:
        plot_cross_section(parts, args.png)
        print(f"wrote {args.png}")

    if args.pdf:
        plot_cross_section(parts, args.pdf)
        print(f"wrote {args.pdf}")

    assembly = build_assembly(p, parts, args.angle)
    vols = volumes_m3(assembly, args.angle)
    vols["pf_shielding"] = sum(v for k, v in vols.items() if "shield_" in k)
    print("\nVolumes scaled to 360 deg [m^3]      model   Kuang18  Aimetta22")
    for k, (ref_k, ref_a) in REFERENCE_VOLUMES.items():
        print(f"  {k:28s} {vols[k]:9.2f} {ref_k:9.2f} {ref_a:9.2f}")

    if args.step:
        assembly.export(args.step)
        print(f"wrote {args.step}")

    if args.h5m:
        from cad_to_dagmc import CadToDagmc
        model = CadToDagmc()
        model.add_cadquery_object(assembly, material_tags="assembly_materials")
        model.export_dagmc_h5m_file(filename=args.h5m, tolerance=args.mesh_tolerance,
                                    angular_tolerance=0.2)
        print(f"wrote {args.h5m}")


if __name__ == "__main__":
    main()
