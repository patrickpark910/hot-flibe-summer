#!/usr/bin/env python3
"""
ARC V3A fusion power plant: parametric CAD model built with Paramak (>= 0.10)
==============================================================================

Builds a neutronics-ready CAD model of the ARC V3A design (Commonwealth Fusion
Systems, 2026):

    plasma (last closed flux surface of the published V3A equilibrium)
    vacuum vessel: 5 mm W first wall / 1 cm V inner wall / 4.5 cm FLiBe
        cooling channels / 3 cm V outer wall, with long-leg X-point-target
        divertors (double null)
    FLiBe immersion blanket inside a 2 cm blanket tank
    tungsten-carbide neutron shield on the inboard side of the tank
    central solenoid (CS1-CS3, upper/lower) with end caps and spacers
    PF1-PF5 coils (upper/lower)
    18 TF coils (case outline from Fig. 2), wedged inner legs

Every component is a 2D R-Z profile (shapely) turned into 3D with
paramak.revolved_shape and collected in a paramak Assembly. Adjacent layers
share exactly the same profile, so their surfaces coincide (good for DAGMC
imprinting) and nothing overlaps.

Units: cm in the model (paramak / OpenMC convention); the geometry tables
below are in metres, as published.

Where the numbers come from
---------------------------
Primary reference
[Hillesheim26]  J.C. Hillesheim et al., "Overview of the physics basis for the
                ARC fusion power plant", J. Plasma Phys. 92 (2026) E69,
                doi:10.1017/S0022377826101706.
                Table 2: R0 = 4.62 m, a = 1.18 m, kappa_sep = 1.8,
                delta_sep = 0.65, B0 = 11.4 T, Ip = 12 MA, Vp = 189 m3,
                Pfus = 1.13 GW. Table 4: 4e20 neutrons/s. Sec. 3: 18 TF coils,
                vessel build 5 mm W / 1 cm V / 4.5 cm FLiBe channels / 3 cm V,
                tungsten-carbide dedicated neutron shielding, no in-vessel
                coils. Sec. 4.1.1: 2 cm gap from the outer-midplane separatrix
                to the wall. Sec. 4.3: FLiBe at ~900 K.
                Fig. 2 (cross-section) was retraced for every boundary below.
[V3Aassets]     J.C. Hillesheim et al., "ARC V3A digital assets" (2026),
                doi:10.5281/zenodo.19498373, CC-BY-4.0: device_V3A.json
                (reduced device description, the paper's Fig. 12) and
                geqdsk-ARCv3a (FreeGS equilibrium used for Fig. 2).

How the boundaries were obtained
--------------------------------
* Fig. 2 was calibrated on its axes and traced. Its coil boxes, vessel, tank
  and shield agree with device_V3A.json to a median of 0.8-1.6 cm, so the
  published vertices are used for those components (LIMITER_M, TANK_INNER_M,
  WC_SHIELD_M, COIL_BOXES_M). The vessel layers in device_V3A.json are exact
  mitred offsets of the limiter, so they are generated from LIMITER_M and the
  layer thicknesses.
* The plasma is the last closed flux surface of geqdsk-ARCv3a (upper half,
  mirrored; the equilibrium is up-down symmetric).
* The TF case outline and the CS end caps/spacers are only in Fig. 2 and come
  from the trace (TF_OUTER_M, TF_BORE_M). The drawn TF outline sits ~1.2 cm
  above the midplane while the coils are symmetric, so the upper half and the
  mirrored lower half of the trace were averaged (they differ by <= 2.8 cm).

Things that are modelling choices rather than published values are marked
ASSUMPTION. The paper stresses that the V3A vessel, tank and shield shapes are
nominal 2D concepts (no ports, pumps, antennas or toroidal breaks).

Requirements:  pip install paramak shapely matplotlib
Optional:      pip install cad_to_dagmc      (for --h5m)

Examples:
    python arc_v3a_paramak_model.py                    # 90 deg sector -> STEP + PNG
    python arc_v3a_paramak_model.py --angle 360        # full torus
    python arc_v3a_paramak_model.py --h5m arc_v3a.h5m  # also write a DAGMC file
    python arc_v3a_paramak_model.py --no-magnets       # plasma, vessel, blanket, shield, tank
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass, field

import cadquery as cq
import numpy as np
import paramak
from paramak.assemblies.assembly import Assembly
from shapely.geometry import LineString, Polygon, box
from shapely.ops import unary_union

M = 100.0  # metres -> cm

# --------------------------------------------------------------------------
# Geometry tables (metres). Up-down symmetric: upper half only, listed from
# one midplane crossing round to the other; the lower half is mirrored.
# --------------------------------------------------------------------------
# plasma-facing surface of the vacuum vessel ("limiter", device_V3A.json)
LIMITER_M = (
    (3.42, 0.00), (3.42, 0.85), (3.68, 2.10), (3.46, 2.30), (3.46, 2.33),
    (3.76, 2.48), (4.36, 3.48), (4.77, 4.09), (5.31, 4.09), (5.31, 3.35),
    (4.94, 3.52), (4.18, 2.35), (4.77, 1.84), (5.23, 1.30), (5.67, 0.75),
    (5.82, 0.30), (5.82, 0.00),
)
# FLiBe-side surface of the blanket tank ("Blanket" outline_inner)
TANK_INNER_M = (
    (2.48, 0.00), (2.48, 1.35), (2.61, 2.37), (2.92, 3.12), (3.51, 3.95),
    (4.11, 4.37), (5.05, 4.68), (5.65, 4.62), (5.65, 2.50), (5.65, 1.80),
    (6.45, 1.10), (7.20, 1.10), (7.20, 0.00),
)
# tungsten-carbide shield ("Shield"); its outer edge is the tank's inner wall
WC_SHIELD_M = (
    (2.48, 0.00), (2.48, 1.35), (2.61, 2.37), (2.92, 3.12), (3.51, 3.95),
    (4.11, 4.37), (3.22, 3.30), (2.77, 2.35), (3.12, 0.75), (3.12, 0.00),
)
# coil boxes (R_min, Z_min, R_max, Z_max) of the upper coils: filament extents
# +/- filament radius in device_V3A.json ("coil locations are correct")
COIL_BOXES_M = {
    "cs1": (0.7269, 0.0567, 1.4831, 1.4433),
    "cs2": (0.7269, 1.5567, 1.4831, 2.2433),
    "cs3": (0.7269, 2.3567, 1.4831, 3.0433),
    "pf1": (1.1050, 3.6050, 1.6950, 4.1950),
    "pf2": (1.5552, 4.4052, 2.2448, 5.0948),
    "pf3": (5.9556, 2.7056, 6.4444, 3.1944),
    "pf4": (5.9556, 2.0056, 6.4444, 2.4944),
    "pf5": (6.7554, 1.4054, 7.1446, 1.7946),
}
CS_COLUMN_TOP_M = 3.40      # top of the CS end cap (and of the TF straight block), Fig. 2
# last closed flux surface, geqdsk-ARCv3a (FreeGS), outboard -> over the top -> inboard
LCFS_M = (
    (5.8034, 0.0172), (5.8003, 0.0858), (5.7930, 0.1541), (5.7819, 0.2218),
    (5.7668, 0.2889), (5.7481, 0.3552), (5.7260, 0.4205), (5.7006, 0.4848),
    (5.6722, 0.5482), (5.6410, 0.6106), (5.6069, 0.6723), (5.5704, 0.7332),
    (5.5312, 0.7937), (5.4895, 0.8539), (5.4453, 0.9140), (5.3984, 0.9745),
    (5.3487, 1.0355), (5.2960, 1.0977), (5.2399, 1.1612), (5.1800, 1.2267),
    (5.1159, 1.2949), (5.0467, 1.3662), (4.9717, 1.4410), (4.8898, 1.5210),
    (4.7996, 1.6055), (4.6996, 1.6965), (4.5879, 1.7925), (4.4629, 1.8927),
    (4.3227, 1.9957), (4.1664, 2.0982), (3.9931, 2.1975), (3.8581, 2.1488),
    (3.7857, 1.9642), (3.7246, 1.7946), (3.6734, 1.6372), (3.6312, 1.4899),
    (3.5956, 1.3532), (3.5663, 1.2250), (3.5423, 1.1045), (3.5217, 0.9918),
    (3.5049, 0.8852), (3.4910, 0.7840), (3.4791, 0.6879), (3.4695, 0.5958),
    (3.4616, 0.5072), (3.4552, 0.4215), (3.4502, 0.3382), (3.4464, 0.2569),
    (3.4437, 0.1770), (3.4419, 0.0980), (3.4412, 0.0196),
)
# TF case outline traced from Fig. 2 (outboard midplane -> over the top ->
# inboard midplane). The straight inboard face is set from the CS (see params).
TF_OUTER_M = (
    (8.903, 0.000), (8.902, 0.262), (8.886, 0.592), (8.840, 1.058), (8.803, 1.329),
    (8.755, 1.605), (8.704, 1.850), (8.635, 2.136), (8.574, 2.353), (8.493, 2.612),
    (8.384, 2.912), (8.307, 3.102), (8.185, 3.371), (8.068, 3.604), (7.937, 3.836),
    (7.817, 4.030), (7.658, 4.258), (7.573, 4.371), (7.390, 4.592), (7.257, 4.732),
    (7.081, 4.902), (6.931, 5.030), (6.774, 5.151), (6.567, 5.287), (6.437, 5.362),
    (6.259, 5.450), (6.075, 5.526), (5.886, 5.589), (5.741, 5.626), (5.546, 5.662),
    (5.348, 5.682), (5.150, 5.688), (4.853, 5.672), (4.656, 5.648), (4.461, 5.615),
    (4.220, 5.557), (4.031, 5.501), (3.799, 5.417), (3.620, 5.336), (3.401, 5.229),
    (3.232, 5.129), (3.031, 4.993), (2.876, 4.876), (2.658, 4.687), (2.490, 4.516),
    (2.331, 4.340), (2.217, 4.189), (2.109, 4.036), (2.010, 3.878), (1.899, 3.675),
    (1.800, 3.469), (1.732, 3.301), (1.672, 3.131), (1.611, 2.917), (1.564, 2.702),
    (1.533, 2.532), (1.509, 2.320), (1.502, 2.149), (1.498, 1.513), (1.501, 0.159),
    (1.498, 0.000),
)
TF_BORE_M = (
    (7.711, 0.000), (7.704, 0.089), (7.704, 0.336), (7.688, 0.633), (7.665, 0.891),
    (7.605, 1.361), (7.570, 1.571), (7.519, 1.818), (7.419, 2.230), (7.291, 2.643),
    (7.203, 2.878), (7.131, 3.051), (6.980, 3.370), (6.860, 3.586), (6.795, 3.694),
    (6.674, 3.870), (6.597, 3.975), (6.485, 4.109), (6.365, 4.239), (6.270, 4.330),
    (6.135, 4.442), (6.028, 4.519), (5.880, 4.610), (5.803, 4.650), (5.685, 4.700),
    (5.565, 4.742), (5.442, 4.771), (5.317, 4.788), (5.192, 4.793), (5.025, 4.787),
    (4.858, 4.768), (4.694, 4.737), (4.532, 4.692), (4.335, 4.624), (4.182, 4.555),
    (4.034, 4.476), (3.891, 4.389), (3.755, 4.294), (3.625, 4.191), (3.502, 4.079),
    (3.358, 3.933), (3.250, 3.810), (3.102, 3.619), (2.992, 3.454), (2.911, 3.320),
    (2.838, 3.183), (2.738, 2.978), (2.681, 2.840), (2.618, 2.668), (2.534, 2.397),
    (2.486, 2.196), (2.450, 1.998), (2.411, 1.711), (2.392, 1.375), (2.392, 0.000),
)


# --------------------------------------------------------------------------
# Parameters (cm)
# --------------------------------------------------------------------------
@dataclass
class ARCParameters:
    """ARC V3A build. Thicknesses in cm."""
    # vacuum vessel, plasma side outwards [Hillesheim26 Sec. 3]
    first_wall_thickness: float = 0.5        # tungsten
    vv_inner_wall_thickness: float = 1.0     # vanadium
    vv_channel_thickness: float = 4.5        # FLiBe cooling channels
    vv_outer_wall_thickness: float = 3.0     # vanadium
    # blanket tank wall [V3Aassets "Blanket" annulus, 2 cm]
    tank_wall_thickness: float = 2.0
    # ASSUMPTION: Fig. 2 draws the TF case touching the CS; leave a 1 cm gap
    cs_to_tf_gap: float = 1.0
    # TF coils [Hillesheim26 Sec. 3: 18 coils]
    n_tf_coils: int = 18
    # ASSUMPTION: wedged inner legs filling +/- 9.5 deg of each 20 deg pitch;
    # the outer leg keeps the toroidal width the wedge has at the inboard
    # bore face (about 0.79 m). Set tf_outer_leg_width to override.
    tf_wedge_half_angle: float = 9.5         # degrees
    tf_outer_leg_width: float | None = None  # cm
    # ASSUMPTION: CS end caps and inter-module spacers (light blue in Fig. 2)
    # are structure, given the coil stack's radial extent
    include_cs_structure: bool = True
    quad_segs: int = 8


# Material tags (DAGMC "mat:" names) and densities in g/cm3
MATERIALS = {
    "plasma": 5e-8,            # Bae 2022 convention
    "tungsten": 19.3,          # first wall armour [Hillesheim26]
    "vanadium": 6.11,          # VV walls [Hillesheim26]; pure V (placeholder for a V alloy)
    "flibe": 1.94,             # 1940 kg/m3 [Badalassi 2023 Table II]; Li-6 enrichment is not
                               # given for V3A (90 % in Bae 2022 / Badalassi 2023)
    "tungsten_carbide": 15.63, # dedicated neutron shielding [Hillesheim26]
    "ss316l": 8.0,             # ASSUMPTION: tank and CS structure; device_V3A.json gives the
                               # tank a resistivity of 7.4e-7 Ohm m, typical of 316 steel
    "magnet": 8.48,            # ASSUMPTION: homogenised CS and PF coils, Bae 2022 mix
    "tf_magnet": 8.48,         # ASSUMPTION: homogenised TF coil + case, same placeholder;
                               # a separate tag so the case-heavy TF can get its own mix
}
# nominal operating temperatures given in [Hillesheim26] (K)
OPERATING_TEMPERATURES_K = {
    "flibe": 900.0,            # Sec. 4.3, "FLiBe temperature around 900 K"
    "vanadium": 873.15,        # Sec. 3, ~600 C assumed for vessel conductivities
    "tungsten": 873.15,
}
# plasma and source parameters [Hillesheim26 Tables 2 and 4]
PLASMA_PARAMETERS = {
    "major_radius_m": 4.62, "minor_radius_m": 1.18, "elongation_sep": 1.8,
    "triangularity_sep": 0.65, "B0_T": 11.4, "Ip_MA": 12.0, "volume_m3": 189.0,
    "fusion_power_MW": 1133.0, "neutron_rate_per_s": 4.0e20,
}

COLORS = {
    "plasma": (0.9, 0.35, 0.6), "tungsten": (0.2, 0.2, 0.2),
    "vanadium": (0.6, 0.6, 0.65), "flibe": (0.5, 0.75, 0.5),
    "tungsten_carbide": (0.0, 0.39, 0.0), "ss316l": (0.45, 0.45, 0.5),
    "magnet": (0.12, 0.56, 1.0), "tf_magnet": (0.78, 0.78, 0.78),
}


# --------------------------------------------------------------------------
# 2D helpers
# --------------------------------------------------------------------------
def _mirror_closed(upper_m) -> Polygon:
    """Closed polygon in cm from an upper-half point list in metres."""
    pts = [(r * M, z * M) for r, z in upper_m]
    ends_on_midplane = abs(pts[0][1]) < 1e-9 and abs(pts[-1][1]) < 1e-9
    tail = pts[-2:0:-1] if ends_on_midplane else pts[::-1]
    lower = [(r, -z) for r, z in tail]
    return _clean(Polygon(pts + lower))


def _clean(geom) -> Polygon:
    geom = geom.buffer(0)
    if geom.geom_type != "Polygon":
        raise ValueError(f"expected a single polygon, got {geom.geom_type}")
    return geom


def _mitre(geom, d):
    return geom.buffer(d, join_style="mitre", mitre_limit=10.0)


def _box_m(b, mirror=False):
    r0, z0, r1, z1 = (v * M for v in b)
    return box(r0, -z1, r1, -z0) if mirror else box(r0, z0, r1, z1)


@dataclass
class Part:
    """One volume: 3D = revolve(outer) - revolve(each cut)."""
    name: str
    material: str
    outer: Polygon
    cuts: list = field(default_factory=list)
    kind: str = "revolve"            # "revolve" or "tf"

    @property
    def region(self):
        geom = self.outer
        for c in self.cuts:
            geom = geom.difference(c)
        return geom


def tf_profile(p: ARCParameters) -> Part:
    """TF case: traced D-shaped ring plus the straight inner-leg block."""
    r_in = COIL_BOXES_M["cs1"][2] * M + p.cs_to_tf_gap
    z_blk = CS_COLUMN_TOP_M * M
    outer = _mirror_closed(TF_OUTER_M)
    outer = outer.union(box(r_in, -z_blk, 190.0, z_blk))           # straight block, Fig. 2
    outer = _clean(outer.intersection(box(r_in, -1000, 1000, 1000)))  # flat inboard face
    bore = _mirror_closed(TF_BORE_M)
    return Part("tf_coils", "tf_magnet", outer, [bore], kind="tf")


def build_profiles(p: ARCParameters, magnets: bool = True) -> list[Part]:
    parts: list[Part] = [Part("plasma", "plasma", _mirror_closed(LCFS_M))]

    # ---- vacuum vessel: mitred offsets of the limiter ----------------------
    limiter = _mirror_closed(LIMITER_M)
    layers = [
        ("first_wall", "tungsten", p.first_wall_thickness),
        ("vv_inner_wall", "vanadium", p.vv_inner_wall_thickness),
        ("vv_cooling_channel", "flibe", p.vv_channel_thickness),
        ("vv_outer_wall", "vanadium", p.vv_outer_wall_thickness),
    ]
    inner, depth = limiter, 0.0
    for name, mat, t in layers:
        depth += t
        outer = _clean(_mitre(limiter, depth))
        parts.append(Part(name, mat, outer, [inner]))
        inner = outer
    vv_envelope = inner

    # ---- blanket, shield, tank ---------------------------------------------
    tank_interior = _mirror_closed(TANK_INNER_M)
    shield = _mirror_closed(WC_SHIELD_M)
    parts.append(Part("flibe_blanket", "flibe", tank_interior, [vv_envelope, shield]))
    parts.append(Part("wc_shield", "tungsten_carbide", shield))
    parts.append(Part("blanket_tank", "ss316l",
                      _clean(_mitre(tank_interior, p.tank_wall_thickness)), [tank_interior]))

    if not magnets:
        return parts

    # ---- coils ----------------------------------------------------------------
    coil_boxes = []
    for side, mirror in (("upper", False), ("lower", True)):
        for name, b in COIL_BOXES_M.items():
            g = _box_m(b, mirror)
            coil_boxes.append(g)
            parts.append(Part(f"{name}_{side}", "magnet", g))
    if p.include_cs_structure:
        r0, r1 = COIL_BOXES_M["cs1"][0] * M, COIL_BOXES_M["cs1"][2] * M
        column = box(r0, -CS_COLUMN_TOP_M * M, r1, CS_COLUMN_TOP_M * M)
        structure = column.difference(unary_union(coil_boxes))
        pieces = [structure] if structure.geom_type == "Polygon" else list(structure.geoms)
        for i, g in enumerate(sorted(pieces, key=lambda q: -q.centroid.y)):
            parts.append(Part(f"cs_structure_{i + 1}", "ss316l", g))
    parts.append(tf_profile(p))
    return parts


def midplane_build(parts: list[Part]) -> list[tuple[float, float, str]]:
    """(r_in, r_out, name) of every part crossing Z = 0, sorted by radius."""
    line = LineString([(0.0, 0.0), (1000.0, 0.0)])
    rows = []
    for pt in parts:
        hit = pt.region.intersection(line)
        segs = [hit] if hit.geom_type == "LineString" else list(getattr(hit, "geoms", []))
        for s in segs:
            if s.geom_type == "LineString" and s.length > 1e-6:
                xs = [c[0] for c in s.coords]
                rows.append((min(xs), max(xs), pt.name))
    return sorted(rows)


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
    """18 coils: the revolved case profile cut to a wedge + constant-width footprint."""
    ring = _revolve(part.outer, 360.0, "tf_outer").cut(_revolve(part.cuts[0], 360.0, "tf_bore"))
    alpha = math.radians(p.tf_wedge_half_angle)
    r_bore = min(x for x, z in part.cuts[0].exterior.coords if abs(z) < 1.0)
    half_w = p.tf_outer_leg_width / 2.0 if p.tf_outer_leg_width else r_bore * math.sin(alpha)
    r_w = half_w / math.sin(alpha)
    far = 2.0 * part.outer.bounds[2]
    h = 2.0 * max(abs(part.outer.bounds[1]), part.outer.bounds[3])
    footprint = [(0.0, 0.0), (r_w * math.cos(alpha), -half_w), (far, -half_w),
                 (far, half_w), (r_w * math.cos(alpha), half_w)]
    prism = cq.Workplane("XY").workplane(offset=-h).polyline(footprint).close().extrude(2 * h)
    coil = ring.intersect(prism).val()
    pitch = 360.0 / p.n_tf_coils
    # coils at 10 + 20k deg: symmetric about 0 deg and any multiple of 10 deg,
    # so a sector with reflecting boundaries is valid for angle = 90, 180, ...
    solids = [coil.rotate(cq.Vector(0, 0, 0), cq.Vector(0, 0, 1), pitch / 2 + k * pitch)
              for k in range(p.n_tf_coils)]
    coils = cq.Workplane("XY").newObject([cq.Compound.makeCompound(solids)])
    if angle < 360.0:
        wedge = _revolve(box(0, -2 * h, far, 2 * h), angle, "wedge")
        coils = coils.intersect(wedge)
    return coils


def build_assembly(p: ARCParameters, parts: list[Part], angle: float) -> Assembly:
    assembly = Assembly()
    for pt in parts:
        if pt.kind == "tf":
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
        v = sum(s.Volume() for s in shape.Solids()) if hasattr(shape, "Solids") else 0.0
        out[name.split("/")[-1]] = v * 1e-6 * 360.0 / angle
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
                xy = np.asarray(ring.coords) / M
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
    ax.set_xlim(0, 9.5); ax.set_ylim(-6.2, 6.2); ax.set_aspect("equal")
    ax.set_xlabel("R [m]"); ax.set_ylabel("Z [m]")
    ax.set_title("ARC V3A (2026) - poloidal slice - Paramak model")
    ax.legend(loc="upper right", fontsize=7, framealpha=0.9)
    ax.grid(alpha=0.25)
    fig.tight_layout(); fig.savefig(filename, dpi=200); plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--angle", type=float, default=90.0,
                    help="toroidal extent in degrees (default 90, use 360 for full torus)")
    ap.add_argument("--step", default="arc_v3a.step", help="STEP output ('' to skip)")
    ap.add_argument("--h5m", default="", help="DAGMC h5m output (needs cad_to_dagmc)")
    ap.add_argument("--png", default="arc_v3a_cross_section.png", help="R-Z plot ('' to skip)")
    ap.add_argument("--pdf", default="arc_v3a_cross_section.pdf", help="R-Z plot ('' to skip)")
    ap.add_argument("--no-magnets", action="store_true",
                    help="leave out the TF coils, central solenoid and PF coils")
    ap.add_argument("--mesh-tolerance", type=float, default=0.1,
                    help="surface faceting tolerance for the h5m file [cm]")
    args = ap.parse_args()

    p = ARCParameters()
    parts = build_profiles(p, magnets=not args.no_magnets)

    print("Midplane radial build [cm]:")
    for a, b, name in midplane_build(parts):
        print(f"  {name:22s} {a:8.2f} -> {b:8.2f}   ({b - a:6.2f})")

    problems = check_profiles(parts)
    if problems:
        raise SystemExit("Geometry overlaps found:\n  " + "\n  ".join(problems))
    print(f"2D check: {len(parts)} profiles, no overlaps")
    print(f"  plasma - first wall gap: {clearance(parts, 'plasma', 'first_wall'):.2f} cm")
    if not args.no_magnets:
        for a, b in [("tf_coils", "blanket_tank"), ("tf_coils", "pf1_upper"),
                     ("tf_coils", "pf2_upper"), ("tf_coils", "cs1_upper"),
                     ("blanket_tank", "pf3_upper"), ("blanket_tank", "pf5_upper")]:
            print(f"  clearance {a} - {b}: {clearance(parts, a, b):.1f} cm")

    if args.png:
        plot_cross_section(parts, args.png)
        print(f"wrote {args.png}")

    if args.pdf:
        plot_cross_section(parts, args.pdf)
        print(f"wrote {args.pdf}")

    assembly = build_assembly(p, parts, args.angle)
    vols = volumes_m3(assembly, args.angle)
    print("\nVolumes scaled to 360 deg [m^3]")
    for name in ["plasma", "first_wall", "vv_inner_wall", "vv_cooling_channel",
                 "vv_outer_wall", "flibe_blanket", "wc_shield", "blanket_tank", "tf_coils"]:
        if name in vols:
            ref = (f"   (Hillesheim26 Table 2: {PLASMA_PARAMETERS['volume_m3']:.0f})"
                   if name == "plasma" else "")
            print(f"  {name:22s} {vols[name]:9.2f}{ref}")
    print(f"  {'all FLiBe':22s} {vols['flibe_blanket'] + vols['vv_cooling_channel']:9.2f}")

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
