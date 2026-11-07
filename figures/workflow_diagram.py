#!/usr/bin/env python3
"""
OpenMC -> OpenFOAM workflow diagram (ARC V3A FLiBe blanket), publication figure
===============================================================================

    .venv/bin/python figures/workflow_diagram.py

Writes figures/workflow_openmc_openfoam.{pdf,svg,png} at 183 mm (7.2 in) wide,
which fits a full-width figure in Elsevier, IEEE and IOP journals. Fonts are
embedded as TrueType (pdf.fonttype 42) so the text stays editable.

Box colours follow the draw.io palette used in FERMI-style workflow figures:
yellow = design input, blue = code, red = metric, green = file/data. Bold
arrows are data that cross from one domain to another (the shared geometry
going to each physics, and the one-way OpenMC -> OpenFOAM coupling); thin
arrows are inputs and outputs within a domain.

Layout is in millimetres with y pointing down. Every box's text is measured
after drawing; the script warns if any text overflows its box.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402
from matplotlib.path import Path as MPath  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = HERE / "workflow_openmc_openfoam"

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------
W, H = 183.0, 134.0          # figure size, mm
MM = 1 / 25.4
PT = 25.4 / 72               # 1 pt in mm

ARIAL = ["/mnt/c/Windows/Fonts/arial.ttf", "/mnt/c/Windows/Fonts/arialbd.ttf",
         "/mnt/c/Windows/Fonts/ariali.ttf", "/mnt/c/Windows/Fonts/arialbi.ttf"]
if all(os.path.exists(f) for f in ARIAL):
    for f in ARIAL:
        font_manager.fontManager.addfont(f)
    FONT = "Arial"
else:                         # bundled fallback
    FONT = "DejaVu Sans"

plt.rcParams.update({
    "font.family": FONT,
    "mathtext.fontset": "custom",
    "mathtext.rm": FONT,
    "mathtext.it": f"{FONT}:italic",
    "mathtext.bf": f"{FONT}:bold",
    "pdf.fonttype": 42,
    "svg.fonttype": "none",
})

TITLE_PT, BODY_PT, PANEL_PT, LEGEND_PT = 7.0, 6.0, 11.0, 6.0
LINE = 1.22                   # line spacing factor
INK = "#1f1f1f"
KINDS = {                     # face, edge
    "design": ("#fff2cc", "#d6b656"),
    "code":   ("#dae8fc", "#6c8ebf"),
    "metric": ("#f8cecc", "#b85450"),
    "data":   ("#d5e8d4", "#82b366"),
}
LEGEND = [("design", "Design"), ("code", "Code"), ("metric", "Metric"), ("data", "File/Data")]
PANEL_FACE, PANEL_EDGE = "#f4f4f4", "#9a9a9a"
THIN, BOLD = 0.6, 1.5         # arrow line widths, pt


@dataclass
class Box:
    kind: str
    x: float
    yc: float                 # vertical centre (rows are centre-aligned)
    w: float
    title: str
    lines: list[str] = field(default_factory=list)
    h: float = 0.0

    def __post_init__(self):
        if not self.h:
            n_t = self.title.count("\n") + 1
            body = n_t * TITLE_PT + len(self.lines) * BODY_PT
            self.h = max(9.0, body * LINE * PT + 3.0)

    @property
    def y(self):
        return self.yc - self.h / 2

    def top(self, f=0.5):
        return (self.x + f * self.w, self.y)

    def bottom(self, f=0.5):
        return (self.x + f * self.w, self.y + self.h)

    def left(self, f=0.5):
        return (self.x, self.y + f * self.h)

    def right(self, f=0.5):
        return (self.x + self.w, self.y + f * self.h)

    def at_x(self, x, side="top"):
        """Point on the top/bottom edge at absolute x."""
        return (x, self.y if side == "top" else self.y + self.h)


# ---------------------------------------------------------------------------
# Layout (mm, y down)
# ---------------------------------------------------------------------------
TOP = 9.0                                   # centre of the geometry strip
ROW = {1: 27.0, 2: 41.0, 3: 55.0, 4: 69.5, 5: 84.5, 6: 100.0, 7: 115.5}

PANEL_N = (3.0, 18.0, 89.0, 85.0)           # x, y, w, h
PANEL_T = (95.0, 18.0, 85.0, 112.0)

c1, c2, c3 = (6.0, 31.0), (39.5, 22.0), (64.0, 25.0)       # neutronics columns
c4, c5, c6 = (98.0, 22.0), (125.0, 25.0), (152.0, 25.0)    # thermal-hydraulics columns

B = {}
# shared geometry (above the panels)
B["design"] = Box("design", 6.0, TOP, 52.0, "ARC V3A reactor design",
                  ["plasma boundary, radial build, coil set"])
B["paramak"] = Box("code", c3[0], TOP, c3[1], "Paramak", ["2D profiles → 3D CAD"])
B["rz"] = Box("data", c5[0], TOP, c5[1], "R–Z profiles", ["2D polygons (shapely)"])

# neutronics
B["plasma"] = Box("design", c1[0], ROW[1], c1[1], "Plasma profiles", ["V3A, 4.0×10²⁰ n/s"])
B["ops"] = Box("code", c1[0], ROW[2], c1[1], "openmc-plasma-source", ["tokamak source"])
B["src"] = Box("data", c1[0], ROW[3], c1[1], "Neutron source", ["(E, R, Z) distribution"])
B["cad"] = Box("data", c3[0], ROW[1], c3[1], "3D CAD model", ["(.step)"])
B["c2d"] = Box("code", c3[0], ROW[2], c3[1], "cad_to_dagmc", ["surface faceting"])
B["h5m"] = Box("data", c3[0], ROW[3], c3[1], "DAGMC geometry", ["(.h5m, 10° sector)"])
B["mat"] = Box("design", c2[0], ROW[3], c2[1], "Materials",
               ["FLiBe (90 % ⁶Li),", "W, V-4Cr-4Ti, WC"])
B["omc"] = Box("code", c1[0], ROW[4], c3[0] + c3[1] - c1[0], "OpenMC",
               ["fixed-source coupled n–γ transport, ENDF/B-VIII.1"])
B["tbr"] = Box("metric", c1[0], ROW[5], c1[1], "Tritium breeding\nratio")
B["heat"] = Box("metric", c2[0], ROW[5], c2[1], "Nuclear heating\nby component")
B["tally"] = Box("data", c3[0], ROW[5], c3[1], "Mesh & cell tallies",
                 ["heating, ¹⁹F(n,x) rates", "(1 cm R–Z bins)"])

# thermal hydraulics
B["cool"] = Box("design", c4[0], ROW[1], c4[1], "Cooling layout",
                ["4 vessel circuits,", "bypass, outlet"])
B["gmsh"] = Box("code", c5[0], ROW[1], c5[1], "Gmsh", ["quads → 2° wedge"])
B["mesh"] = Box("data", c5[0], ROW[2], c5[1], "Wedge mesh", ["(.msh → polyMesh)"])
B["flow"] = Box("design", c4[0], ROW[4], 31.0, "Flow & heat loads",
                [r"$T_{\rm in}$ = 800 K, $\Delta T$ = 100 K", "plasma surface heat 249 MW"])
B["prop"] = Box("design", 141.0, ROW[4], 36.0, "FLiBe properties",
                [r"$\rho(T)$, $\mu(T)$, $c_p$, $k$"])
B["map"] = Box("code", c4[0], ROW[5], 49.0, "Case setup & source mapping",
               [r"circuit flows $\dot m_i = Q_i\,/\,(c_p\,\Delta T_i)$",
                "conservative bin → cell map (< 1 %)"])
B["case"] = Box("data", c6[0], ROW[5], c6[1], "OpenFOAM case",
                [r"$q'''$, $S_i$, wall $q''$, $\dot m_i$"])
B["foam"] = Box("code", c4[0], ROW[6], c6[0] + c6[1] - c4[0], "OpenFOAM v2606",
                ["buoyantSimpleFoam: steady 2D-axisymmetric RANS, realizable k–ε",
                 "frozen flow → scalarTransport with radioactive decay + mean age"])
B["vel"] = Box("metric", c4[0], ROW[7], c4[1], "Velocity &\npressure")
B["temp"] = Box("metric", c5[0], ROW[7], c5[1], "Temperature\n(peak, outlet)")
B["iso"] = Box("metric", c6[0], ROW[7], c6[1], "Isotope outlet\nactivity, mean age")


def arrows():
    """(points, bold) for every arrow; points run tail -> head."""
    b = B
    mx = b["mesh"].x + b["mesh"].w / 2
    px = b["prop"].x + 3.5
    return [
        # geometry
        ([b["design"].right(), b["paramak"].left()], False),
        ([b["paramak"].right(), b["rz"].left()], True),
        ([b["paramak"].bottom(), b["cad"].top()], True),
        ([b["rz"].bottom(), b["gmsh"].top()], True),
        # neutronics
        ([b["plasma"].bottom(), b["ops"].top()], False),
        ([b["ops"].bottom(), b["src"].top()], False),
        ([b["cad"].bottom(), b["c2d"].top()], False),
        ([b["c2d"].bottom(), b["h5m"].top()], False),
        ([b["src"].bottom(), b["omc"].at_x(b["src"].x + b["src"].w / 2)], False),
        ([b["h5m"].bottom(), b["omc"].at_x(b["h5m"].x + b["h5m"].w / 2)], False),
        ([b["mat"].bottom(), b["omc"].at_x(b["mat"].x + b["mat"].w / 2)], False),
        ([b["omc"].at_x(b["tbr"].x + b["tbr"].w / 2, "bottom"), b["tbr"].top()], False),
        ([b["omc"].at_x(b["heat"].x + b["heat"].w / 2, "bottom"), b["heat"].top()], False),
        ([b["omc"].at_x(b["tally"].x + b["tally"].w / 2, "bottom"), b["tally"].top()], False),
        # coupling
        ([b["tally"].right(), b["map"].left()], True),
        # thermal hydraulics
        ([b["cool"].right(), b["gmsh"].left()], False),
        ([b["gmsh"].bottom(), b["mesh"].top()], False),
        ([b["mesh"].bottom(), b["map"].at_x(mx)], False),
        ([b["flow"].bottom(), b["map"].at_x(b["flow"].x + b["flow"].w / 2)], False),
        ([(px, b["prop"].y + b["prop"].h), b["map"].at_x(px)], False),
        ([b["map"].right(), b["case"].left()], False),
        ([b["case"].bottom(), b["foam"].at_x(b["case"].x + b["case"].w / 2)], False),
        ([b["foam"].at_x(b["vel"].x + b["vel"].w / 2, "bottom"), b["vel"].top()], False),
        ([b["foam"].at_x(b["temp"].x + b["temp"].w / 2, "bottom"), b["temp"].top()], False),
        ([b["foam"].at_x(b["iso"].x + b["iso"].w / 2, "bottom"), b["iso"].top()], False),
    ]


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------
def draw_panel(ax, rect, title):
    x, y, w, h = rect
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=4",
                                fc=PANEL_FACE, ec=PANEL_EDGE, lw=0.8, zorder=0))
    ax.text(x + 4.0, y + h - 4.5, title, fontsize=PANEL_PT, fontweight="bold",
            color="#3a3a3a", ha="left", va="center", zorder=1)


def draw_box(ax, b: Box, texts: list):
    face, edge = KINDS[b.kind]
    ax.add_patch(FancyBboxPatch((b.x, b.y), b.w, b.h, boxstyle="round,pad=0,rounding_size=1.2",
                                fc=face, ec=edge, lw=0.7, zorder=3))
    rows = [(t, TITLE_PT, "bold") for t in b.title.split("\n")]
    rows += [(t, BODY_PT, "normal") for t in b.lines]
    total = sum(s * LINE * PT for _, s, _ in rows)
    yy = b.yc - total / 2
    cx = b.x + b.w / 2
    for t, s, wt in rows:
        lh = s * LINE * PT
        txt = ax.text(cx, yy + lh / 2, t, fontsize=s, fontweight=wt, color=INK,
                      ha="center", va="center", zorder=4)
        texts.append((b, txt))
        yy += lh


def draw_arrow(ax, pts, bold):
    lw = BOLD if bold else THIN
    style = "-|>,head_length=0.55,head_width=0.28" if bold else "-|>,head_length=0.5,head_width=0.22"
    path = MPath(pts, [MPath.MOVETO] + [MPath.LINETO] * (len(pts) - 1))
    ax.add_patch(FancyArrowPatch(path=path, arrowstyle=style, mutation_scale=7 if bold else 6,
                                 lw=lw, color=INK, joinstyle="miter", capstyle="butt",
                                 zorder=2))


def draw_legend(ax, x0, y0, cw=18.0, ch=5.0, gap=3.0):
    """Colour key in one row, arrow key in a second row below it."""
    for i, (kind, label) in enumerate(LEGEND):
        x = x0 + i * (cw + gap)
        face, edge = KINDS[kind]
        ax.add_patch(FancyBboxPatch((x, y0), cw, ch, boxstyle="round,pad=0,rounding_size=0.8",
                                    fc=face, ec=edge, lw=0.6, zorder=3))
        ax.text(x + cw / 2, y0 + ch / 2, label, fontsize=LEGEND_PT + 0.5, color=INK,
                ha="center", va="center", zorder=4)
    for j, (bold, label) in enumerate([(True, "data transfer between domains"),
                                       (False, "input / output")]):
        y = y0 + ch + 5.0 + j * 5.0
        draw_arrow(ax, [(x0, y), (x0 + 9.0, y)], bold)
        ax.text(x0 + 11.5, y, label, fontsize=LEGEND_PT + 0.5, color=INK,
                ha="left", va="center")


def check_fit(fig, texts, pad=0.8):
    """Warn about any text wider than its box (minus padding)."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    px_per_mm = fig.dpi / 25.4
    bad = 0
    for b, t in texts:
        wmm = t.get_window_extent(r).width / px_per_mm
        if wmm > b.w - 2 * pad:
            bad += 1
            print(f"  overflow: '{t.get_text()}' is {wmm:.1f} mm in a {b.w:.1f} mm box")
    print("text fits every box" if not bad else f"{bad} overflow(s)")


def main():
    fig = plt.figure(figsize=(W * MM, H * MM))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.axis("off")

    draw_panel(ax, PANEL_N, "Neutronics")
    draw_panel(ax, PANEL_T, "Thermal Hydraulics")
    texts = []
    for b in B.values():
        draw_box(ax, b, texts)
    for pts, bold in arrows():
        draw_arrow(ax, pts, bold)
    draw_legend(ax, PANEL_N[0] + 3.0, PANEL_N[1] + PANEL_N[3] + 6.0)

    check_fit(fig, texts)
    for ext in ("pdf", "svg", "png"):
        fig.savefig(f"{OUT}.{ext}", dpi=600 if ext == "png" else None)
    print(f"wrote {OUT}.pdf/.svg/.png")


if __name__ == "__main__":
    main()
