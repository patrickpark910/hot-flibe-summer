"""Movie of the steady buoyantSimpleFoam iterations in this case directory.

Each frame is one saved iteration (writeInterval 50 in system/controlDict):
|U| and T in the R-Z plane. The iterations are solver passes, not physical
time, so the movie shows how the steady solution wanders, not how the FLiBe
moves.

    .venv/bin/python cfd_iterations/make_movie.py [--fps 12]

Writes postProcessing/movie/flow_iterations.mp4 (ffmpeg) or .gif (no ffmpeg),
plus last_frame.png. Reads the mesh and fields with cfd/foam_io.py (unchanged).
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "cfd"))
import foam_io  # noqa: E402

CORNER_RZ = (3.58, -2.07)   # m, lower inboard junction corner (where the transient run diverged)


def main():
    ap = argparse.ArgumentParser(description="steady-iteration movie")
    ap.add_argument("--case", default=str(HERE))
    ap.add_argument("--fps", type=int, default=12)
    args = ap.parse_args()
    case = Path(args.case)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FFMpegWriter, FuncAnimation, PillowWriter
    from matplotlib.collections import PolyCollection
    from matplotlib.colors import LogNorm, Normalize

    # ---- mesh: front wedge faces as R-Z polygons, one per cell ---------------
    pm = case / "constant" / "polyMesh"
    pts = foam_io.read_points(pm / "points")
    faces = foam_io.read_faces(pm / "faces")
    owner = foam_io.read_labels(pm / "owner")
    fr = foam_io.read_boundary(pm / "boundary")["front"]
    polys, cell = [], []
    for fi in range(fr["startFace"], fr["startFace"] + fr["nFaces"]):
        p = pts[faces[fi]]
        polys.append(np.column_stack([np.hypot(p[:, 0], p[:, 1]), p[:, 2]]))
        cell.append(owner[fi])
    cell = np.asarray(cell)

    # ---- saved iterations ------------------------------------------------------
    times = []
    for d in case.iterdir():
        try:
            it = float(d.name)
        except ValueError:
            continue
        if it > 0 and (d / "U").exists() and (d / "T").exists():
            times.append((it, d))
    times.sort()
    if not times:
        raise SystemExit(f"no saved iterations with U and T in {case}")
    print(f"reading {len(times)} saved iterations ({times[0][1].name} .. {times[-1][1].name})")
    Umag = np.array([np.linalg.norm(foam_io.read_field(d / "U")["internal"], axis=1)[cell] for _, d in times])
    T = np.array([np.asarray(foam_io.read_field(d / "T")["internal"], float)[cell] for _, d in times])

    # Fixed colour limits for the whole movie; the top 0.1 % (corner cells) is clipped.
    u_hi = np.percentile(Umag, 99.9)
    u_lo = max(np.percentile(Umag[Umag > 0], 0.5), u_hi * 1e-4)
    t_lo, t_hi = T.min(), np.percentile(T, 99.9)

    cmap = matplotlib.colormaps["coolwarm"]   # same as post_cfd.py maps.png
    fig, axs = plt.subplots(1, 2, figsize=(8.6, 6.8), facecolor="white")
    pcs = []
    for ax, (title, data, norm) in zip(axs, (
            ("|U| (m/s), log scale", Umag, LogNorm(vmin=u_lo, vmax=u_hi)),
            ("T (K)", T, Normalize(vmin=t_lo, vmax=t_hi)))):
        pc = PolyCollection(polys, array=np.clip(data[0], norm.vmin, None), cmap=cmap, norm=norm,
                            edgecolors="none")
        ax.add_collection(pc)
        ax.plot(*CORNER_RZ, marker="o", ms=14, mfc="none", mec="#0b0b0b", mew=1.2)
        ax.set_xlim(2.3, 7.4)
        ax.set_ylim(-4.9, 4.9)
        ax.set_aspect("equal")
        ax.set_title(title, fontsize=10, loc="left", color="#0b0b0b")
        ax.set_xlabel("R (m)", fontsize=9, color="#52514e")
        ax.tick_params(labelsize=8, colors="#52514e")
        cb = fig.colorbar(pc, ax=ax, shrink=0.8, extend="max")
        cb.ax.tick_params(labelsize=8)
        pcs.append((pc, data, norm))
    axs[0].set_ylabel("Z (m)", fontsize=9, color="#52514e")
    sup = fig.suptitle("", fontsize=11, x=0.02, ha="left", color="#0b0b0b")
    fig.text(0.02, 0.015, "circle: lower inboard junction corner. Steady solver iterations, not physical time.",
             fontsize=8, color="#52514e")
    fig.tight_layout(rect=(0, 0.045, 1, 0.96))

    def draw(k):
        for pc, data, norm in pcs:
            pc.set_array(np.clip(data[k], norm.vmin, None))
        sup.set_text(f"ARC V3A FLiBe, steady solver iteration {times[k][1].name} / {times[-1][1].name}")
        return [pc for pc, _, _ in pcs] + [sup]

    out_dir = case / "postProcessing" / "movie"
    out_dir.mkdir(parents=True, exist_ok=True)
    anim = FuncAnimation(fig, draw, frames=len(times), blit=False)
    if shutil.which("ffmpeg"):
        out = out_dir / "flow_iterations.mp4"
        anim.save(out, writer=FFMpegWriter(fps=args.fps, bitrate=4000), dpi=130)
    else:
        out = out_dir / "flow_iterations.gif"
        anim.save(out, writer=PillowWriter(fps=args.fps), dpi=90)
    print(f"wrote {out}")
    draw(len(times) - 1)
    fig.savefig(out_dir / "last_frame.png", dpi=130)
    print(f"wrote {out_dir / 'last_frame.png'}")


if __name__ == "__main__":
    main()
