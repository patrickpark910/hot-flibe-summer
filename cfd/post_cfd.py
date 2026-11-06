#!/usr/bin/env python3
"""
ARC V3A FLiBe CFD post-processing (run in the Python env after ./Allrun solve)
==============================================================================

Reads the converged case (ASCII) and reports everything scaled to the full
360 deg machine:
  * mass flow in/out, flow-weighted outlet temperature, energy balance
  * peak FLiBe temperature (cells and wall faces) and where it is
  * y+ on every wall patch
  * per vessel circuit: discharge flow, temperature, mean age and N-16 reaching the bulk
  * per isotope: production, inventory (gap / bulk), decay rate, outflow,
    fraction leaving through the outlet, outlet activity concentration;
    production = decay + outflow must close within 1 %
  * flow-weighted mean age at the outlet (checked against M / m_dot) and its
    distribution over the outlet and over the FLiBe mass
  * PNG maps of |U|, T, age and N-16 activity concentration

    python cfd/post_cfd.py [--case cfd/case_v3a] [--time <dir>]

Writes postProcessing/post_cfd/report.txt and PNGs inside the case.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import foam_io  # noqa: E402

BLUE = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5",
        "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]   # sequential 100 -> 700


class Report:
    def __init__(self):
        self.lines = []

    def __call__(self, s=""):
        print(s)
        self.lines.append(s)


def latest_time(case: Path) -> str:
    times = []
    for d in case.iterdir():
        if d.is_dir() and re.fullmatch(r"[0-9.eE+-]+", d.name):
            try:
                if float(d.name) > 0:
                    times.append((float(d.name), d.name))
            except ValueError:
                pass
    if not times:
        raise SystemExit("no solution time directory: run ./Allrun solve first")
    return max(times)[1]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--case", default=str(HERE / "case_v3a"))
    ap.add_argument("--time", default=None)
    args = ap.parse_args()
    case = Path(args.case)
    t = args.time or latest_time(case)
    td = case / t
    out_dir = case / "postProcessing" / "post_cfd"
    out_dir.mkdir(parents=True, exist_ok=True)
    rep = Report()
    S = json.loads((case / "case_summary.json").read_text())
    k = 360.0 / S["wedge_deg"]
    a, b = S["rho_coeffs"]
    rho = lambda T: a + b * T
    cp, T_in = S["cp"], S["T_in"]

    # ---- mesh -------------------------------------------------------------------
    pm = case / "constant" / "polyMesh"
    pts = foam_io.read_points(pm / "points")
    faces = foam_io.read_faces(pm / "faces")
    owner = foam_io.read_labels(pm / "owner")
    nei = foam_io.read_labels(pm / "neighbour")
    bnd = foam_io.read_boundary(pm / "boundary")
    zones = foam_io.read_cell_zones(pm / "cellZones")
    V = np.asarray(foam_io.read_field(case / "0" / "V")["internal"], float)
    C = foam_io.read_field(case / "0" / "C")["internal"]
    Rc, Zc = np.hypot(C[:, 0], C[:, 1]), C[:, 2]
    nI = len(nei)
    zone_of = np.empty(len(V), dtype=object)
    for z, lab in zones.items():
        zone_of[lab] = z
    in_gap = np.array([str(z).startswith("gap_") for z in zone_of])

    def fcentre(fi):
        p = pts[faces[fi]].mean(axis=0)
        return math.hypot(p[0], p[1]), p[2]

    # ---- fields -----------------------------------------------------------------
    def fld(name, where=td):
        f = where / name
        return foam_io.read_field(f) if f.exists() else None

    T, U, phi = fld("T"), fld("U"), fld("phi")
    if T is None or phi is None:
        raise SystemExit(f"{td} lacks T or phi")
    Tc = np.asarray(T["internal"], float)
    rho_c = rho(Tc)
    Uc = U["internal"] if U is not None else None
    phiI = np.asarray(phi["internal"], float)

    def patch_vals(F, p, n):
        v = F["boundary"][p]["value"]
        return np.full(n, float(v)) if np.ndim(v) == 0 else np.asarray(v, float)

    rep(f"ARC V3A FLiBe CFD results - case {case.name}, time {t}, wedge {S['wedge_deg']} deg "
        f"(all values full machine, x{k:g})")
    rep("=" * 96)

    # ---- mass and energy -----------------------------------------------------------
    rep("\nMass flow by patch (kg/s; + = out)")
    m_out = m_in = H_net = 0.0
    T_out_num = 0.0
    outlet = {}
    for p, info in S["patches"].items():
        if info["role"] not in ("inlet", "outlet"):
            continue
        n = bnd[p]["nFaces"]
        ph = patch_vals(phi, p, n) * k
        Tf = patch_vals(T, p, n)
        H_net += float((ph * cp * (Tf - T_in)).sum())
        if info["role"] == "inlet":
            m_in += -ph.sum()
            rep(f"  {p:16s} {ph.sum():10.2f}   (target {-info['mdot_full']:.2f})")
        else:
            m_out += ph.sum()
            pos = ph > 0
            T_out_num += float((ph[pos] * Tf[pos]).sum())
            outlet = dict(phi=ph, T=Tf, start=bnd[p]["startFace"], n=n, pos=pos)
            rep(f"  {p:16s} {ph.sum():10.2f}   (backflow {-(ph[~pos]).sum():.3f} kg/s on {int((~pos).sum())} faces)")
    m_out_pos = float(outlet["phi"][outlet["pos"]].sum())
    T_out = T_out_num / m_out_pos
    rep(f"  in {m_in:.2f}, out {m_out:.2f}, imbalance {100 * (m_out - m_in) / m_in:+.4f} %")
    rep(f"  flow-weighted outlet temperature {T_out:.2f} K (target {S['flows']['T_out_target']:.1f} K)")
    Q_walls = sum(i["Q_full"] for i in S["patches"].values() if "Q_full" in i)
    q = fld("qFlibe", case / "0")
    Q_vol = float((np.asarray(q["internal"]) * V).sum() * k) if q else S["mapped"]["qFlibe"]["total"]
    Q_imp = Q_walls + Q_vol
    rep(f"\nEnergy balance: imposed {Q_imp / 1e6:.2f} MW (walls {Q_walls / 1e6:.2f}, FLiBe volume {Q_vol / 1e6:.2f});"
        f" convected out m cp (T - T_in) {H_net / 1e6:.2f} MW; error {100 * (H_net / Q_imp - 1):+.3f} %"
        f" (target within 1-2 %)")
    rep("  (gravity work rho U.g and kinetic energy are in the solver's energy equation; both are < 0.1 MW here)")

    # ---- peak temperature ------------------------------------------------------------
    ic = int(np.argmax(Tc))
    rep(f"\nPeak FLiBe temperature (cell): {Tc[ic]:.1f} K at R {Rc[ic]:.3f} m, Z {Zc[ic]:+.3f} m ({zone_of[ic]})")
    best = (-1.0, None, None)
    for p, info in S["patches"].items():
        if info["role"] != "wallHeat":
            continue
        n = bnd[p]["nFaces"]
        Tf = patch_vals(T, p, n)
        j = int(np.argmax(Tf))
        if Tf[j] > best[0]:
            best = (float(Tf[j]), p, fcentre(bnd[p]["startFace"] + j))
    rep(f"Peak wall temperature: {best[0]:.1f} K on {best[1]} at R {best[2][0]:.3f} m, Z {best[2][1]:+.3f} m")
    rep("  (no conjugate heat transfer: this is the FLiBe-side wall temperature)")

    # ---- y+ ----------------------------------------------------------------------------
    yp = fld("yPlus")
    if yp is not None:
        rep("\ny+ on walls (min / median / max)")
        for p, info in S["patches"].items():
            if info["role"].startswith("wall") and p in yp["boundary"] and yp["boundary"][p]["value"] is not None:
                v = patch_vals(yp, p, bnd[p]["nFaces"])
                rep(f"  {p:16s} {v.min():8.2f} {np.median(v):8.2f} {v.max():8.2f}")

    # ---- circuit discharges (internal faces gap_c -> bulk) -------------------------------
    own_z, nei_z = zone_of[owner[:nI]], zone_of[nei]
    circuits = [z for z in zones if z.startswith("gap_")]
    disc = {}
    for z in circuits:
        f_on = np.nonzero(((own_z == z) & (nei_z == "bulk")) | ((own_z == "bulk") & (nei_z == z)))[0]
        sgn = np.where(own_z[f_on] == z, 1.0, -1.0)
        flux = phiI[f_on] * sgn * k                                       # gap -> bulk
        up = np.where(phiI[f_on] > 0, owner[f_on], nei[f_on])            # upwind cell
        disc[z] = dict(faces=f_on, flux=flux, up=up)

    # ---- isotopes ----------------------------------------------------------------------
    iso_rows = []
    for iso in S["isotopes"]:
        Cf, Sf = fld(f"C_{iso}"), fld(f"S_{iso}", case / "0")
        if Cf is None or Sf is None:
            rep(f"\n{iso}: C_{iso} or S_{iso} not found - run ./Allrun scalars")
            continue
        lam = S["decay"][iso]["lam"]
        Cc = np.asarray(Cf["internal"], float)                            # atoms/kg
        P = float((np.asarray(Sf["internal"]) * V).sum() * k)
        N_all = rho_c * Cc * V * k
        N, Ng = float(N_all.sum()), float(N_all[in_gap].sum())
        Cout = patch_vals(Cf, "outlet", outlet["n"])
        ph, pos = outlet["phi"], outlet["pos"]
        O = float((ph[pos] * Cout[pos]).sum())
        Qv = float((ph[pos] / rho(outlet["T"][pos])).sum())               # m3/s out
        D = lam * N
        err = (P - D - O) / P
        iso_rows.append((iso, P, N, Ng, D, O, err, lam * O / Qv, lam))
        to_bulk = {z: float((d["flux"] * Cc[d["up"]]).sum()) for z, d in disc.items()}
        prod_z = {z: float((np.asarray(Sf["internal"])[zones[z]] * V[zones[z]]).sum() * k) for z in circuits}
        iso_rows[-1] += (to_bulk, prod_z)
    if iso_rows:
        rep("\nActivation products (full machine)")
        rep(f"  {'':5s} {'half-life':>10s} {'production':>11s} {'inventory':>11s} {'in gap':>7s} {'decay':>11s}"
            f" {'outflow':>11s} {'leaving':>8s} {'balance':>8s} {'outlet λC':>11s}")
        rep(f"  {'':5s} {'s':>10s} {'1/s':>11s} {'atoms':>11s} {'%':>7s} {'Bq':>11s} {'1/s':>11s} {'%':>8s}"
            f" {'%':>8s} {'Bq/m3':>11s}")
        bad = False
        for iso, P, N, Ng, D, O, err, act, lam in [r[:9] for r in iso_rows]:
            bad |= abs(err) > 0.01
            rep(f"  {iso:5s} {S['decay'][iso]['half_life_s']:10.5g} {P:11.4e} {N:11.4e} {100 * Ng / N:7.2f} {D:11.4e}"
                f" {O:11.4e} {100 * O / P:8.3f} {100 * err:+8.3f} {act:11.4e}")
        rep(f"  activity carried out (lambda x outflow): " + ", ".join(f"{r[0]} {r[8] * r[5]:.3e} Bq/s" for r in iso_rows))
        if bad:
            rep("  WARNING: production = decay + outflow does not close within 1 % for at least one isotope")

    # ---- circuits -----------------------------------------------------------------------
    age = fld("age")
    agec = np.asarray(age["internal"], float) if age is not None else None
    rep("\nVessel circuits: discharge into the bulk (internal faces gap -> bulk)")
    n16 = next((r for r in iso_rows if r[0] == "N16"), None)
    for z, d in disc.items():
        m = float(d["flux"].sum())
        Tm = float((d["flux"] * Tc[d["up"]]).sum() / m)
        s = f"  {z:14s} m_dot {m:8.1f} kg/s  T {Tm:7.2f} K"
        if agec is not None:
            s += f"  mean age {float((d['flux'] * agec[d['up']]).sum() / m):7.2f} s"
        if n16 is not None:
            s += f"  N-16 reaching bulk {100 * n16[9][z] / n16[10][z]:6.2f} % of its production"
        rep(s)

    # ---- age -----------------------------------------------------------------------------
    if agec is not None:
        a_out = patch_vals(age, "outlet", outlet["n"])
        ph, pos = outlet["phi"], outlet["pos"]
        a_mean = float((ph[pos] * a_out[pos]).sum() / ph[pos].sum())
        M = float((rho_c * V).sum() * k)
        rep(f"\nMean age: flow-weighted at the outlet {a_mean:.2f} s; M / m_dot = {M / m_out_pos:.2f} s "
            f"(should agree; difference {100 * (a_mean * m_out_pos / M - 1):+.2f} %)")
        w = ph[pos] / ph[pos].sum()
        srt = np.argsort(a_out[pos])
        cw = np.cumsum(w[srt])
        pct = {q: float(a_out[pos][srt][np.searchsorted(cw, q / 100)]) for q in (5, 25, 50, 75, 95)}
        rep("  outlet age percentiles (flow-weighted): " + ", ".join(f"p{q} {v:.1f} s" for q, v in pct.items()))
        mw = rho_c * V
        srt = np.argsort(agec)
        cw = np.cumsum(mw[srt]) / mw.sum()
        pv = {q: float(agec[srt][np.searchsorted(cw, q / 100)]) for q in (5, 50, 95)}
        rep("  FLiBe mass age percentiles: " + ", ".join(f"p{q} {v:.1f} s" for q, v in pv.items()))
        rep("  note: this is the distribution of local mean age; a true residence-time distribution needs a transient tracer")

    (out_dir / "report.txt").write_text("\n".join(rep.lines) + "\n")
    print(f"\nwrote {out_dir / 'report.txt'}")
    plots(case, td, out_dir, pts, faces, owner, bnd, Uc, Tc, agec, rho_c, V, iso_rows, outlet, age)


def plots(case, td, out_dir, pts, faces, owner, bnd, Uc, Tc, agec, rho_c, V, iso_rows, outlet, age):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection
    from matplotlib.colors import LinearSegmentedColormap, LogNorm, Normalize

    cmap = LinearSegmentedColormap.from_list("blue_seq", BLUE)
    fr = bnd["front"]
    fids = range(fr["startFace"], fr["startFace"] + fr["nFaces"])
    polys, cell = [], []
    for fi in fids:
        p = pts[faces[fi]]
        polys.append(np.column_stack([np.hypot(p[:, 0], p[:, 1]), p[:, 2]]))
        cell.append(owner[fi])
    cell = np.asarray(cell)
    panels = []
    if Uc is not None:
        panels.append(("|U| (m/s)", np.linalg.norm(Uc, axis=1)[cell], "log"))
    panels.append(("T (K)", Tc[cell], "lin"))
    if agec is not None:
        panels.append(("mean age (s)", agec[cell], "lin"))
    n16 = next((r for r in iso_rows if r[0] == "N16"), None)
    if n16 is not None:
        Cn = np.asarray(foam_io.read_field(td / "C_N16")["internal"], float)
        panels.append(("N-16 activity (Bq/m3)", (n16[8] * rho_c * Cn)[cell], "log"))
    fig, axs = plt.subplots(1, len(panels), figsize=(4.2 * len(panels), 7.2), facecolor="white")
    for ax, (title, val, scale) in zip(np.atleast_1d(axs), panels):
        if scale == "log":
            pos = val[val > 0]
            norm = LogNorm(vmin=max(pos.min(), pos.max() * 1e-4), vmax=pos.max())
            val = np.clip(val, norm.vmin, None)
        else:
            norm = Normalize(vmin=val.min(), vmax=val.max())
        pc = PolyCollection(polys, array=val, cmap=cmap, norm=norm, edgecolors="none")
        ax.add_collection(pc)
        ax.set_xlim(2.3, 7.4)
        ax.set_ylim(-4.9, 4.9)
        ax.set_aspect("equal")
        ax.set_title(title, fontsize=10, loc="left", color="#0b0b0b")
        ax.set_xlabel("R (m)", fontsize=9, color="#52514e")
        ax.tick_params(labelsize=8, colors="#52514e")
        cb = fig.colorbar(pc, ax=ax, shrink=0.8)
        cb.ax.tick_params(labelsize=8)
    np.atleast_1d(axs)[0].set_ylabel("Z (m)", fontsize=9, color="#52514e")
    fig.tight_layout()
    fig.savefig(out_dir / "maps.png", dpi=170)
    plt.close(fig)
    print(f"wrote {out_dir / 'maps.png'}")
    if agec is not None:
        a_out = age["boundary"]["outlet"]["value"]
        a_out = np.full(outlet["n"], float(a_out)) if np.ndim(a_out) == 0 else np.asarray(a_out, float)
        ph, pos = outlet["phi"], outlet["pos"]
        fig, axs = plt.subplots(1, 2, figsize=(11, 4), facecolor="white")
        axs[0].hist(a_out[pos], bins=40, weights=ph[pos] / ph[pos].sum(), color=BLUE[7], edgecolor="white", lw=0.5)
        axs[0].set_title("Outlet: flow-weighted local mean age", fontsize=10, loc="left")
        axs[0].set_xlabel("age (s)")
        axs[0].set_ylabel("fraction of outlet mass flow")
        mw = rho_c * V
        axs[1].hist(agec, bins=60, weights=mw / mw.sum(), color=BLUE[7], edgecolor="white", lw=0.5)
        axs[1].set_title("Tank: mass-weighted local mean age", fontsize=10, loc="left")
        axs[1].set_xlabel("age (s)")
        axs[1].set_ylabel("fraction of FLiBe mass")
        for ax in axs:
            ax.spines[["top", "right"]].set_visible(False)
            ax.grid(axis="y", color="#e5e4df", lw=0.6)
        fig.tight_layout()
        fig.savefig(out_dir / "age_distribution.png", dpi=170)
        plt.close(fig)
        print(f"wrote {out_dir / 'age_distribution.png'}")


if __name__ == "__main__":
    main()
