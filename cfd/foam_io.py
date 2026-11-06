"""Minimal reader/writer for ASCII OpenFOAM files (fields, polyMesh, zones).

Only what make_case.py and post_cfd.py need. Cases must be written with
writeFormat ascii (make_case.py sets this).
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

_NUM = r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?"


def header(cls: str, obj: str, location: str | None = None) -> str:
    loc = f'    location    "{location}";\n' if location else ""
    return ("/*--------------------------------*- C++ -*----------------------------------*\\\n"
            "  Written by cfd/make_case.py (ARC V3A FLiBe case)\n"
            "\\*---------------------------------------------------------------------------*/\n"
            "FoamFile\n{\n    version     2.0;\n    format      ascii;\n"
            f"    class       {cls};\n{loc}    object      {obj};\n}}\n"
            "// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //\n\n")


def _strip(text: str) -> str:
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
    return re.sub(r"//[^\n]*", "", text)


def _body(text: str) -> str:
    """Text after the FoamFile header block."""
    text = _strip(text)
    m = re.search(r"FoamFile\s*\{[^}]*\}", text)
    return text[m.end():] if m else text


def _list_at(text: str, pos: int):
    """Parse 'N ( ... )' or 'N(...)' starting at/after pos. Returns (inner text, end pos)."""
    m = re.compile(r"\s*(\d+)\s*\(").match(text, pos)
    if not m:
        raise ValueError(f"no list at {text[pos:pos + 40]!r}")
    depth, i = 1, m.end()
    while depth:
        c = text[i]
        if c == "(":
            depth += 1
        elif c == ")":
            depth -= 1
        i += 1
    return int(m.group(1)), text[m.end():i - 1], i


def _numbers(s: str) -> np.ndarray:
    s = s.replace("(", " ").replace(")", " ")
    return np.array(s.split(), dtype=float) if s.strip() else np.zeros(0)


def _parse_value(spec: str, ncomp: int):
    """'uniform X', 'uniform (x y z)' or 'nonuniform List<T> N (...)'."""
    spec = spec.strip()
    if spec.startswith("uniform"):
        v = _numbers(spec[len("uniform"):])
        return v[0] if ncomp == 1 else v
    m = re.match(r"nonuniform\s+List<\w+>", spec)
    if m:
        n, inner, _ = _list_at(spec, m.end())
        v = _numbers(inner)
        return v if ncomp == 1 else v.reshape(n, ncomp)
    raise ValueError(f"cannot parse value {spec[:60]!r}")


def _block(text: str, start: int):
    """Contents of the {...} block whose opening brace is at or after start."""
    i = text.index("{", start)
    depth, j = 1, i + 1
    while depth:
        if text[j] == "{":
            depth += 1
        elif text[j] == "}":
            depth -= 1
        j += 1
    return text[i + 1:j - 1], j


def _entry(block: str, key: str):
    """Raw text of 'key <value>;' at the top level of a block (lists allowed)."""
    m = re.search(rf"(?<![\w.]){re.escape(key)}\s+", block)
    if not m:
        return None
    i, depth = m.end(), 0
    while i < len(block):
        c = block[i]
        if c == "(":
            depth += 1
        elif c == ")":
            depth -= 1
        elif c == ";" and depth == 0:
            return block[m.end():i]
        i += 1
    return block[m.end():]


def read_field(path) -> dict:
    """vol/surface field -> dict(cls, internal, boundary={patch: dict(type, value)})."""
    raw = Path(path).read_text()
    cls = re.search(r"class\s+(\w+)\s*;", raw).group(1)
    ncomp = 3 if "Vector" in cls else 1
    body = _body(raw)
    internal = _parse_value(_entry(body, "internalField"), ncomp)
    bf, _ = _block(body, body.index("boundaryField"))
    boundary, pos = {}, 0
    pat = re.compile(r"\s*\"?([\w.:()|*-]+)\"?\s*\{")
    while True:
        m = pat.match(bf, pos)
        if not m:
            break
        blk, pos = _block(bf, m.start())
        typ = _entry(blk, "type")
        val = _entry(blk, "value")
        boundary[m.group(1)] = dict(type=typ.strip() if typ else None,
                                    value=_parse_value(val, ncomp) if val else None)
    return dict(cls=cls, internal=internal, boundary=boundary)


def read_points(path) -> np.ndarray:
    body = _body(Path(path).read_text())
    n, inner, _ = _list_at(body, 0)
    return _numbers(inner).reshape(n, 3)


def read_labels(path) -> np.ndarray:
    body = _body(Path(path).read_text())
    n, inner, _ = _list_at(body, 0)
    return _numbers(inner).astype(np.int64)


def read_faces(path) -> list[np.ndarray]:
    raw = Path(path).read_text()
    body = _body(raw)
    if re.search(r"class\s+faceCompactList", raw):
        n, inner, end = _list_at(body, 0)
        offs = _numbers(inner).astype(np.int64)
        _, inner2, _ = _list_at(body, end)
        lab = _numbers(inner2).astype(np.int64)
        return [lab[offs[k]:offs[k + 1]] for k in range(n - 1)]
    n, inner, _ = _list_at(body, 0)
    faces = [np.array(f.split(), dtype=np.int64) for f in re.findall(r"\d+\s*\(([^)]*)\)", inner)]
    if len(faces) != n:
        raise ValueError("face count mismatch")
    return faces


def read_boundary(path) -> dict:
    body = _body(Path(path).read_text())
    _, inner, _ = _list_at(body, 0)
    out = {}
    for m in re.finditer(r"([\w.-]+)\s*\{([^}]*)\}", inner):
        blk = m.group(2)
        out[m.group(1)] = dict(type=_entry(blk, "type").strip(), nFaces=int(_entry(blk, "nFaces")),
                               startFace=int(_entry(blk, "startFace")))
    return out


def read_cell_zones(path) -> dict:
    body = _body(Path(path).read_text())
    _, inner, _ = _list_at(body, 0)
    out, pos = {}, 0
    pat = re.compile(r"\s*([\w.-]+)\s*\{")
    while True:
        m = pat.match(inner, pos)
        if not m:
            break
        blk, pos = _block(inner, m.start())
        lab = re.search(r"cellLabels\s+List<label>", blk)
        _, li, _ = _list_at(blk, lab.end())
        out[m.group(1)] = _numbers(li).astype(np.int64)
    return out


def write_scalar_field(path, name: str, values: np.ndarray, dims: str, patches: dict,
                       comment: str = "") -> None:
    """Nonuniform volScalarField; patches = {name: 'calculated'|'wedge'|...}."""
    lines = [header("volScalarField", name, "0")]
    if comment:
        lines.append("// " + comment.replace("\n", "\n// ") + "\n\n")
    lines.append(f"dimensions      {dims};\n\n")
    lines.append(f"internalField   nonuniform List<scalar>\n{len(values)}\n(\n")
    lines.append("\n".join(f"{v:.10g}" for v in values))
    lines.append("\n)\n;\n\nboundaryField\n{\n")
    for p, t in patches.items():
        if t == "wedge":
            lines.append(f"    {p}\n    {{\n        type            wedge;\n    }}\n")
        else:
            lines.append(f"    {p}\n    {{\n        type            calculated;\n        value           uniform 0;\n    }}\n")
    lines.append("}\n")
    Path(path).write_text("".join(lines))
