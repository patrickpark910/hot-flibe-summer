#!/usr/bin/env python3
"""
Render a 3D cutaway PNG of the ARC Paramak model (arc_paramak_model.py).

Builds the model for a toroidal extent of --angle degrees, tessellates every
part and renders it off-screen with VTK, looking into the open sector so the
cut faces (vacuum vessel, blanket, shields, coils) are visible. Colours are
the material colours used in arc_paramak_model.COLORS.

Keep this file next to arc_paramak_model.py.

Requirements:  pip install vtk   (plus the requirements of arc_paramak_model.py)

Examples:
    python render_arc_3d.py                                # 270 deg cutaway -> arc_3d_cutaway.png
    python render_arc_3d.py --angle 90 -o arc_90.png       # single 90 deg sector
    python render_arc_3d.py --hide tf_coils central_solenoid
"""

import argparse
import math
import os
import sys

import vtk

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import arc_paramak_model as arc  # noqa: E402


def to_actor(shape, rgb, tolerance):
    """Tessellates a CadQuery shape into a shaded VTK actor."""
    verts, tris = shape.tessellate(tolerance, 0.3)
    points = vtk.vtkPoints()
    for v in verts:
        points.InsertNextPoint(v.x, v.y, v.z)
    cells = vtk.vtkCellArray()
    for tri in tris:
        cells.InsertNextCell(3)
        for i in tri:
            cells.InsertCellPoint(i)
    poly = vtk.vtkPolyData()
    poly.SetPoints(points)
    poly.SetPolys(cells)
    normals = vtk.vtkPolyDataNormals()
    normals.SetInputData(poly)
    normals.SetFeatureAngle(40)
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputConnection(normals.GetOutputPort())
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetColor(*rgb)
    actor.GetProperty().SetSpecular(0.2)
    return actor


def main():
    ap = argparse.ArgumentParser(description="Render a 3D cutaway of the ARC model")
    ap.add_argument("--angle", type=float, default=270.0,
                    help="toroidal extent of the model in degrees (default 270)")
    ap.add_argument("-o", "--output", default="arc_3d_cutaway.png", help="PNG file to write")
    ap.add_argument("--size", type=int, nargs=2, default=(1600, 1300), metavar=("W", "H"))
    ap.add_argument("--tolerance", type=float, default=1.0,
                    help="tessellation tolerance in cm (smaller = smoother, slower)")
    ap.add_argument("--hide", nargs="*", default=[],
                    help="part names to leave out, e.g. tf_coils central_solenoid")
    ap.add_argument("--elevation", type=float, default=17.0,
                    help="camera elevation above the midplane in degrees")
    ap.add_argument("--zoom", type=float, default=None,
                    help="camera zoom (default 1.25 above 180 deg, else 0.9)")
    args = ap.parse_args()

    params = arc.ARCParameters()
    parts, _ = arc.build_profiles(params)
    material = {pt.name: pt.material for pt in parts}
    assembly = arc.build_assembly(params, parts, args.angle)

    renderer = vtk.vtkRenderer()
    renderer.SetBackground(1, 1, 1)
    for shape, name, _loc, _color in assembly:
        leaf = name.split("/")[-1]
        if leaf in args.hide:
            continue
        renderer.AddActor(to_actor(shape, arc.COLORS[material[leaf]], args.tolerance))

    window = vtk.vtkRenderWindow()
    window.SetOffScreenRendering(1)
    window.SetSize(*args.size)
    window.AddRenderer(renderer)

    # The model is revolved counter-clockwise from +X, so the open sector runs
    # from `angle` to 360 deg; look in from the middle of it.
    phi = math.radians((args.angle + 360.0) / 2.0) if args.angle < 360 else math.radians(-45)
    elev = math.radians(args.elevation)
    dist = 2600.0
    eye = (dist * math.cos(elev) * math.cos(phi),
           dist * math.cos(elev) * math.sin(phi),
           dist * math.sin(elev))
    camera = renderer.GetActiveCamera()
    camera.SetFocalPoint(0, 0, 0)
    camera.SetPosition(*eye)
    camera.SetViewUp(0, 0, 1)
    renderer.ResetCamera()
    camera.Zoom(args.zoom if args.zoom else (1.25 if args.angle > 180 else 0.9))

    light = vtk.vtkLight()
    light.SetPosition(eye[0], eye[1], eye[2] + 1500)
    light.SetFocalPoint(0, 0, 0)
    renderer.AddLight(light)

    window.Render()
    grab = vtk.vtkWindowToImageFilter()
    grab.SetInput(window)
    grab.Update()
    writer = vtk.vtkPNGWriter()
    writer.SetFileName(args.output)
    writer.SetInputConnection(grab.GetOutputPort())
    writer.Write()
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
