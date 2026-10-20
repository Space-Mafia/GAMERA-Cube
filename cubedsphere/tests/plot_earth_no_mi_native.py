#!/usr/bin/env python3
"""Render native-cell XY/XZ sections of a cubed-sphere Earth run.

The section polygons are intersections of the original 3-D hexahedra with
the requested Cartesian plane. No field regridding or image interpolation is
performed. The gnomonic face frames and radial map match cubedsphere_mesh.c
and nonorthogonal_radial_map.c (Earth production v4).
"""

import argparse
import html
import math
from pathlib import Path

import numpy as np


FRAMES = np.asarray([
    [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
    [[-1, 0, 0], [0, -1, 0], [0, 0, 1]],
    [[0, 1, 0], [-1, 0, 0], [0, 0, 1]],
    [[0, -1, 0], [1, 0, 0], [0, 0, 1]],
    [[0, 0, 1], [1, 0, 0], [0, 1, 0]],
    [[0, 0, -1], [1, 0, 0], [0, -1, 0]],
], dtype=float)
EDGES = ((0, 1), (0, 2), (0, 4), (1, 3), (1, 5), (2, 3),
         (2, 6), (3, 7), (4, 5), (4, 6), (5, 7), (6, 7))
B_NORM_NT = 1e9 * 100000.0 * math.sqrt(
    4.0 * math.pi * 1e-7 * 1.67262192369e-21)
PRESSURE_NORM_NPA = 1e9 * 1.67262192369e-21 * 100000.0 ** 2
MOMENT_Z = -3.10e-5 / (B_NORM_NT * 1e-9)


def radial_nodes(ni, inner_uniform_cells=None):
    if ni < 8:
        raise ValueError("Earth-v4 radial resolution must have at least eight cells")
    inner, knot, outer = 2.5, 14.0, 200.0
    # The production-v4 C map fixes the knot at logical 23/72 regardless
    # of resolution. Only an explicitly configured inner-cell count changes
    # it (e.g. the legacy 48-cell/23-inner-cell experiment).
    if inner_uniform_cells is not None and not (0 < inner_uniform_cells < ni):
        raise ValueError("inner uniform cell count must lie inside the grid")
    logical_knot = (23.0 / 72.0 if inner_uniform_cells is None else
                    inner_uniform_cells / ni)
    slope = (knot - inner) / logical_knot
    target = slope * (1.0 - logical_knot) / (outer - knot)
    lo, hi = 0.0, 20.0
    for _ in range(100):
        middle = 0.5 * (lo + hi)
        if middle / math.expm1(middle) > target:
            lo = middle
        else:
            hi = middle
    stretch = 0.5 * (lo + hi)
    nodes = []
    for i in range(ni + 1):
        s = i / ni
        if s <= logical_knot:
            nodes.append(inner + (knot - inner) * s / logical_knot)
        else:
            f = (s - logical_knot) / (1.0 - logical_knot)
            nodes.append(knot + (outer - knot) * math.expm1(stretch * f) /
                         math.expm1(stretch))
    return np.asarray(nodes)


def angular_corners(face, j, k, n_j, n_k=None):
    if n_k is None:
        n_k = n_j
    a = np.asarray([-math.pi / 4 + (j + dj) * math.pi / (2 * n_j)
                    for dj in (0, 1)])
    b = np.asarray([-math.pi / 4 + (k + dk) * math.pi / (2 * n_k)
                    for dk in (0, 1)])
    frame = FRAMES[face]
    result = []
    for aa in a:
        for bb in b:
            direction = frame[0] + math.tan(aa) * frame[1] + math.tan(bb) * frame[2]
            result.append(direction / np.linalg.norm(direction))
    return np.asarray(result)


def section_polygon(corners, r0, r1, normal_axis, shown_axes):
    vertices = np.vstack((r0 * corners, r1 * corners))
    signed = vertices[:, normal_axis]
    if np.min(signed) > 1e-9 or np.max(signed) < -1e-9:
        return None
    points = {}
    for u, v in EDGES:
        du, dv = signed[u], signed[v]
        if abs(du) < 1e-9:
            p = vertices[u, shown_axes]
            points[tuple(np.round(p, 8))] = p
        if abs(dv) < 1e-9:
            p = vertices[v, shown_axes]
            points[tuple(np.round(p, 8))] = p
        if du * dv < -1e-18:
            t = du / (du - dv)
            p = (vertices[u] + t * (vertices[v] - vertices[u]))[shown_axes]
            points[tuple(np.round(p, 8))] = p
    if len(points) < 3:
        return None
    polygon = np.asarray(list(points.values()))
    middle = np.mean(polygon, axis=0)
    order = np.argsort(np.arctan2(polygon[:, 1] - middle[1],
                                  polygon[:, 0] - middle[0]))
    polygon = polygon[order]
    area = 0.5 * abs(np.dot(polygon[:, 0], np.roll(polygon[:, 1], -1)) -
                     np.dot(polygon[:, 1], np.roll(polygon[:, 0], -1)))
    return polygon if area > 1e-9 else None


def blend(a, b, fraction):
    return tuple(round((1 - fraction) * a[i] + fraction * b[i]) for i in range(3))


def diverging_color(value, bound):
    blue, pale, red = (30, 75, 132), (247, 247, 242), (166, 27, 44)
    t = max(-1.0, min(1.0, value / bound))
    return blend(pale, red, t) if t >= 0 else blend(pale, blue, -t)


def pressure_color(value, low, high):
    stops = ((35, 12, 63), (55, 91, 141), (29, 167, 150),
             (197, 211, 61), (255, 224, 70))
    if not math.isfinite(value) or value <= 0:
        return (35, 12, 63)
    t = max(0.0, min(1.0, (math.log10(value) - math.log10(low)) /
                            (math.log10(high) - math.log10(low))))
    segment = min(3, int(t * 4))
    return blend(stops[segment], stops[segment + 1], t * 4 - segment)


def svg_rgb(color):
    return "#%02x%02x%02x" % color


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path)
    parser.add_argument("svg", type=Path)
    parser.add_argument("--time-code", type=float, required=True)
    parser.add_argument("--extent", type=float, default=35.0)
    parser.add_argument("--xy-field", choices=("bz", "density"), default="bz")
    parser.add_argument("--mi", action="store_true",
                        help="Label a snapshot made with aligned-dipole MI coupling")
    parser.add_argument("--inner-uniform-cells", type=int,
                        help="Only for explicitly configured nondefault Earth-v4 maps")
    args = parser.parse_args()
    data = np.genfromtxt(args.csv, delimiter=",", names=True)
    ni = int(np.max(data["i"])) + 1
    na = int(np.max(data["j"])) + 1
    nk = int(np.max(data["k"])) + 1
    if len(data) != 6 * ni * na * nk:
        raise ValueError("Expected a complete six-face snapshot")
    if not all(np.isfinite(data[name]).all() for name in data.dtype.names):
        raise ValueError("Nonfinite snapshot field")
    radii = radial_nodes(ni, args.inner_uniform_cells)
    panels = ((2, (0, 1)), (1, (0, 2)))
    xlo, xhi = -args.extent, args.extent
    ylo, yhi = -args.extent, args.extent
    panel_w = 710
    panel_h = 710
    lefts = (80, 880)
    top = 125
    width, height = 1680, 1000
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
           f'viewBox="0 0 {width} {height}">',
           '<rect width="100%" height="100%" fill="#ffffff"/>']
    code_seconds = 6371000.0 / 100000.0
    model = "Aligned-dipole MI Earth" if args.mi else "No-MI Earth"
    title = (f"{model} cubed-sphere MHD  ·  t={args.time_code * code_seconds:.1f} s "
             f"({args.time_code:.3f} code)  ·  {ni}×{na}×{nk}×6 native cells")
    out.append(f'<text x="840" y="45" text-anchor="middle" '
               f'font-family="Arial" font-size="27" fill="#172b44">{html.escape(title)}</text>')
    subtitle = ("Inner MI EMF + E×B drift; SigmaP=5 S; aligned dipole; idealized SW"
                if args.mi else
                "Hard wall at 2.5 RE; inner E_t=0; idealized solar wind; no MI coupling")
    out.append(f'<text x="840" y="78" text-anchor="middle" '
               f'font-family="Arial" font-size="17" fill="#52677d">{subtitle}</text>')
    for panel, (normal_axis, shown_axes) in enumerate(panels):
        left = lefts[panel]
        out.append(f'<rect x="{left}" y="{top}" width="{panel_w}" height="{panel_h}" '
                   f'fill="#f7f8fa" stroke="#172b44" stroke-width="1"/>')
        out.append(f'<defs><clipPath id="panel-{panel}"><rect x="{left}" '
                   f'y="{top}" width="{panel_w}" height="{panel_h}"/></clipPath></defs>')
        out.append(f'<g clip-path="url(#panel-{panel})">')
        def xy(point):
            x = left + (point[0] - xlo) * panel_w / (xhi - xlo)
            y = top + (yhi - point[1]) * panel_h / (yhi - ylo)
            return x, y
        for face in range(6):
            rows = data[data["face"] == face]
            for j in range(na):
                for k in range(nk):
                    corners = angular_corners(face, j, k, na, nk)
                    if np.min(corners[:, normal_axis]) > 1e-9 or \
                       np.max(corners[:, normal_axis]) < -1e-9:
                        continue
                    column = rows[(rows["j"] == j) & (rows["k"] == k)]
                    for row in column:
                        i = int(row["i"])
                        polygon = section_polygon(corners, radii[i], radii[i + 1],
                                                  normal_axis, shown_axes)
                        if polygon is None:
                            continue
                        if panel == 0:
                            if args.xy_field == "bz":
                                x, y, z = row["x"], row["y"], row["z"]
                                r = math.sqrt(x*x + y*y + z*z)
                                dipole_bz = MOMENT_Z * (3.0 * z*z / r**5 - 1.0 / r**3)
                                value = (row["bz"] - dipole_bz) * B_NORM_NT
                                color = diverging_color(value, 20.0)
                            else:
                                color = pressure_color(row["rho"], 0.1, 30.0)
                        else:
                            value = row["p"] * PRESSURE_NORM_NPA
                            color = pressure_color(value, 1e-4, 2.0)
                        points = " ".join("%.2f,%.2f" % xy(p) for p in polygon)
                        out.append(f'<polygon points="{points}" fill="{svg_rgb(color)}" '
                                   f'stroke="none"/>')
        out.append('</g>')
        earth = panel_w * 2.5 / (xhi - xlo)
        cx, cy = xy((0.0, 0.0))
        out.append(f'<circle cx="{cx:.2f}" cy="{cy:.2f}" r="{earth:.2f}" '
                   f'fill="#20242a" stroke="#ffffff" stroke-width="1.5"/>')
        for tick in range(-int(args.extent), int(args.extent) + 1, 10):
            tx, _ = xy((tick, 0.0))
            _, ty = xy((0.0, tick))
            out.append(f'<text x="{tx:.1f}" y="{top + panel_h + 27}" '
                       f'text-anchor="middle" font-family="Arial" font-size="13">{tick}</text>')
            out.append(f'<text x="{left - 10}" y="{ty + 5:.1f}" '
                       f'text-anchor="end" font-family="Arial" font-size="13">{tick}</text>')
        xlabel = "X / RE"
        ylabel = "Y / RE" if panel == 0 else "Z / RE"
        out.append(f'<text x="{left + panel_w / 2}" y="{top + panel_h + 56}" '
                   f'text-anchor="middle" font-family="Arial" font-size="17">{xlabel}</text>')
        out.append(f'<text x="{left - 55}" y="{top + panel_h / 2}" '
                   f'transform="rotate(-90 {left - 55} {top + panel_h / 2})" '
                   f'text-anchor="middle" font-family="Arial" font-size="17">{ylabel}</text>')
        panel_title = (("XY: Bz − dipole Bz [nT]" if args.xy_field == "bz"
                        else "XY: density [cm⁻³]") if panel == 0 else
                       "XZ: thermal pressure [nPa]")
        out.append(f'<text x="{left + panel_w / 2}" y="{top - 12}" '
                   f'text-anchor="middle" font-family="Arial" font-size="19">'
                   f'{html.escape(panel_title)}</text>')
        color_y = top + panel_h + 80
        for q in range(256):
            if panel == 0:
                color = (diverging_color(-20.0 + 40.0 * q / 255.0, 20.0)
                         if args.xy_field == "bz" else
                         pressure_color(10 ** (-1 + math.log10(300) * q / 255.0),
                                        0.1, 30.0))
            else:
                color = pressure_color(10 ** (-4 + math.log10(20000) * q / 255.0),
                                       1e-4, 2.0)
            out.append(f'<rect x="{left + q * panel_w / 256:.2f}" y="{color_y}" '
                       f'width="{panel_w / 256 + 0.05:.2f}" height="18" '
                       f'fill="{svg_rgb(color)}"/>')
        legend = (("−20                    0                    +20 nT" if args.xy_field == "bz"
                   else "0.1                 1                  30 cm⁻³") if panel == 0 else
                  "10⁻⁴               10⁻²                 2 nPa")
        out.append(f'<text x="{left + panel_w / 2}" y="{color_y + 42}" '
                   f'text-anchor="middle" font-family="Arial" font-size="15">{legend}</text>')
    out.append('</svg>')
    args.svg.parent.mkdir(parents=True, exist_ok=True)
    args.svg.write_text("\n".join(out), encoding="utf-8")
    print(args.svg)


if __name__ == "__main__":
    main()
