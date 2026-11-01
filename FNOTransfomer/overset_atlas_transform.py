"""Topology-aware overlapping-atlas transform for two-body TSTO flow fields.

The physical domain is not forced into one Cartesian rectangle.  It is covered
by four curvilinear charts: one periodic O-chart per body, one periodic outer
annulus around the combined convex envelope, and one H-chart spanning the
inter-body gap.  Overlap transition maps and a partition of unity provide the
forward/inverse coupling needed by a neural operator.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.spatial import ConvexHull

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from cgns_tsto_io import (
    ensure_hdf5_cgns,
    interpolate_field,
    make_interpolators,
    parse_cgns,
)
from mesh_transform import _resample_closed, _signed_area


plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif", "Liberation Serif"],
        "font.size": 8,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
    }
)


FIELDS = ("Mach", "Pressure", "Density", "VelocityX", "VelocityY", "Temperature")


def logical_metrics(x_map: np.ndarray, y_map: np.ndarray, periodic: bool) -> dict[str, np.ndarray]:
    ny, nx = x_map.shape
    eta = np.linspace(0.0, 1.0, ny)
    if periodic:
        xi = np.arange(nx, dtype=np.float64) / nx
        dxi = 1.0 / nx
        x_xi = (np.roll(x_map, -1, axis=1) - np.roll(x_map, 1, axis=1)) / (2.0 * dxi)
        y_xi = (np.roll(y_map, -1, axis=1) - np.roll(y_map, 1, axis=1)) / (2.0 * dxi)
    else:
        xi = np.linspace(0.0, 1.0, nx)
        x_xi = np.gradient(x_map, xi, axis=1)
        y_xi = np.gradient(y_map, xi, axis=1)
    x_eta = np.gradient(x_map, eta, axis=0)
    y_eta = np.gradient(y_map, eta, axis=0)
    jacobian = x_xi * y_eta - x_eta * y_xi
    return {
        "xi": xi,
        "eta": eta,
        "x_xi": x_xi,
        "x_eta": x_eta,
        "y_xi": y_xi,
        "y_eta": y_eta,
        "J": jacobian,
    }


def orient_positive(chart: dict) -> dict:
    if np.nanmedian(chart["J"]) < 0.0:
        for key, values in list(chart.items()):
            if isinstance(values, np.ndarray) and values.ndim == 2:
                chart[key] = values[:, ::-1]
        chart.update(logical_metrics(chart["x"], chart["y"], chart["periodic"]))
    return chart


def _oriented_boundary(data: dict, loop: np.ndarray) -> np.ndarray:
    source_wall = np.column_stack([data["x"][loop], data["y"][loop]])
    if _signed_area(source_wall) > 0.0:
        source_wall = source_wall[::-1]
    return source_wall


def _resample_closed_at_phase(points: np.ndarray, count: int, target: np.ndarray) -> np.ndarray:
    """Resample a closed polygon with a continuous, geometry-defined phase.

    The phase is the closest point on the *piecewise-linear* boundary to
    ``target``.  Rebuilding the cumulative arc length from that point before
    sampling avoids the half-cell phase error caused by resampling first and
    applying an integer cyclic shift afterwards.
    """
    points = np.asarray(points, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    next_points = np.roll(points, -1, axis=0)
    tangent = next_points - points
    denominator = np.sum(tangent * tangent, axis=1)
    fraction = np.zeros(len(points), dtype=np.float64)
    valid = denominator > 1.0e-30
    fraction[valid] = np.clip(
        np.sum((target[None, :] - points[valid]) * tangent[valid], axis=1)
        / denominator[valid],
        0.0,
        1.0,
    )
    projections = points + fraction[:, None] * tangent
    segment_id = int(np.argmin(np.sum((projections - target[None, :]) ** 2, axis=1)))
    phase_point = projections[segment_id]
    ordered = [phase_point]
    ordered.extend(points[(segment_id + 1 + index) % len(points)] for index in range(len(points)))
    ordered = np.asarray(ordered, dtype=np.float64)
    extended = np.vstack([ordered, ordered[0]])
    segment = np.linalg.norm(np.diff(extended, axis=0), axis=1)
    cumulative = np.concatenate([[0.0], np.cumsum(segment)])
    samples = np.linspace(0.0, cumulative[-1], count, endpoint=False)
    return np.column_stack(
        [np.interp(samples, cumulative, extended[:, axis]) for axis in range(2)]
    )


def _geometric_phase_target(points: np.ndarray) -> np.ndarray:
    # The lexicographic minimum is a deterministic geometric landmark.  The
    # subsequent continuous projection handles nonuniform boundary spacing.
    return points[np.lexsort((points[:, 1], points[:, 0]))[0]]


def _phase_align_closed(wall: np.ndarray, reference: np.ndarray | None) -> np.ndarray:
    if reference is None:
        return wall
    if wall.shape != reference.shape:
        raise ValueError("Closed-curve phase reference shape does not match the chart resolution")
    # Track the reference landmark after removing the instantaneous centroid
    # translation.  Unlike a raw node id, this remains meaningful when the
    # mesh writer renumbers boundary edges; unlike an unconstrained global
    # correlation, it cannot jump to the opposite side of a sharp leading edge.
    target = reference[0] + (np.mean(wall, axis=0) - np.mean(reference, axis=0))
    landmark = int(np.argmin(np.sum((wall - target[None, :]) ** 2, axis=1)))
    # ``wall`` is already phase-resampled by _resample_closed_at_phase.  This
    # function is retained as a compatibility guard for callers that provide
    # an already sampled curve; it now only validates the reference contract.
    return wall


def positive_body_chart(
    data: dict,
    loop: np.ndarray,
    nx: int,
    ny: int,
    requested_offset: float,
    phase_reference: np.ndarray | None = None,
) -> dict:
    boundary = _oriented_boundary(data, loop)
    if phase_reference is None:
        wall = _resample_closed_at_phase(boundary, nx, _geometric_phase_target(boundary))
    else:
        target = np.asarray(phase_reference[0], dtype=np.float64).copy()
        provisional = _resample_closed_at_phase(boundary, nx, _geometric_phase_target(boundary))
        target += np.mean(provisional, axis=0) - np.mean(np.asarray(phase_reference), axis=0)
        wall = _resample_closed_at_phase(boundary, nx, target)
    smoothing_width = max(2.0, nx / 64.0)
    smoothed_wall = np.column_stack(
        [
            gaussian_filter1d(wall[:, 0], smoothing_width, mode="wrap"),
            gaussian_filter1d(wall[:, 1], smoothing_width, mode="wrap"),
        ]
    )
    tangent = np.roll(smoothed_wall, -1, axis=0) - np.roll(smoothed_wall, 1, axis=0)
    normal = np.column_stack([-tangent[:, 1], tangent[:, 0]])
    normal /= np.maximum(np.linalg.norm(normal, axis=1, keepdims=True), 1.0e-14)
    eta = np.linspace(0.0, 1.0, ny)
    radial_fraction = np.sinh(2.0 * eta) / np.sinh(2.0)
    offset = requested_offset
    while offset >= requested_offset * (0.35 / 3.0):
        outer = wall + offset * normal
        chart = {
            "x": (1.0 - radial_fraction[:, None]) * wall[None, :, 0]
            + radial_fraction[:, None] * outer[None, :, 0],
            "y": (1.0 - radial_fraction[:, None]) * wall[None, :, 1]
            + radial_fraction[:, None] * outer[None, :, 1],
            "periodic": True,
            "selected_offset": offset,
            "requested_offset": requested_offset,
            "normal_smoothing_width": smoothing_width,
            "phase_alignment": "cyclic_shape_registration_to_sequence_reference"
            if phase_reference is not None
            else "lexicographic_geometric_landmark",
        }
        chart.update(logical_metrics(chart["x"], chart["y"], True))
        chart = orient_positive(chart)
        triangles = structured_triangles(ny, nx, True)
        points = np.column_stack([chart["x"].ravel(), chart["y"].ravel()])
        first = points[triangles[:, 0]]
        second = points[triangles[:, 1]]
        third = points[triangles[:, 2]]
        signed_areas = 0.5 * cross_2d(second - first, third - first)
        chart["cell_signed_area_min"] = float(np.min(signed_areas))
        chart["cell_nonpositive_count"] = int(np.count_nonzero(signed_areas <= 0.0))
        if np.all(chart["J"] > 0.0) and chart["cell_nonpositive_count"] == 0:
            return chart
        offset *= 0.85
    raise RuntimeError("No positive-Jacobian body O-chart could be constructed")


def cross_2d(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    return first[..., 0] * second[..., 1] - first[..., 1] * second[..., 0]


def ray_polygon_radius(polygon: np.ndarray, center: np.ndarray, angles: np.ndarray) -> np.ndarray:
    starts = polygon
    segments = np.roll(polygon, -1, axis=0) - polygon
    relative = starts - center
    radii = np.empty(len(angles), dtype=np.float64)
    for index, angle in enumerate(angles):
        direction = np.array([np.cos(angle), np.sin(angle)])
        denominators = cross_2d(np.broadcast_to(direction, segments.shape), segments)
        valid_denominator = np.abs(denominators) > 1.0e-12
        ray_parameter = np.full(len(segments), np.nan)
        edge_parameter = np.full(len(segments), np.nan)
        ray_parameter[valid_denominator] = cross_2d(relative[valid_denominator], segments[valid_denominator]) / denominators[valid_denominator]
        edge_parameter[valid_denominator] = cross_2d(relative[valid_denominator], np.broadcast_to(direction, relative[valid_denominator].shape)) / denominators[valid_denominator]
        valid = valid_denominator & (ray_parameter > 0.0) & (edge_parameter >= -1.0e-10) & (edge_parameter <= 1.0 + 1.0e-10)
        if not np.any(valid):
            raise RuntimeError(f"Ray at angle {angle} does not intersect polygon")
        radii[index] = float(np.min(ray_parameter[valid]))
    return radii


def _body_hull(data: dict) -> np.ndarray:
    body_points = np.vstack(
        [np.column_stack([data["x"][loop], data["y"][loop]]) for loop in data["wall_loops"].values()]
    )
    return body_points[ConvexHull(body_points).vertices]


def sequence_anchor(data_sequence: list[dict], nx: int, envelope_margin: float) -> dict:
    """Build geometry-only parameters shared by every frame in one sequence.

    The anchor is deliberately independent of flow variables.  A single center,
    angular phase, and bridge interval prevents a frame-by-frame reparameteriza-
    tion from being mistaken for physical grid motion.  The union hull is used
    only to choose a safe fixed center and x interval; each frame still supplies
    its own instantaneous body geometry.
    """
    all_body_points = np.vstack(
        [
            np.vstack(
                [np.column_stack([data["x"][loop], data["y"][loop]]) for loop in data["wall_loops"].values()]
            )
            for data in data_sequence
        ]
    )
    union_hull = all_body_points[ConvexHull(all_body_points).vertices]
    center = np.mean(union_hull, axis=0)
    # The center must be inside every instantaneous envelope.  For a moving
    # body sequence, the intersection of instantaneous hulls is the safe set.
    # The centroid of the reference hull is the least model-dependent choice;
    # reject a sequence later if a ray has no valid intersection.
    reference_hull = _body_hull(data_sequence[0])
    reference_center = np.mean(reference_hull, axis=0)
    outer_polygon = np.column_stack(
        [data_sequence[0]["x"][data_sequence[0]["outer_loop"]], data_sequence[0]["y"][data_sequence[0]["outer_loop"]]]
    )
    angles = np.linspace(0.0, 2.0 * np.pi, nx, endpoint=False, dtype=np.float64)
    union_inner_radius = ray_polygon_radius(union_hull, reference_center, angles) + envelope_margin
    body_reference = {}
    for name, loop in data_sequence[0]["wall_loops"].items():
        boundary = _oriented_boundary(data_sequence[0], loop)
        body_reference[name] = _resample_closed_at_phase(
            boundary, 384, _geometric_phase_target(boundary)
        ).astype(np.float64)
    return {
        "center": reference_center.astype(np.float64),
        "angles": angles,
        "outer_polygon": outer_polygon.astype(np.float64),
        "inner_radius": union_inner_radius.astype(np.float64),
        "body_reference": body_reference,
        "gap_x_min": float(np.min(all_body_points[:, 0])),
        "gap_x_max": float(np.max(all_body_points[:, 0])),
        "envelope_margin": float(envelope_margin),
        "anchor_type": "sequence_reference_centroid_fixed_angle_arc",
    }


def outer_annulus_chart(
    data: dict, nx: int, ny: int, envelope_margin: float, anchor: dict | None = None,
    radial_alpha: float = 0.05,
) -> dict:
    inner_polygon = _body_hull(data)
    outer_loop = data["outer_loop"]
    outer_polygon = np.column_stack([data["x"][outer_loop], data["y"][outer_loop]])
    if anchor is None:
        center = np.mean(inner_polygon, axis=0)
        angles = np.linspace(0.0, 2.0 * np.pi, nx, endpoint=False)
        reference_outer_polygon = outer_polygon
        inner_radius = ray_polygon_radius(inner_polygon, center, angles) + envelope_margin
    else:
        center = np.asarray(anchor["center"], dtype=np.float64)
        angles = np.asarray(anchor["angles"], dtype=np.float64)
        if len(angles) != nx:
            raise ValueError("The sequence anchor angular resolution does not match nx")
        reference_outer_polygon = np.asarray(anchor["outer_polygon"], dtype=np.float64)
        envelope_margin = float(anchor["envelope_margin"])
        # A fixed envelope is the geometric domain of the outer chart for the
        # complete sequence.  It encloses every instantaneous body position,
        # so the chart does not acquire artificial radial motion from a changing
        # convex-hull vertex or center.
        inner_radius = np.asarray(anchor["inner_radius"], dtype=np.float64)
        if len(inner_radius) != nx:
            raise ValueError("The sequence anchor inner-envelope resolution does not match nx")
    outer_radius = ray_polygon_radius(reference_outer_polygon, center, angles)
    if np.any(inner_radius >= outer_radius):
        raise RuntimeError("Combined inner envelope reaches the farfield boundary")
    eta = np.linspace(0.0, 1.0, ny)
    if not 0.0 <= radial_alpha <= 1.0:
        raise ValueError("radial_alpha must be in [0, 1]")
    radial_fraction = radial_alpha * eta + (1.0 - radial_alpha) * 0.5 * (1.0 - np.cos(np.pi * eta))
    radius = (1.0 - radial_fraction[:, None]) * inner_radius + radial_fraction[:, None] * outer_radius
    x_map = center[0] + radius * np.cos(angles)[None, :]
    y_map = center[1] + radius * np.sin(angles)[None, :]
    chart = {
        "x": x_map,
        "y": y_map,
        "periodic": True,
        "center": center,
        "inner_radius": inner_radius,
        "outer_radius": outer_radius,
        "inner_x": center[0] + inner_radius * np.cos(angles),
        "inner_y": center[1] + inner_radius * np.sin(angles),
        "parameterization": "fixed sequence anchor center and angular coordinate",
        "radial_alpha": radial_alpha,
        "continuous_boundary_radial_derivative_min": float(radial_alpha * np.min(outer_radius - inner_radius)),
    }
    chart.update(logical_metrics(x_map, y_map, True))
    return orient_positive(chart)


def vertical_intersections(polygon: np.ndarray, x_values: np.ndarray) -> list[np.ndarray]:
    first = polygon
    second = np.roll(polygon, -1, axis=0)
    output = []
    for x_value in x_values:
        crosses = ((first[:, 0] <= x_value) & (second[:, 0] > x_value)) | (
            (second[:, 0] <= x_value) & (first[:, 0] > x_value)
        )
        fraction = (x_value - first[crosses, 0]) / (second[crosses, 0] - first[crosses, 0])
        values = first[crosses, 1] + fraction * (second[crosses, 1] - first[crosses, 1])
        output.append(np.sort(values))
    return output


def gap_h_chart(
    data: dict, envelope_chart: dict, nx: int, ny: int, anchor: dict | None = None
) -> dict:
    loops = list(data["wall_loops"].items())
    if len(loops) != 2:
        raise ValueError("This H-chart template requires exactly two bodies; general multi-body bridging is not implemented")
    polygons = {
        name: np.column_stack([data["x"][loop], data["y"][loop]]) for name, loop in loops
    }
    lower_name, upper_name = sorted(polygons, key=lambda name: np.mean(polygons[name][:, 1]))
    lower_polygon = polygons[lower_name]
    upper_polygon = polygons[upper_name]
    envelope_polygon = np.column_stack([envelope_chart["inner_x"], envelope_chart["inner_y"]])
    if anchor is None:
        xmin = float(np.min(envelope_polygon[:, 0]))
        xmax = float(np.max(envelope_polygon[:, 0]))
    else:
        xmin = float(anchor["gap_x_min"])
        xmax = float(anchor["gap_x_max"])
    xmin = max(xmin, float(np.min(envelope_polygon[:, 0])))
    xmax = min(xmax, float(np.max(envelope_polygon[:, 0])))
    padding = 0.015 * (xmax - xmin)
    x_line = np.linspace(xmin + padding, xmax - padding, nx)
    envelope_hits = vertical_intersections(envelope_polygon, x_line)
    lower_hits = vertical_intersections(lower_polygon, x_line)
    upper_hits = vertical_intersections(upper_polygon, x_line)
    if any(len(values) < 2 for values in envelope_hits):
        raise RuntimeError("The combined envelope cannot define the bridge H-chart")
    lower_surface = np.asarray(
        [body[-1] if len(body) >= 2 else envelope[0] for body, envelope in zip(lower_hits, envelope_hits)]
    )
    upper_surface = np.asarray(
        [body[0] if len(body) >= 2 else envelope[-1] for body, envelope in zip(upper_hits, envelope_hits)]
    )
    gap = upper_surface - lower_surface
    if np.any(gap <= 0.0):
        raise RuntimeError("The bodies overlap in the proposed H-chart direction")
    eta = np.linspace(0.0, 1.0, ny)
    x_map = np.broadcast_to(x_line, (ny, nx)).copy()
    y_map = (1.0 - eta[:, None]) * lower_surface + eta[:, None] * upper_surface
    chart = {
        "x": x_map,
        "y": y_map,
        "periodic": False,
        "lower_body": lower_name,
        "upper_body": upper_name,
        "gap_min": float(np.min(gap)),
        "gap_max": float(np.max(gap)),
        "bridge_definition": "body-facing walls where present; combined-envelope boundary elsewhere",
        "parameterization": "fixed sequence x-line with instantaneous wall intersections",
    }
    chart.update(logical_metrics(x_map, y_map, False))
    return orient_positive(chart)


def add_fields(data: dict, charts: dict[str, dict]) -> None:
    interpolators, tree = make_interpolators(data)
    for chart in charts.values():
        chart["fields"] = {}
        for field_name in FIELDS:
            if field_name in interpolators:
                chart["fields"][field_name] = interpolate_field(
                    interpolators[field_name], tree, data, field_name, chart["x"], chart["y"]
                )


def structured_triangles(ny: int, nx: int, periodic: bool) -> np.ndarray:
    last_column = nx if periodic else nx - 1
    row, column = np.meshgrid(np.arange(ny - 1), np.arange(last_column), indexing="ij")
    lower_left = (row * nx + column).ravel()
    lower_right = (row * nx + (column + 1) % nx).ravel()
    triangles = np.empty((2 * len(lower_left), 3), dtype=np.int64)
    triangles[::2] = np.column_stack([lower_left, lower_right, lower_right + nx])
    triangles[1::2] = np.column_stack([lower_left, lower_right + nx, lower_left + nx])
    return triangles


def inverse_chart_values(chart: dict, query_x: np.ndarray, query_y: np.ndarray, field_name: str):
    if "inverse_cache" in chart and chart.get("inverse_cache_query") == (id(query_x), id(query_y)):
        eta_values, xi_values, donor_ids, barycentric = chart["inverse_cache"]
        valid = donor_ids[:, 0] >= 0
        result = np.full(len(query_x), np.nan)
        result[valid] = np.sum(chart["fields"][field_name].ravel()[donor_ids[valid]] * barycentric[valid], axis=1)
        return result, eta_values, xi_values, donor_ids, barycentric
    ny, nx = chart["x"].shape
    triangles = structured_triangles(ny, nx, chart["periodic"])
    if chart.get("cartesian", False):
        scaled_x = (query_x - chart["x"][0, 0]) / (chart["x"][0, -1] - chart["x"][0, 0]) * (nx - 1)
        scaled_y = (query_y - chart["y"][0, 0]) / (chart["y"][-1, 0] - chart["y"][0, 0]) * (ny - 1)
        column = np.clip(np.floor(scaled_x).astype(int), 0, nx - 2)
        row = np.clip(np.floor(scaled_y).astype(int), 0, ny - 2)
        triangle_ids = 2 * (row * (nx - 1) + column) + ((scaled_y - row) > (scaled_x - column))
        valid_query = (scaled_x >= 0) & (scaled_x <= nx - 1) & (scaled_y >= 0) & (scaled_y <= ny - 1)
        valid_query &= chart["valid_donor_cells"][triangle_ids]
        triangle_ids[~valid_query] = -1
    else:
        triangulation = mtri.Triangulation(chart["x"].ravel(), chart["y"].ravel(), triangles)
        if "valid_donor_cells" in chart:
            triangulation.set_mask(~chart["valid_donor_cells"])
        triangle_ids = triangulation.get_trifinder()(query_x, query_y).astype(np.int64)
    valid = triangle_ids >= 0
    donor_ids = np.full((len(query_x), 3), -1, dtype=np.int32)
    barycentric = np.zeros((len(query_x), 3), dtype=np.float64)
    donor_ids[valid] = triangles[triangle_ids[valid]].astype(np.int32)
    points = np.column_stack([chart["x"].ravel(), chart["y"].ravel()])
    first = points[donor_ids[valid, 0]]
    second = points[donor_ids[valid, 1]]
    third = points[donor_ids[valid, 2]]
    query = np.column_stack([query_x[valid], query_y[valid]])
    denominator = cross_2d(second - first, third - first)
    weight_second = cross_2d(query - first, third - first) / denominator
    weight_third = cross_2d(second - first, query - first) / denominator
    weight_first = 1.0 - weight_second - weight_third
    barycentric[valid] = np.column_stack([weight_first, weight_second, weight_third])

    def interpolate(values: np.ndarray) -> np.ndarray:
        result = np.full(len(query_x), np.nan, dtype=np.float64)
        flat = values.ravel()
        result[valid] = np.sum(flat[donor_ids[valid]] * barycentric[valid], axis=1)
        return result

    eta_grid = np.broadcast_to(chart["eta"][:, None], (ny, nx))
    xi_grid = np.broadcast_to(chart["xi"][None, :], (ny, nx))
    xi_values = interpolate(xi_grid)
    if chart["periodic"]:
        donor_xi = xi_grid.ravel()[donor_ids[valid]].copy()
        crosses_seam = np.ptp(donor_xi, axis=1) > 0.5
        donor_xi[crosses_seam] += (donor_xi[crosses_seam] < 0.5)
        xi_values[valid] = np.mod(np.sum(donor_xi * barycentric[valid], axis=1), 1.0)
    chart["inverse_cache"] = (interpolate(eta_grid), xi_values, donor_ids, barycentric)
    chart["inverse_cache_query"] = (id(query_x), id(query_y))
    return (
        interpolate(chart["fields"][field_name]),
        interpolate(eta_grid),
        xi_values,
        donor_ids,
        barycentric,
    )


def chart_weight(name: str, eta: np.ndarray, xi: np.ndarray) -> np.ndarray:
    if name in {"smj", "xmj"}:
        raw = np.cos(0.5 * np.pi * np.clip(eta, 0.0, 1.0)) ** 2
    elif name == "outer":
        raw = np.sin(0.5 * np.pi * np.clip(eta, 0.0, 1.0)) ** 2
    else:
        cross_gap = np.sin(np.pi * np.clip(eta, 0.0, 1.0)) ** 2
        streamwise = np.sin(np.pi * np.clip(xi, 0.0, 1.0)) ** 2
        raw = 0.35 + 0.65 * cross_gap * streamwise
    return np.where(np.isfinite(eta), 0.03 + 0.97 * raw, 0.0)


def roundtrip(data: dict, charts: dict[str, dict], field_name: str = "Mach"):
    query_x = data["x"]
    query_y = data["y"]
    weighted_sum = np.zeros_like(query_x, dtype=np.float64)
    weight_sum = np.zeros_like(query_x, dtype=np.float64)
    membership = np.zeros_like(query_x, dtype=np.int16)
    chart_coverage = {}
    transitions = {}
    for name, chart in charts.items():
        values, eta_values, xi_values, donor_ids, barycentric = inverse_chart_values(
            chart, query_x, query_y, field_name
        )
        valid = np.isfinite(values)
        weights = chart_weight(name, eta_values, xi_values)
        weights[~valid] = 0.0
        weighted_sum += np.nan_to_num(values, nan=0.0) * weights
        weight_sum += weights
        membership += valid.astype(np.int16)
        chart_coverage[name] = float(np.mean(valid))
        transitions[name] = {
            "donor_ids": donor_ids,
            "barycentric_weights": barycentric.astype(np.float64),
            "xi": xi_values.astype(np.float64),
            "eta": eta_values.astype(np.float64),
            "partition_weight": weights.astype(np.float64),
        }
    reconstructed = np.full_like(query_x, np.nan, dtype=np.float64)
    covered = weight_sum > 0.0
    reconstructed[covered] = weighted_sum[covered] / weight_sum[covered]
    for transition in transitions.values():
        normalized = np.zeros_like(transition["partition_weight"])
        normalized[covered] = transition["partition_weight"][covered] / weight_sum[covered]
        transition["normalized_partition_weight"] = normalized.astype(np.float64)
    source = data["fields"][field_name]
    error = reconstructed[covered] - source[covered]
    scale = float(np.nanmax(source) - np.nanmin(source))
    report = {
        "field": field_name,
        "coverage_fraction": float(np.mean(covered)),
        "overlap_fraction": float(np.mean(membership >= 2)),
        "maximum_chart_multiplicity": int(np.max(membership)),
        "chart_coverage_fraction": chart_coverage,
        "rmse": float(np.sqrt(np.mean(error**2))),
        "normalized_rmse_by_range": float(np.sqrt(np.mean(error**2)) / max(scale, 1.0e-30)),
        "mae": float(np.mean(np.abs(error))),
        "maximum_absolute_error": float(np.max(np.abs(error))),
    }
    return report, reconstructed, transitions


def save_transitions(transitions: dict[str, dict], output_dir: Path) -> None:
    payload = {}
    for chart_name, transition in transitions.items():
        for key, values in transition.items():
            payload[f"{chart_name}_{key}"] = values
    np.savez_compressed(output_dir / "overset_transition_operator.npz", **payload)


def transition_quality(transitions: dict[str, dict]) -> dict:
    barycentric_sum_error = 0.0
    minimum_barycentric_weight = np.inf
    partition_sum = None
    for transition in transitions.values():
        valid = transition["donor_ids"][:, 0] >= 0
        barycentric = transition["barycentric_weights"][valid]
        barycentric_sum_error = max(
            barycentric_sum_error, float(np.max(np.abs(np.sum(barycentric, axis=1) - 1.0)))
        )
        minimum_barycentric_weight = min(
            minimum_barycentric_weight, float(np.min(barycentric))
        )
        if partition_sum is None:
            partition_sum = transition["normalized_partition_weight"].astype(np.float64)
        else:
            partition_sum += transition["normalized_partition_weight"]
    covered = partition_sum > 0.0
    return {
        "maximum_barycentric_weights_sum_error": barycentric_sum_error,
        "minimum_barycentric_weight": minimum_barycentric_weight,
        "maximum_partition_of_unity_error": float(
            np.max(np.abs(partition_sum[covered] - 1.0))
        ),
    }


def save_charts(charts: dict[str, dict], output_dir: Path) -> None:
    payload = {}
    for name, chart in charts.items():
        for key in ("x", "y", "xi", "eta", "x_xi", "x_eta", "y_xi", "y_eta", "J"):
            # Coordinates and metric terms are retained in float64.  Down-casting
            # them to float32 would contaminate the temporal GCL audit before any
            # physical interpolation error is measured.
            payload[f"{name}_{key}"] = np.asarray(chart[key], dtype=np.float64)
        for field_name, values in chart["fields"].items():
            payload[f"{name}_{field_name}"] = values.astype(np.float32)
    np.savez_compressed(output_dir / "overset_atlas_charts.npz", **payload)


def plot_validation(data: dict, charts: dict[str, dict], reconstructed: np.ndarray, report: dict, output_dir: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(14.5, 8.4))
    fig.subplots_adjust(left=0.05, right=0.97, bottom=0.075, top=0.89, wspace=0.31, hspace=0.43)
    axes = axes.ravel()
    colors = {"smj": "#C43C39", "xmj": "#275D8C", "outer": "#527A3D", "gap": "#7A5195"}
    triangulation = mtri.Triangulation(data["x"], data["y"], data["triangles"])
    mach = data["fields"]["Mach"]
    vmin, vmax = np.nanpercentile(mach, [1.0, 99.0])

    for index, axis in enumerate(axes):
        axis.text(-0.10, 1.03, chr(ord("a") + index), transform=axis.transAxes, fontweight="bold", fontsize=10)

    image = axes[0].tripcolor(triangulation, mach, shading="gouraud", cmap="turbo", vmin=vmin, vmax=vmax, rasterized=True)
    axes[0].set(title="Original physical CGNS Mach field", xlabel="physical x", ylabel="physical y", aspect="equal")
    fig.colorbar(image, ax=axes[0], fraction=0.046, pad=0.03, label="Mach")

    for name, chart in charts.items():
        for row in np.linspace(0, chart["x"].shape[0] - 1, 9, dtype=int):
            axes[1].plot(chart["x"][row], chart["y"][row], color=colors[name], linewidth=0.45, alpha=0.65)
        for column in np.linspace(0, chart["x"].shape[1] - 1, 14, dtype=int):
            axes[1].plot(chart["x"][:, column], chart["y"][:, column], color=colors[name], linewidth=0.35, alpha=0.45)
        axes[1].plot([], [], color=colors[name], label=name)
    axes[1].set(title="Four-chart overlapping physical atlas", xlabel="physical x", ylabel="physical y", aspect="equal")
    axes[1].legend(ncol=2)

    image = None
    for chart_index, (name, chart) in enumerate(charts.items()):
        extent = (chart_index * 1.12, chart_index * 1.12 + 1.0, 0.0, 1.0)
        image = axes[2].imshow(chart["fields"]["Mach"], origin="lower", extent=extent, aspect="auto", cmap="turbo", vmin=vmin, vmax=vmax, rasterized=True)
        axes[2].text(
            chart_index * 1.12 + 0.5,
            0.975,
            name,
            ha="center",
            va="top",
            color=colors[name],
            bbox={"facecolor": "white", "alpha": 0.72, "edgecolor": "none", "pad": 1.2},
        )
    axes[2].set(title="Neural-readable computational charts", xlabel="chart-local coordinate", ylabel=r"$\eta$", xlim=(-0.05, 4.42))
    fig.colorbar(image, ax=axes[2], fraction=0.046, pad=0.03, label="Mach")

    image = axes[3].tripcolor(triangulation, reconstructed, shading="gouraud", cmap="turbo", vmin=vmin, vmax=vmax, rasterized=True)
    axes[3].set(title="Atlas inverse map on original FE topology", xlabel="physical x", ylabel="physical y", aspect="equal")
    fig.colorbar(image, ax=axes[3], fraction=0.046, pad=0.03, label="Mach")

    absolute_error = np.abs(reconstructed - mach)
    image = axes[4].tripcolor(triangulation, absolute_error, shading="gouraud", cmap="magma", rasterized=True)
    axes[4].set(title="Forward/inverse Mach absolute error", xlabel="physical x", ylabel="physical y", aspect="equal")
    fig.colorbar(image, ax=axes[4], fraction=0.046, pad=0.03, label="absolute error")

    axes[5].axis("off")
    lines = [
        "Topology-aware overlapping atlas", "",
        f"coverage = {report['roundtrip']['coverage_fraction']:.3%}",
        f"multi-chart overlap = {report['roundtrip']['overlap_fraction']:.3%}",
        f"Mach NRMSE = {report['roundtrip']['normalized_rmse_by_range']:.3e}", "",
    ]
    for name, values in report["charts"].items():
        lines.append(f"{name}: J_min = {values['jacobian_min']:.3e}")
    lines.extend([
        "", "No Cartesian background is used.",
        "Each chart is a local bijection; chart", "transitions handle the two-body topology.",
        "The coupling graph is reconstructed", "geometrically, not read from Fluent."
    ])
    axes[5].text(0.03, 0.97, "\n".join(lines), va="top", fontsize=9, linespacing=1.4)
    fig.suptitle("TSTO Ma5.0 | topology-aware overlapping curvilinear atlas validation", fontsize=13)
    fig.savefig(output_dir / "overset_atlas_validation.png", dpi=300)
    fig.savefig(output_dir / "overset_atlas_validation.svg")
    plt.close(fig)


def process_snapshot(
    source: Path, output_dir: Path, cache_dir: Path, anchor: dict | None = None
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    converted = ensure_hdf5_cgns(source, cache_dir)
    data = parse_cgns(converted)
    outer_chart = outer_annulus_chart(data, 512, 192, 0.8, anchor=anchor)
    charts = {
        "smj": positive_body_chart(
            data,
            data["wall_loops"]["smj"],
            384,
            144,
            3.0,
            phase_reference=(anchor or {}).get("body_reference", {}).get("smj"),
        ),
        "xmj": positive_body_chart(
            data,
            data["wall_loops"]["xmj"],
            384,
            144,
            3.0,
            phase_reference=(anchor or {}).get("body_reference", {}).get("xmj"),
        ),
        "outer": outer_chart,
        "gap": gap_h_chart(data, outer_chart, 512, 192, anchor=anchor),
    }
    add_fields(data, charts)
    roundtrip_report, reconstructed, transitions = roundtrip(data, charts)
    chart_report = {}
    for name, chart in charts.items():
        triangles = structured_triangles(*chart["x"].shape, chart["periodic"])
        points = np.column_stack([chart["x"].ravel(), chart["y"].ravel()])
        first = points[triangles[:, 0]]
        second = points[triangles[:, 1]]
        third = points[triangles[:, 2]]
        cell_areas = 0.5 * cross_2d(second - first, third - first)
        chart_report[name] = {
            "shape": [int(chart["x"].shape[1]), int(chart["x"].shape[0])],
            "periodic_xi": bool(chart["periodic"]),
            "jacobian_min": float(np.min(chart["J"])),
            "jacobian_max": float(np.max(chart["J"])),
            "jacobian_nonpositive_count": int(np.count_nonzero(chart["J"] <= 0.0)),
            "minimum_cell_signed_area": float(np.min(cell_areas)),
            "nonpositive_cell_count": int(np.count_nonzero(cell_areas <= 0.0)),
        }
        if "selected_offset" in chart:
            chart_report[name]["selected_offset"] = float(chart["selected_offset"])
        if name == "gap":
            chart_report[name]["gap_min"] = chart["gap_min"]
            chart_report[name]["gap_max"] = chart["gap_max"]
        if chart_report[name]["jacobian_nonpositive_count"] or chart_report[name]["nonpositive_cell_count"]:
            raise RuntimeError(f"Chart {name} failed the positive-orientation quality gate")
    def jsonable(value):
        if hasattr(value, "tolist"):
            return value.tolist()
        if isinstance(value, dict):
            return {key: jsonable(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [jsonable(item) for item in value]
        return value

    anchor_report = jsonable(anchor or {})
    report = {
        "source": str(source),
        "method": "Topology-Aware Overlapping Curvilinear Atlas (TAOCA)",
        "source_nodes": int(len(data["x"])),
        "source_triangles": int(len(data["triangles"])),
        "charts": chart_report,
        "roundtrip": roundtrip_report,
        "transition_quality": transition_quality(transitions),
        "native_overset_connectivity": False,
        "sequence_anchor": anchor_report,
        "connectivity_note": "The source CGNS contains no ZoneGridConnectivity_t. Chart transitions are geometric reconstructions, not recovered Fluent donor/receptor pairs.",
    }
    save_charts(charts, output_dir)
    save_transitions(transitions, output_dir)
    np.savez_compressed(output_dir / "inverse_on_original_topology.npz", Mach=reconstructed.astype(np.float32))
    (output_dir / "validation_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    plot_validation(data, charts, reconstructed, report, output_dir)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, default=THIS_DIR / "cgns_hdf5_cache")
    parser.add_argument(
        "--anchor-json",
        type=Path,
        default=None,
        help="Optional sequence anchor JSON produced by run_coherent_taoca_audit.py",
    )
    args = parser.parse_args()
    anchor = json.loads(args.anchor_json.read_text(encoding="utf-8")) if args.anchor_json else None
    report = process_snapshot(args.input, args.output_dir, args.cache_dir, anchor=anchor)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
