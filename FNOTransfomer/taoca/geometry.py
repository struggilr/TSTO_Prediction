"""Portable geometry kernels from the audited TAOCA implementation."""
from __future__ import annotations
import numpy as np
import matplotlib.tri as mtri
from scipy.ndimage import gaussian_filter1d
from scipy.spatial import ConvexHull
def _signed_area(points: np.ndarray) -> float:
    return float(0.5 * np.sum(points[:, 0] * np.roll(points[:, 1], -1) - points[:, 1] * np.roll(points[:, 0], -1)))

def _resample_closed(points: np.ndarray, count: int) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    if points.shape[0] < 3:
        raise ValueError("A closed boundary needs at least three points")
    if np.linalg.norm(points[0] - points[-1]) < 1e-12:
        points = points[:-1]
    extended = np.vstack([points, points[0]])
    segment = np.linalg.norm(np.diff(extended, axis=0), axis=1)
    cumulative = np.concatenate([[0.0], np.cumsum(segment)])
    samples = np.linspace(0.0, cumulative[-1], count, endpoint=False)
    result = np.empty((count, 2), dtype=np.float64)
    for axis in range(2):
        result[:, axis] = np.interp(samples, cumulative, extended[:, axis])
    return result

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

