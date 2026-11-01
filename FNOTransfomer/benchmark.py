from pathlib import Path
import argparse
import copy
import importlib.util
import json
import sys
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
from scipy.spatial import cKDTree
import shapely
from shapely.geometry import Polygon

ROOT = Path(r'E:\1-AllCode\mesh transformation')
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import cgns_tsto_io as io
sys.path.insert(0, str(HERE))
import overset_atlas_transform as atlas

WANG = Path(r'H:\研2\4-我的研究\5-Fluent计算-二维TSTO\Wang2022_M7_Beta8_Static2D')


def load_wang():
    spec = importlib.util.spec_from_file_location('wang_mesh', WANG / 'validate_tsto_mesh.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    points, triangles, faces, zones = module.parse_fluent_ascii(WANG / 'TSTO_Wang2022_M7_Beta8_Static2D.msh')
    edges = {name: [] for name in ('smj', 'xmj', 'in', 'out')}
    for first, second, _, _, zone in faces:
        name = zones[zone]['name']
        if name in edges:
            edges[name].append((first - 1, second - 1))
    points = points[1:]
    data = {'x': points[:, 0], 'y': points[:, 1], 'triangles': triangles - 1,
            'wall_loops': {name: io.ordered_loop_from_edges(edges[name]) for name in ('smj', 'xmj')},
            'outer_loop': io.ordered_loop_from_edges(edges['in'] + edges['out'])}
    length = np.ptp(points[data['wall_loops']['xmj'], 0])
    data['fields'] = synthetic(points, length)
    return data


def synthetic(points, length):
    scaled = points / length
    return {'Affine': 1 + 0.7 * scaled[:, 0] - 0.3 * scaled[:, 1],
            'Smooth': np.sin(2 * scaled[:, 0]) * np.cos(3 * scaled[:, 1]),
            'ShockProxy': np.tanh(80 * (scaled[:, 0] - 0.7 + 0.2 * scaled[:, 1]))}


def domain(data):
    points = np.column_stack([data['x'], data['y']])
    bodies = shapely.union_all([Polygon(points[loop]) for loop in data['wall_loops'].values()])
    return Polygon(points[data['outer_loop']]).difference(bodies), bodies


def charts_for(data, anchor, scale, factor, alpha):
    outer = atlas.outer_annulus_chart(data, int(512 * factor), int(192 * factor), 0.8 * scale, anchor, alpha)
    return {'smj': atlas.positive_body_chart(data, data['wall_loops']['smj'], int(384 * factor), int(144 * factor), 3 * scale,
                                          (anchor or {}).get('body_reference', {}).get('smj')),
            'xmj': atlas.positive_body_chart(data, data['wall_loops']['xmj'], int(384 * factor), int(144 * factor), 3 * scale,
                                          (anchor or {}).get('body_reference', {}).get('xmj')),
            'outer': outer, 'gap': atlas.gap_h_chart(data, outer, int(512 * factor), int(192 * factor), anchor)}


def sample_source(data, charts):
    if '_source_triangulation' not in data:
        data['_source_triangulation'] = mtri.Triangulation(data['x'], data['y'], data['triangles'])
    source = data['_source_triangulation']
    finder = source.get_trifinder()
    _, bodies = domain(data)
    for chart in charts.values():
        chart.pop('inverse_cache', None)
        chart.pop('inverse_cache_query', None)
        chart['source_valid'] = finder(chart['x'], chart['y']) >= 0
        chart['fields'] = {name: np.asarray(mtri.LinearTriInterpolator(source, values)(chart['x'], chart['y']).filled(np.nan))
                           for name, values in data['fields'].items()}
        triangles = atlas.structured_triangles(*chart['x'].shape, chart['periodic'])
        points = np.column_stack([chart['x'].ravel(), chart['y'].ravel()])
        polygons = shapely.polygons(points[triangles])
        collision = shapely.area(shapely.intersection(polygons, bodies)) > 1e-14 * max(bodies.area, 1e-12)
        chart['valid_donor_cells'] = np.all(chart['source_valid'].ravel()[triangles], axis=1) & ~collision


def geometry(data, charts):
    fluid, bodies = domain(data)
    report = {}
    footprints = []
    accepted_footprints = []
    for name, chart in charts.items():
        points = np.column_stack([chart['x'].ravel(), chart['y'].ravel()])
        triangles = atlas.structured_triangles(*chart['x'].shape, chart['periodic'])
        polygons = shapely.polygons(points[triangles])
        areas = shapely.area(polygons)
        rows = np.stack([chart['x'], chart['y']], axis=-1)
        strips = []
        for row in range(len(rows) - 1):
            if chart['periodic']:
                first = Polygon(rows[row])
                second = Polygon(rows[row + 1])
                strips.append(second.symmetric_difference(first))
            else:
                strips.append(Polygon(np.vstack([rows[row], rows[row + 1][::-1]])))
        if not all(part.is_valid for part in strips):
            raise ValueError(f'{name}: invalid strip boundary; self-intersection detected')
        merged = shapely.union_all(strips)
        footprints.append(merged)
        signed = atlas.cross_2d(points[triangles[:, 1]] - points[triangles[:, 0]], points[triangles[:, 2]] - points[triangles[:, 0]]) / 2
        valid_triangles = chart['valid_donor_cells']
        overlap_area = max(0, areas.sum() - merged.area)
        accepted = shapely.coverage_union_all(polygons[valid_triangles]) if overlap_area < 1e-9 * max(merged.area, 1) else shapely.union_all(polygons[valid_triangles])
        accepted_footprints.append(accepted)
        report[name] = {'minimum_J': float(chart['J'].min()), 'nonpositive_cells': int(np.sum(signed <= 0)),
                        'self_overlap_area': float(max(0, areas.sum() - merged.area)),
                        'solid_penetration_area': float(merged.intersection(bodies).area),
                        'accepted_solid_penetration_area': float(accepted.intersection(bodies).area),
                        'source_invalid_node_fraction': float(np.mean(~chart['source_valid'])),
                        'area': merged.area, 'selected_offset': chart.get('selected_offset'),
                        'continuous_boundary_radial_derivative_min': chart.get('continuous_boundary_radial_derivative_min')}
    covered = shapely.union_all(footprints).intersection(fluid)
    report['fluid_area'] = fluid.area
    report['geometric_area_coverage'] = covered.area / fluid.area
    report['valid_donor_area_coverage'] = shapely.union_all(accepted_footprints).intersection(fluid).area / fluid.area
    return report


def reconstruct(data, charts, field, mode='blend'):
    candidates, weights = [], []
    for name, chart in charts.items():
        values, eta, xi, _, _ = atlas.inverse_chart_values(chart, data['x'], data['y'], field)
        weight = atlas.chart_weight(name, eta, xi)
        weight[~np.isfinite(values)] = 0
        candidates.append(values)
        weights.append(weight)
    values = np.array(candidates)
    weights = np.array(weights)
    covered = weights.sum(axis=0) > 0
    result = np.full(len(data['x']), np.nan)
    if mode == 'single':
        selected = weights.argmax(axis=0)
        result[covered] = values[selected[covered], np.flatnonzero(covered)]
    else:
        result[covered] = (np.nan_to_num(values) * weights).sum(axis=0)[covered] / weights.sum(axis=0)[covered]
    return result, np.sum(weights > 0, axis=0)


def metrics(source, result, mask):
    if not np.any(mask):
        return {'count': 0}
    error = result[mask] - source[mask]
    return {'count': int(mask.sum()), 'rmse': float(np.sqrt(np.mean(error ** 2))),
            'nrmse_by_range': float(np.sqrt(np.mean(error ** 2)) / max(np.ptp(source), 1e-30)),
            'max_absolute': float(np.max(np.abs(error)))}


def benchmark(data, charts, destination):
    if not all('valid_donor_cells' in chart for chart in charts.values()):
        sample_source(data, charts)
    count = sum(chart['x'].size for chart in charts.values())
    ratio = np.ptp(data['x']) / np.ptp(data['y'])
    nx = int(np.sqrt(count * ratio))
    ny = int(round(count / nx))
    xgrid, ygrid = np.meshgrid(np.linspace(data['x'].min(), data['x'].max(), nx), np.linspace(data['y'].min(), data['y'].max(), ny))
    cart = {'cartesian': {'x': xgrid, 'y': ygrid, 'periodic': False, 'cartesian': True}}
    cart['cartesian'].update(atlas.logical_metrics(xgrid, ygrid, False))
    sample_source(data, cart)
    meshfree = {'cartesian': dict(cart['cartesian'])}
    query = np.column_stack([xgrid.ravel(), ygrid.ravel()])
    distances, neighbors = cKDTree(np.column_stack([data['x'], data['y']])).query(query, k=8)
    idw_weights = 1 / np.maximum(distances, 1e-14) ** 2
    idw_weights /= idw_weights.sum(axis=1, keepdims=True)
    meshfree['cartesian']['fields'] = {}
    for name, values in data['fields'].items():
        sampled = np.sum(values[neighbors] * idw_weights, axis=1).reshape(xgrid.shape)
        sampled[~cart['cartesian']['source_valid']] = np.nan
        meshfree['cartesian']['fields'][name] = sampled
    methods = {'TAOCA': (charts, 'blend'), 'single_donor': (charts, 'single'),
               'three_charts': ({name: chart for name, chart in charts.items() if name != 'gap'}, 'blend'),
               'two_O': ({name: charts[name] for name in ('smj', 'xmj')}, 'blend'),
               'Cartesian': (cart, 'blend'), 'Cartesian_IDW8': (meshfree, 'blend')}
    results = {}
    predictions = {}
    for method, (selected, mode) in methods.items():
        started = time.perf_counter()
        predicted = {name: reconstruct(data, selected, name, mode) for name in data['fields']}
        results[method] = {'inverse_seconds': time.perf_counter() - started,
                           'sample_points': sum(chart['x'].size for chart in selected.values()),
                           'array_bytes': sum(chart['x'].nbytes + chart['y'].nbytes + sum(values.nbytes for values in chart['fields'].values()) for chart in selected.values())}
        predictions[method] = predicted
    for field, source in data['fields'].items():
        common = np.logical_and.reduce([np.isfinite(predictions[name][field][0]) for name in ('TAOCA', 'Cartesian', 'single_donor', 'Cartesian_IDW8')])
        for method, predicted in predictions.items():
            result, membership = predicted[field]
            valid = np.isfinite(result)
            results[method][field] = {'coverage': float(valid.mean()), 'own_valid': metrics(source, result, valid),
                                     'common_TAOCA_Cartesian_single': metrics(source, result, common & valid),
                                     'overlap': metrics(source, result, valid & (membership >= 2))}
    field = 'Mach' if 'Mach' in data['fields'] else 'ShockProxy'
    result = predictions['TAOCA'][field][0]
    plt.figure(figsize=(10, 4))
    plt.scatter(data['x'], data['y'], c=np.abs(result - data['fields'][field]), s=0.5, cmap='inferno')
    missing = ~np.isfinite(result)
    plt.scatter(data['x'][missing], data['y'][missing], s=2, c='cyan', label='Uncovered/invalid')
    plt.axis('equal')
    plt.colorbar(label=f'{field} absolute roundtrip error')
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(destination / 'roundtrip_error.png', dpi=180)
    plt.close()
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--case', choices=['legacy', 'wang'], required=True)
    parser.add_argument('--factor', type=float, default=1)
    parser.add_argument('--output-root', type=Path)
    parser.add_argument('--times', nargs='+', default=[f'0.{number:02d}0000' for number in range(1, 11)])
    args = parser.parse_args()
    output = args.output_root or HERE / 'results' / args.case
    output.mkdir(parents=True, exist_ok=True)
    if args.case == 'wang':
        data_sequence = [load_wang()]
        scale = np.ptp(data_sequence[0]['x'][data_sequence[0]['wall_loops']['xmj']]) / 100
        labels = ['static']
    else:
        labels = args.times
        data_sequence = [io.parse_cgns(ROOT / 'cgns_hdf5_cache' / f'0718_3-C0-{label}.5-1to2_7-Ma5.hdf5.cgns') for label in labels]
        scale = 1
    anchor = atlas.sequence_anchor(data_sequence, int(512 * args.factor), 0.8 * scale)
    anchor['body_reference'] = {name: atlas._resample_closed_at_phase(atlas._oriented_boundary(data_sequence[0], loop), int(384 * args.factor),
                                  atlas._geometric_phase_target(atlas._oriented_boundary(data_sequence[0], loop))) for name, loop in data_sequence[0]['wall_loops'].items()}
    reports = {}
    for label, data in zip(labels, data_sequence):
        destination = output / label
        destination.mkdir(exist_ok=True)
        existing = destination / 'report.json'
        if existing.exists():
            saved = json.loads(existing.read_text())
            expected = 2 * int(384 * args.factor) * int(144 * args.factor) + 2 * int(512 * args.factor) * int(192 * args.factor)
            if saved.get('benchmark', {}).get('TAOCA', {}).get('sample_points') == expected and 'Cartesian_IDW8' in saved.get('benchmark', {}):
                reports[label] = saved
                payload = np.load(destination / 'overset_atlas_charts.npz')
                for name in ('smj', 'xmj'):
                    anchor['body_reference'][name] = np.column_stack([payload[name+'_x'][0],payload[name+'_y'][0]])
                print(label, 'resumed completed frame', flush=True)
                continue
        started = time.perf_counter()
        try:
            charts = charts_for(data, anchor, scale, args.factor, 0.05)
            elapsed = time.perf_counter() - started
            sample_source(data, charts)
            report = {'chart_build_seconds': elapsed, 'geometry': geometry(data, charts),
                      'benchmark': benchmark(data, charts, destination), 'scale': scale,
                      'scope': 'real flow fields' if args.case == 'legacy' else 'analytic fields on real mesh; no converged CFD claim'}
            (destination / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
            atlas.save_charts(charts, destination)
            if label == 'static' or label == '0.010000':
                old = charts_for(data, anchor, scale, args.factor, 0)
                sample_source(data, old)
                report['old_cosine_geometry'] = geometry(data, old)
                old_destination = destination / 'old_cosine'
                old_destination.mkdir(exist_ok=True)
                report['old_cosine_benchmark'] = benchmark(data, old, old_destination)
            for name in ('smj', 'xmj'):
                anchor['body_reference'][name] = np.column_stack([charts[name]['x'][0], charts[name]['y'][0]])
        except (RuntimeError, ValueError) as error:
            report = {'failure': str(error), 'scale': scale, 'scope': 'failed template; no fallback hiding failure'}
        reports[label] = report
        data.pop('_source_triangulation', None)
        (destination / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
        (output / 'summary.json').write_text(json.dumps(reports, indent=2), encoding='utf-8')
        print(label, json.dumps({key: value for key, value in report.items() if key in ('failure', 'chart_build_seconds')}), flush=True)
    (output / 'summary.json').write_text(json.dumps(reports, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
