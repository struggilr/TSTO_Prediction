"""Single-frame diagnostic: append a local shock-focused Cartesian atlas patch.

The patch box is selected using snapshot 30's reference pressure-gradient band.
This is an oracle mapping experiment only; it must not be presented as a causal
time-series method or used as a training input until patch placement is inferred
from past frames/model predictions.
"""
import json
from pathlib import Path

import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
from shapely import Polygon, polygons, intersection, area, union_all

from analyze_snapshot30_shock_error import chart_donors, fuse, high_gradient_nodes, metrics

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'outputs/TAOCA_boundary_v2_64x384/fenli_paired/Ma5.0/0029.h5'
OUT = ROOT / 'outputs/TAOCA_boundary_v2_64x384/snapshot30_Ma5.0/shock_patch_prototype'
CHANNELS = ['Pressure', 'Density', 'Temperature', 'Velocity X', 'Velocity Y']


def barycentric(points, triangles, query, finder):
    ti = finder(query[:, 0], query[:, 1])
    valid = ti >= 0
    ids = triangles[np.maximum(ti, 0)]
    p = points[ids]
    den = ((p[:, 1, 0] - p[:, 0, 0]) * (p[:, 2, 1] - p[:, 0, 1])
           - (p[:, 1, 1] - p[:, 0, 1]) * (p[:, 2, 0] - p[:, 0, 0]))
    b = ((query[:, 0] - p[:, 0, 0]) * (p[:, 2, 1] - p[:, 0, 1])
         - (query[:, 1] - p[:, 0, 1]) * (p[:, 2, 0] - p[:, 0, 0])) / den
    c = ((p[:, 1, 0] - p[:, 0, 0]) * (query[:, 1] - p[:, 0, 1])
         - (p[:, 1, 1] - p[:, 0, 1]) * (query[:, 0] - p[:, 0, 0])) / den
    w = np.column_stack([1.0 - b - c, b, c])
    w[~valid] = 0.0
    return ids, w, valid


def make_patch(points, src_tri, fields, wall_polygons, bounds=(-5.0, 75.0, -8.0, 30.0), nx=256, ny=192):
    # Reference-mesh interpolation onto a locally refined Cartesian patch.
    source_tri = mtri.Triangulation(points[:, 0], points[:, 1], src_tri)
    source_finder = source_tri.get_trifinder()
    xmin, xmax, ymin, ymax = bounds
    xx, yy = np.meshgrid(np.linspace(xmin, xmax, nx), np.linspace(ymin, ymax, ny))
    q = np.column_stack([xx.ravel(), yy.ravel()])
    ids, bary, valid_nodes = barycentric(points, src_tri, q, source_finder)
    values = np.einsum('nk,nkc->nc', bary, fields.T[ids])
    values[~valid_nodes] = 0.0

    idx = np.arange(nx * ny).reshape(ny, nx)
    tris = np.concatenate([
        np.column_stack([idx[:-1, :-1].ravel(), idx[:-1, 1:].ravel(), idx[1:, 1:].ravel()]),
        np.column_stack([idx[:-1, :-1].ravel(), idx[1:, 1:].ravel(), idx[1:, :-1].ravel()]),
    ]).astype(np.int32)
    coords = q[tris]
    cell_poly = polygons(coords)
    solids = union_all(wall_polygons)
    collides = area(intersection(cell_poly, solids)) > 1e-12
    good = valid_nodes[tris].all(axis=1) & ~collides
    inv_tri = tris[good]
    patch_tri = mtri.Triangulation(xx.ravel(), yy.ravel(), inv_tri)
    inv_ids, inv_bary, inv_valid = barycentric(q, inv_tri, points, patch_tri.get_trifinder())
    mapped = np.einsum('nk,nkc->nc', inv_bary, values[inv_ids])
    mapped[~inv_valid] = np.nan

    # Smoothly fade the patch into the original atlas over a 5-unit edge band.
    edge_dist = np.minimum.reduce([points[:, 0] - xmin, xmax - points[:, 0],
                                   points[:, 1] - ymin, ymax - points[:, 1]])
    z = np.clip(edge_dist / 5.0, 0.0, 1.0)
    patch_weight = z * z * (3.0 - 2.0 * z)
    patch_weight[~inv_valid] = 0.0
    return mapped.T, inv_valid, patch_weight, {
        'bounds': list(bounds), 'resolution': [ny, nx], 'nodes': int(nx * ny),
        'valid_sample_fraction': float(valid_nodes.mean()),
        'valid_inverse_coverage': float(inv_valid.mean()),
        'valid_cells': int(good.sum()), 'total_cells': int(len(tris)),
        'solid_intersected_cells': int(collides.sum()),
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    with h5py.File(SOURCE, 'r') as f:
        x, y = f['source/x'][:], f['source/y'][:]
        points = np.column_stack([x, y])
        tri = f['source/triangles'][:].astype(np.int32)
        fields = f['source/fields'][:].astype(np.float64)
        time = float(f.attrs['time'])
        donors = chart_donors(f, 64)
        base, base_valid = fuse(donors, 'geometric_over_h2')
        # Reconstruct each wall polygon from its exact O-chart wall trace.
        wall_polys = []
        for name in ('smj', 'xmj'):
            chart_xy = f[f'taoca/{name}/xy'][:].astype(np.float64)
            wall = np.column_stack([chart_xy[0, 0], chart_xy[1, 0]])
            wall_polys.append(Polygon(wall))

    patch, patch_valid, patch_weight, patch_meta = make_patch(
        points, tri, fields, wall_polys)
    combined = base.copy()
    use_patch = patch_valid & base_valid
    combined[:, use_patch] = (
        patch[:, use_patch] * patch_weight[use_patch]
        + base[:, use_patch] * (1.0 - patch_weight[use_patch]))
    # Where the original atlas has no donor but the patch does, use the patch.
    only_patch = patch_valid & ~base_valid
    combined[:, only_patch] = patch[:, only_patch]
    combined_valid = base_valid | patch_valid

    # Use the strongest 0.5% cells as a shock-core proxy. A top-5% mask mixes
    # shock fronts with broad compression regions and dilutes the diagnosis.
    shock, threshold, shock_cell_fraction = high_gradient_nodes(points, tri, fields[0], 99.5)
    # Whole-domain comparison uses every baseline-valid node. The combined
    # method falls back exactly to the baseline outside the local patch.
    common = base_valid.copy()
    patch_zone = base_valid & patch_valid
    std = np.std(fields, axis=1)
    baseline_metrics = metrics(fields, base, common, std, shock)
    patch_metrics = metrics(fields, combined, common, std, shock)
    zone_base_metrics = metrics(fields, base, patch_zone, std, shock)
    zone_patch_metrics = metrics(fields, combined, patch_zone, std, shock)
    rows = []
    for b, p, zb, zp in zip(baseline_metrics, patch_metrics, zone_base_metrics, zone_patch_metrics):
        rows.append({
            'field': b['field'], 'baseline_full_support_nrmse': b['nrmse'],
            'patch_full_support_nrmse': p['nrmse'],
            'full_support_error_reduction_percent': 100.0 * (1.0 - p['nrmse'] / b['nrmse']),
            'baseline_high_gradient_nrmse': b['shock_nrmse'],
            'patch_high_gradient_nrmse': p['shock_nrmse'],
            'high_gradient_error_reduction_percent': 100.0 * (1.0 - p['shock_nrmse'] / b['shock_nrmse']),
            'baseline_nonshock_nrmse': b['nonshock_nrmse'],
            'patch_nonshock_nrmse': p['nonshock_nrmse'],
            'patch_zone_baseline_nrmse': zb['nrmse'],
            'patch_zone_method_nrmse': zp['nrmse'],
            'patch_zone_error_reduction_percent': 100.0 * (1.0 - zp['nrmse'] / zb['nrmse']),
            'patch_zone_high_gradient_baseline_nrmse': zb['shock_nrmse'],
            'patch_zone_high_gradient_method_nrmse': zp['shock_nrmse'],
            'patch_zone_high_gradient_error_reduction_percent': 100.0 * (1.0 - zp['shock_nrmse'] / zb['shock_nrmse']),
        })
    result = {
        'source_file': str(SOURCE), 'snapshot_index_1based': 30, 'time': time,
        'status': 'single-frame oracle diagnostic; NOT causal and NOT a training result',
        'high_gradient_region': {
            'definition': 'vertices of top 0.5% source triangles by P1 pressure-gradient magnitude (shock-core proxy, not a shock tracker)',
            'triangle_fraction': shock_cell_fraction, 'threshold': threshold,
            'node_fraction': float(shock.mean()),
        },
        'patch': patch_meta,
        'coverage': {'baseline_atlas': float(base_valid.mean()),
                     'patch': float(patch_valid.mean()),
                     'combined': float(combined_valid.mean()),
                     'whole_domain_scored_support': float(common.mean()),
                     'patch_zone_scored_support': float(patch_zone.mean()),
                     'high_gradient_nodes_supported_by_patch': float((patch_valid & shock).sum() / shock.sum())},
        'metrics': rows,
        'interpretation_limit': 'Patch placement uses this frame reference field to define an oracle shock box. A deployable time-series atlas must infer the patch from past frames or predicted shock location; no future truth can set the chart.',
    }
    (OUT / 'snapshot30_shock_patch_metrics.json').write_text(json.dumps(result, indent=2), encoding='utf-8')

    plt.rcParams.update({'font.family': 'Times New Roman', 'font.size': 9,
                         'figure.facecolor': 'white', 'axes.facecolor': 'white'})
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
    xx = np.arange(5); width = .34
    for ax, key, title in [(axs[0], 'full_support_nrmse', 'Whole baseline-supported domain NRMSE'),
                            (axs[1], 'high_gradient_nrmse', 'High-gradient region NRMSE')]:
        b = [r[f'baseline_{key}'] for r in rows]
        p = [r[f'patch_{key}'] for r in rows]
        ax.bar(xx-width/2, b, width, label='64×384 TAOCA', color='#277DA1', alpha=.9)
        ax.bar(xx+width/2, p, width, label='TAOCA + local shock patch', color='#F94144', alpha=.85)
        ax.set_xticks(xx, ['p', 'ρ', 'T', 'u', 'v']); ax.set_ylabel('NRMSE'); ax.set_title(title)
        ax.grid(axis='y', alpha=.2); ax.legend(frameon=False, fontsize=8)
    fig.savefig(OUT / 'snapshot30_shock_patch_nrmse.png', dpi=300, bbox_inches='tight')
    plt.close(fig)

    md = ['# Ma=5.0 第 30 帧：TAOCA 激波局部加密图试验', '',
          '**结论范围：单帧、无神经网络的映射诊断。局部图位置使用本帧真实压力梯度定义，属于 oracle 上限试验，不能作为无泄漏时序训练结果。**', '',
          f'- 时间：t={time:.3f}',
          f"- 激波核心代理区：压力 P1 梯度最高 0.5% 源单元顶点（占节点 {shock.mean():.2%}）；不是独立验证的激波追踪器。",
          f"- 局部图：{patch_meta['resolution'][0]}×{patch_meta['resolution'][1]}，物理范围 {patch_meta['bounds']}，额外节点 {patch_meta['nodes']:,}。",
          f"- 基线 TAOCA 覆盖率 {base_valid.mean():.3%}；加入局部图后 {combined_valid.mean():.3%}；完整域比较在基线有效支持 {common.mean():.3%} 进行；局部 patch 支持 {patch_zone.mean():.3%}。",
          f"- patch 对高梯度诊断区节点的支撑率：{(patch_valid & shock).sum() / shock.sum():.2%}。", '',
          '|通道|基线完整支持 NRMSE|加入局部图 NRMSE|完整域误差降低|基线高梯度 NRMSE|加入局部图高梯度 NRMSE|高梯度误差降低|patch区误差降低|',
          '|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        md.append(f"|{r['field']}|{r['baseline_full_support_nrmse']:.5f}|{r['patch_full_support_nrmse']:.5f}|{r['full_support_error_reduction_percent']:+.1f}%|{r['baseline_high_gradient_nrmse']:.5f}|{r['patch_high_gradient_nrmse']:.5f}|{r['high_gradient_error_reduction_percent']:+.1f}%|{r['patch_zone_error_reduction_percent']:+.1f}%|")
    md += ['', '![TAOCA 与局部激波图映射误差对照](snapshot30_shock_patch_nrmse.png)', '',
           '## 方法与限制',
           '在四图 TAOCA 之外，局部规则图仅覆盖预先指定的激波包络；包络边缘以平滑权重淡出，与原 TAOCA 图册融合。完整支持指标在所有基线有效源节点上计算；patch区对照另限于patch有效源节点。流场均由原始 FE 三角网格 P1 插值到图上，再由图册 donor 插值回源节点。未训练网络、未改变守恒性质，也未证明跨时间稳定。',
           '若高梯度误差明显下降，下一步应使用上一帧压力梯度/模型预测的激波位置生成因果包络，并在完整连续帧上比较；若下降有限，问题更可能来自源网格 P1 表示、donor 质量或压力梯度区域定义，应先解决这些因素，而非继续堆图。', '']
    (OUT / 'snapshot30_shock_patch_analysis.md').write_text('\n'.join(md), encoding='utf-8')
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
