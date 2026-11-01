"""Measure TAOCA round-trip error and explicit residual-lifting trade-offs.

This is an offline transform diagnostic, not a forecasting result. Residuals are
computed from the evaluated frame itself, so they cannot be supplied as future
model inputs. In a predictor, the residual stream must itself be predicted.
"""
import json
from pathlib import Path

import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np

from analyze_snapshot30_shock_error import chart_donors, fuse, high_gradient_nodes

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'outputs/TAOCA_boundary_v2_64x384/fenli_paired/Ma5.0/0029.h5'
OUT = ROOT / 'outputs/TAOCA_boundary_v2_64x384/snapshot30_Ma5.0/lossless_residual_lifting'
CHANNELS = ['Pressure', 'Density', 'Temperature', 'Velocity X', 'Velocity Y']


def nrmse(truth, pred, use, std):
    e = pred - truth
    return [float(np.sqrt(np.mean(e[k, use] ** 2)) / std[k]) for k in range(5)]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    with h5py.File(SOURCE, 'r') as f:
        truth = f['source/fields'][:].astype(np.float64)
        points = np.column_stack([f['source/x'][:], f['source/y'][:]])
        tri = f['source/triangles'][:].astype(np.int32)
        donors = chart_donors(f, 64)
        coarse, valid = fuse(donors, 'geometric_over_h2')
        time = float(f.attrs['time'])
        wall_curves = []
        for name in ('smj', 'xmj'):
            xy = f[f'taoca/{name}/xy'][:].astype(np.float64)
            wall_curves.append(np.column_stack([xy[0, 0], xy[1, 0]]))

    std = np.std(truth, axis=1)
    shock, threshold, shock_cell_fraction = high_gradient_nodes(points, tri, truth[0], 99.5)
    residual = truth - coarse
    score = np.max(np.abs(residual) / std[:, None], axis=0)
    score[~valid] = -np.inf
    candidates = [(0.0, 'atlas_only'), (0.01, 'retain_1pct'), (0.025, 'retain_2p5pct'),
                  (0.05, 'retain_5pct'), (0.10, 'retain_10pct'), (0.20, 'retain_20pct'),
                  (0.50, 'retain_50pct'), (1.0, 'retain_100pct')]
    order = np.argsort(score)[::-1]
    n_valid = int(valid.sum())
    keep20 = np.zeros(len(valid), dtype=bool)
    keep20[order[:int(np.ceil(0.20 * n_valid))]] = True
    rows = []
    for fraction, label in candidates:
        retain = np.zeros(len(valid), dtype=bool)
        keep_n = int(np.ceil(fraction * n_valid))
        if keep_n:
            retain[order[:keep_n]] = True
        pred = coarse.copy()
        pred[:, retain] += residual[:, retain]
        pred[:, ~valid] = np.nan
        global_nrmse = nrmse(truth, pred, valid, std)
        shock_use = valid & shock
        shock_nrmse = nrmse(truth, pred, shock_use, std)
        # A compact sparse stream stores all five float32 residual channels at
        # each retained source node, plus one uint8 node mask.
        residual_bytes = keep_n * 5 * 4 + len(valid)
        atlas_field_bytes = 4 * 64 * 384 * 5 * 4
        rows.append({
            'scheme': label, 'retained_source_node_fraction': float(retain.sum() / n_valid),
            'retained_source_nodes': int(retain.sum()),
            'global_nrmse_by_channel': global_nrmse,
            'shock_core_nrmse_by_channel': shock_nrmse,
            'relative_residual_storage_vs_four_chart_float32_fields': residual_bytes / atlas_field_bytes,
            'max_abs_error_by_channel': [float(np.nanmax(np.abs(pred[k, valid] - truth[k, valid]))) for k in range(5)],
        })
    exact = truth.copy()
    exact[:, valid] = coarse[:, valid] + residual[:, valid]
    max_roundtrip = float(np.max(np.abs(exact[:, valid] - truth[:, valid])))
    max_roundtrip_rel = [float(np.max(np.abs(exact[k, valid] - truth[k, valid])) / std[k]) for k in range(5)]
    residual_f32 = residual.astype(np.float32).astype(np.float64)
    practical = coarse.copy()
    practical[:, valid] += residual_f32[:, valid]
    f32_error = practical[:, valid] - truth[:, valid]
    pred20 = coarse.copy(); pred20[:, keep20] += residual[:, keep20]
    pred100 = coarse.copy(); pred100[:, valid] += residual[:, valid]
    result = {
        'source_file': str(SOURCE), 'snapshot_index_1based': 30, 'time': time,
        'method': 'two-stream lifting: chart coefficients c=S(u), residual r=u-R(c), inverse R(c)+r',
        'status': 'offline same-frame reconstruction diagnostic; residual stream is target information and must be predicted in forecasting',
        'source_valid_coverage': float(valid.mean()),
        'shock_core_definition': 'vertices of the top 0.5% source P1 pressure-gradient triangles; diagnostic proxy only',
        'shock_core_cell_fraction': shock_cell_fraction,
        'shock_core_node_fraction': float(shock.mean()),
        'shock_core_gradient_threshold': threshold,
        'training_std': std.tolist(),
        'exact_full_residual_roundtrip': {'max_absolute_error': max_roundtrip,
                                           'max_normalized_error_by_channel': max_roundtrip_rel,
                                           'nrmse_by_channel': nrmse(truth, exact, valid, std)},
        'float32_residual_roundtrip': {
            'max_absolute_error_by_channel': [float(np.max(np.abs(f32_error[k]))) for k in range(5)],
            'nrmse_by_channel': [float(np.sqrt(np.mean(f32_error[k] ** 2)) / std[k]) for k in range(5)],
        },
        'retention_sweep': rows,
        'interpretation': 'Exact reconstruction is achieved only because the residual carries the discarded information. It does not imply conservation, lower forecast error, or a zero-error neural prediction.',
    }
    (OUT / 'snapshot30_residual_lifting_metrics.json').write_text(json.dumps(result, indent=2), encoding='utf-8')

    colors = ['#277DA1', '#43AA8B', '#F9C74F', '#F8961E', '#F94144']
    x = np.array([r['retained_source_node_fraction'] * 100 for r in rows])
    fig, ax = plt.subplots(figsize=(7.6, 4.6), constrained_layout=True)
    for k, name in enumerate(CHANNELS):
        ax.plot(x, [r['shock_core_nrmse_by_channel'][k] for r in rows], marker='o', lw=1.8,
                ms=4, color=colors[k], label=name)
    ax.set_xlabel('Source nodes carrying explicit residual (%)')
    ax.set_ylabel('Shock-core NRMSE')
    ax.set_title('TAOCA round-trip error versus residual retention — snapshot 30')
    ax.grid(alpha=.22); ax.legend(frameon=False, ncol=2)
    fig.savefig(OUT / 'snapshot30_residual_retention_shock_nrmse.png', dpi=300, bbox_inches='tight')
    plt.close(fig)

    # Spatial cloud maps: source CFD, plain TAOCA round-trip, sparse 20% residual,
    # and the full residual identity reconstruction. Error colors are normalized
    # by each channel's training-frame standard deviation and share a row scale.
    tri_plot = mtri.Triangulation(points[:, 0], points[:, 1], tri)
    tri_plot.set_mask(~valid[tri].all(axis=1))
    tri_truth = mtri.Triangulation(points[:, 0], points[:, 1], tri)
    fig, axs = plt.subplots(5, 5, figsize=(20, 15), constrained_layout=True)
    for k, name in enumerate(CHANNELS):
        ref = truth[k]
        lo, hi = np.nanpercentile(ref[valid], [1, 99])
        # Preserve a non-degenerate scale for nearly uniform channels.
        if hi <= lo: hi = lo + 1.0
        v0 = axs[k, 0].tripcolor(tri_truth, ref, shading='gouraud', cmap='turbo', vmin=lo, vmax=hi)
        axs[k, 0].set_title(f'{name}: source CFD')
        v1 = axs[k, 1].tripcolor(tri_plot, coarse[k], shading='gouraud', cmap='turbo', vmin=lo, vmax=hi)
        axs[k, 1].set_title(f'{name}: TAOCA round-trip')
        err_fields = [(coarse[k] - ref) / std[k], (pred20[k] - ref) / std[k],
                      (pred100[k] - ref) / std[k]]
        lim = max(float(np.percentile(np.abs(err_fields[0][valid]), 99.5)), 1e-8)
        for col, (err, title) in enumerate(zip(err_fields, ['TAOCA error / σ', '20% residual error / σ', '100% residual error / σ']), start=2):
            vm = np.where(valid, err, np.nan)
            obj = axs[k, col].tripcolor(tri_plot, vm, shading='gouraud', cmap='RdBu_r', vmin=-lim, vmax=lim)
            if k == 0: axs[k, col].set_title(title)
        for ax in axs[k]:
            for wall in wall_curves:
                ax.plot(np.r_[wall[:, 0], wall[0, 0]], np.r_[wall[:, 1], wall[0, 1]], color='k', lw=.32, alpha=.7)
            ax.set_aspect('equal', adjustable='box')
            ax.set_xlim(-60, 160); ax.set_ylim(-75, 75)
            ax.tick_params(labelsize=6); ax.grid(False)
        fig.colorbar(v0, ax=[axs[k, 0], axs[k, 1]], shrink=.72, pad=.02, label='Physical value')
        fig.colorbar(obj, ax=[axs[k, 2], axs[k, 3], axs[k, 4]], shrink=.72, pad=.02, label='Error / training σ')
    fig.suptitle('Ma=5.0 TSTO snapshot 30: TAOCA round-trip and residual correction', fontsize=15)
    fig.savefig(OUT / 'snapshot30_five_channel_roundtrip_error_cloudmaps.png', dpi=250, bbox_inches='tight')
    plt.close(fig)

    lines = [
        '# Ma=5.0 第 30 帧：TAOCA 往返误差的残差补偿试验', '',
        '**性质：离线同帧重建实验。残差由被评价的本帧真值计算，不能作为未来预测输入；要用于预测，模型必须预测残差。**', '',
        f'- 时间：t={time:.3f}；TAOCA 为 4 张 64×384 图；源节点有效覆盖 {valid.mean():.3%}。',
        '- 残差定义：先将源场正向映射并回映射得到粗重建，再计算 `r = u_source - u_coarse`；逆变换为 `u_recon = u_coarse + r`。',
        f"- 保留全部有效节点残差时最大绝对往返误差：{max_roundtrip:.3e}；按训练标准差归一化的最大误差：{max(max_roundtrip_rel):.3e}。这是数值恒等式验证，不是无损 TAOCA 几何映射本身。", '',
        f"- 残差以 float32 存储时，五通道 NRMSE：{' / '.join(f'{v:.3e}' for v in result['float32_residual_roundtrip']['nrmse_by_channel'])}。", '',
        '|残差保留源节点比例|全域五通道 NRMSE|激波核心五通道 NRMSE|残差存储 / 四图 float32 场|',
        '|---:|---|---|---:|',
    ]
    for r in rows:
        lines.append(f"|{r['retained_source_node_fraction']:.1%}|{' / '.join(f'{v:.4g}' for v in r['global_nrmse_by_channel'])}|{' / '.join(f'{v:.4g}' for v in r['shock_core_nrmse_by_channel'])}|{r['relative_residual_storage_vs_four_chart_float32_fields']:.2f}×|")
    lines += ['', '![五通道场值与往返误差云图](snapshot30_five_channel_roundtrip_error_cloudmaps.png)', '',
              '![残差保留率与激波核心往返误差](snapshot30_residual_retention_shock_nrmse.png)', '',
              '## 结论与边界',
              '双流表示把映射误差显式放进残差通道，因此理论上可通过完整残差精确还原输入快照。零误差是由保留全部被丢弃信息保证的，残差流的体量可能很大；稀疏残差则仍会留下误差。它不等于守恒重映射，不保证残差更容易预测，也不表示未来帧可使用真值残差。',
              '后续若进入时序模型，应把残差当作待预测状态：只允许历史残差进入输入，未来残差作为监督目标；另与 Cartesian 直接预测、纯 TAOCA 预测进行同架构同预算比较。', '']
    (OUT / 'snapshot30_residual_lifting_analysis.md').write_text('\n'.join(lines), encoding='utf-8')
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
