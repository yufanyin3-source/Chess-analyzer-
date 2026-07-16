"""
viz.distribution
================================================================================
ACPL 分布图：lognorm/gamma 拟合 + matched-N bootstrap 95% CI（向量化实现）。
从原 extra_style.py 迁移（逻辑与输出 100% 保持不变）。
"""

import logging

logger = logging.getLogger(__name__)

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from scipy.stats import lognorm, gamma, gaussian_kde

from chess_analyzer.stats.bootstrap import (matched_n_single_tier, ACPL_METRICS,
    ACPL_LABELS, MIN_BASELINE_N, ACPL_N_OUTER, ACPL_K_INNER, ACPL_K_POINT)

mpl.rcParams['font.sans-serif'] = ["Noto Sans CJK SC", "Noto Sans CJK JP", "SimHei",
                                    "Microsoft YaHei", "PingFang SC", "WenQuanYi Zen Hei", "DejaVu Sans"]
mpl.rcParams['axes.unicode_minus'] = False


def fit_distribution_for_plot(baseline_vals):
    data_for_fit = baseline_vals.copy()
    if (data_for_fit == 0).any():
        eps = max(data_for_fit[data_for_fit > 0].min() / 2, 1e-3) if (data_for_fit > 0).any() else 1e-3
        data_for_fit = np.where(data_for_fit == 0, eps, data_for_fit)
    dist_type, params = None, None
    try:
        s, loc, scale = lognorm.fit(data_for_fit, floc=0)
        ll = np.sum(lognorm.logpdf(data_for_fit, s, loc=0, scale=scale))
        if np.isfinite(ll):
            dist_type, params = 'lognorm', {'s': s, 'scale': scale}
    except Exception:
        pass
    if dist_type is None:
        try:
            a, loc, scale = gamma.fit(data_for_fit, floc=0)
            ll = np.sum(gamma.logpdf(data_for_fit, a, loc=0, scale=scale))
            if np.isfinite(ll):
                dist_type, params = 'gamma', {'a': a, 'scale': scale}
        except Exception:
            pass
    return dist_type, params


def plot_acpl_distributions(player_df, baseline_df, player_name, output_dir):
    for metric in ACPL_METRICS:
        if metric not in player_df.columns or metric not in baseline_df.columns:
            continue
        tier_data = []
        tiers = sorted(player_df['tier'].dropna().unique())
        for tier in tiers:
            b_vals = baseline_df[(baseline_df['tier'] == tier) & (baseline_df[metric].notna())][metric].values
            p_vals = player_df[(player_df['tier'] == tier) & (player_df[metric].notna())][metric].values
            if len(b_vals) < MIN_BASELINE_N or len(p_vals) == 0:
                continue
            tier_data.append((tier, b_vals, p_vals))
        if not tier_data:
            logger.warning(f"⚠️ {metric} 无有效分段数据，跳过")
            continue

        n_plots = len(tier_data)
        n_cols = min(3, n_plots)
        n_rows = (n_plots + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows))
        axes = np.atleast_1d(axes).flatten()

        for ax, (tier, b_vals, p_vals) in zip(axes, tier_data):
            dist_type, params = fit_distribution_for_plot(b_vals)

            percentile, ci_low, ci_high, player_mean, n_player, matched_point = matched_n_single_tier(
                p_vals, b_vals, ACPL_N_OUTER, ACPL_K_INNER, ACPL_K_POINT
            )
            # ACPL是低优指标：这里"percentile"定义为 100*mean(matched>player_mean)，
            # 即基线均值比朱锦尔差的比例——对ACPL而言"基线值更大"=基线更差，
            # 所以这个数字已经直接是"朱锦尔优于X%基线"，无需再翻转。

            ax.hist(b_vals, bins=30, density=True, alpha=0.25, color='gray', label='单盘ACPL分布（基线）')
            x_max = max(b_vals.max(), matched_point.max()) * 1.1
            x_vals = np.linspace(0.01, x_max, 300)
            if dist_type == 'lognorm':
                pdf_vals = lognorm.pdf(x_vals, params['s'], loc=0, scale=params['scale'])
                ax.plot(x_vals, pdf_vals, color='gray', linewidth=1.5, label='单盘分布拟合曲线')
            elif dist_type == 'gamma':
                pdf_vals = gamma.pdf(x_vals, params['a'], loc=0, scale=params['scale'])
                ax.plot(x_vals, pdf_vals, color='gray', linewidth=1.5, label='单盘分布拟合曲线')

            if len(np.unique(matched_point)) > 1:
                kde = gaussian_kde(matched_point)
                kde_x = np.linspace(matched_point.min(), matched_point.max(), 300)
                kde_y = kde(kde_x)
                ax.plot(kde_x, kde_y, color='crimson', linewidth=2, label=f'{n_player}局均值分布（正确参照系）')
                fill_x = kde_x[kde_x >= player_mean]
                if len(fill_x) > 0:
                    ax.fill_between(fill_x, 0, kde(fill_x), color='crimson', alpha=0.25)

            ax.axvline(player_mean, color='blue', linestyle='--', linewidth=2,
                       label=f'{player_name}均值={player_mean:.1f}')
            ax.text(0.55, 0.85,
                    f'优于比例: {percentile:.1f}%\n95%CI: [{ci_low:.1f}%, {ci_high:.1f}%]\n(基于{n_player}盘)',
                    transform=ax.transAxes, fontsize=8,
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.85))
            ax.set_title(f'{tier} (基线n={len(b_vals)})')
            ax.legend(loc='upper right', fontsize=7)
            ax.set_xlabel(ACPL_LABELS[metric])
            ax.set_ylabel('密度')

        for j in range(len(tier_data), len(axes)):
            axes[j].set_visible(False)

        plt.tight_layout()
        out_path = output_dir / f"{player_name.replace(' ', '_')}_acpl_dist_{metric}.png"
        plt.savefig(out_path, dpi=150)
        plt.close()
        logger.info(f"✅ ACPL分布图 ({metric}) 保存至: {out_path}")
