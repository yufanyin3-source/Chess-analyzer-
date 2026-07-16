"""
viz.radar
================================================================================
风格雷达 + 开局偏好图、能力雷达图（含 95% CI 须）。
从原 extra_style.py 迁移（逻辑与输出 100% 保持不变）。
"""

import logging

logger = logging.getLogger(__name__)

import math

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

from chess_analyzer.stats.bootstrap import (compute_tier_percentiles,
    tier_stratified_category_pct, STYLE_RADAR_FEATURES, STYLE_DIRECTIONS)

mpl.rcParams['font.sans-serif'] = ["Noto Sans CJK SC", "Noto Sans CJK JP", "SimHei",
                                    "Microsoft YaHei", "PingFang SC", "WenQuanYi Zen Hei", "DejaVu Sans"]
mpl.rcParams['axes.unicode_minus'] = False


def plot_style_and_opening(player_style, lichess_style, player_opening, lichess_opening,
                            player_name, output_path):
    fig = plt.figure(figsize=(14, 6))
    gs = fig.add_gridspec(1, 2, wspace=0.3)

    # ---- 左图：风格雷达（【修复D】现在按tier分层 + 带CI须）----
    style_result = compute_tier_percentiles(player_style, lichess_style, STYLE_RADAR_FEATURES,
                                             direction_map=STYLE_DIRECTIONS)
    overall_style = {STYLE_LABEL_MAP.get(feat, feat): v for feat, v in style_result.items()}

    if len(overall_style) >= 3:
        ax1 = fig.add_subplot(gs[0, 0], projection='polar')
        features = list(overall_style.keys())
        values = [overall_style[f]['percentile'] for f in features]
        ci_lows = [overall_style[f]['ci_low'] for f in features]
        ci_highs = [overall_style[f]['ci_high'] for f in features]
        N = len(features)
        angles = [n / float(N) * 2 * math.pi for n in range(N)]
        angles_closed = angles + angles[:1]
        values_plot = values + values[:1]

        # CI radial whisker（先画，压在底层）
        for ang, lo, hi in zip(angles, ci_lows, ci_highs):
            ax1.plot([ang, ang], [lo, hi], color='steelblue', alpha=0.5, linewidth=4, solid_capstyle='round')

        ax1.plot(angles_closed, [50] * (N + 1), 'k--', linewidth=1, label='基线平均 (50分位)')
        ax1.fill(angles_closed, [50] * (N + 1), alpha=0.05, color='gray')
        ax1.plot(angles_closed, values_plot, 'o-', linewidth=2, label=player_name, color='crimson')
        ax1.fill(angles_closed, values_plot, alpha=0.2, color='crimson')
        ax1.set_xticks(angles)
        ax1.set_xticklabels(features, fontsize=9)
        ax1.set_ylim(0, 100)
        ax1.set_yticks([20, 40, 60, 80, 100])
        ax1.set_yticklabels(['20', '40', '60', '80', '100'], fontsize=7)
        ax1.grid(True)
        ax1.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1), fontsize=8)
        ax1.set_title(f'{player_name} 风格雷达 (vs Lichess Rapid, 按tier分层, 灰须=95%CI)', fontsize=11)
    else:
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.text(0.5, 0.5, '风格维度不足', ha='center', va='center')
        ax1.set_title('风格雷达')

    # ---- 右图：开局差异棒棒糖（【修复D】按tier分层加权合并）----
    ax2 = fig.add_subplot(gs[0, 1])
    if player_opening is not None and lichess_opening is not None and not player_opening.empty and not lichess_opening.empty:
        all_labels, p_pct_agg, b_pct_agg = tier_stratified_category_pct(player_opening, lichess_opening)
        if all_labels:
            p_pct = np.array([p_pct_agg[lab] for lab in all_labels])
            b_pct = np.array([b_pct_agg[lab] for lab in all_labels])
            diff = p_pct - b_pct
            idx = np.argsort(diff)
            sorted_labels = [all_labels[i] for i in idx]
            sorted_diff = diff[idx]
            colors = ['crimson' if d > 0 else 'steelblue' for d in sorted_diff]

            y_pos = np.arange(len(sorted_labels))
            ax2.hlines(y=y_pos, xmin=0, xmax=sorted_diff, color=colors, linewidth=2, alpha=0.7)
            ax2.scatter(sorted_diff, y_pos, color=colors, s=40, zorder=5)
            ax2.axvline(0, color='black', linestyle='--', linewidth=1, alpha=0.5)
            ax2.set_yticks(y_pos)
            ax2.set_yticklabels(sorted_labels, fontsize=8)
            ax2.set_xlabel('差值 (目标-基线，按tier加权) %', fontsize=10)
            ax2.set_title(f'{player_name} 开局偏好偏差 (按tier分层)', fontsize=12)
            ax2.grid(axis='x', alpha=0.3)
            ax2.text(0.98, 0.02, '红=偏好 蓝=回避', transform=ax2.transAxes,
                      fontsize=8, ha='right', va='bottom',
                      bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        else:
            ax2.text(0.5, 0.5, '无匹配tier数据', ha='center', va='center')
            ax2.set_title('开局偏好')
    else:
        ax2.text(0.5, 0.5, '无开局数据', ha='center', va='center')
        ax2.set_title('开局偏好')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"✅ 风格+开局合成图保存至: {output_path}")
    return style_result


# --------------------- 能力雷达图（1×3，带CI须）---------------------
def plot_ability_radars(ability_percentiles, player_name, output_path):
    groups = list(ability_percentiles.keys())
    if len(groups) == 0:
        logger.warning("⚠️ 无能力数据，跳过能力雷达图")
        return

    fig, axes = plt.subplots(1, len(groups), figsize=(6 * len(groups), 5), subplot_kw=dict(polar=True))
    if len(groups) == 1:
        axes = [axes]
    for ax, group_name in zip(axes, groups):
        feat_dict = ability_percentiles[group_name]
        if not feat_dict or len(feat_dict) < 2:
            ax.text(0.5, 0.5, '数据不足', ha='center', va='center')
            ax.set_title(group_name)
            continue
        features = list(feat_dict.keys())
        values = [feat_dict[f]['percentile'] for f in features]
        ci_lows = [feat_dict[f]['ci_low'] for f in features]
        ci_highs = [feat_dict[f]['ci_high'] for f in features]
        N = len(features)
        angles = [n / float(N) * 2 * math.pi for n in range(N)]
        angles_closed = angles + angles[:1]
        values_plot = values + values[:1]

        for ang, lo, hi in zip(angles, ci_lows, ci_highs):
            ax.plot([ang, ang], [lo, hi], color='steelblue', alpha=0.5, linewidth=4, solid_capstyle='round')

        ax.plot(angles_closed, [50] * (N + 1), 'k--', linewidth=1, label='TWIC平均 (50分位)')
        ax.fill(angles_closed, [50] * (N + 1), alpha=0.05, color='gray')
        ax.plot(angles_closed, values_plot, 'o-', linewidth=2, label=player_name, color='crimson')
        ax.fill(angles_closed, values_plot, alpha=0.2, color='crimson')
        ax.set_xticks(angles)
        ax.set_xticklabels(features, fontsize=8)
        ax.set_ylim(0, 100)
        ax.set_yticks([20, 40, 60, 80, 100])
        ax.set_yticklabels(['20', '40', '60', '80', '100'], fontsize=6)
        ax.grid(True)
        ax.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1), fontsize=7)
        ax.set_title(f'{group_name}（灰须=95%CI）', fontsize=10)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    logger.info(f"✅ 能力雷达图保存至: {output_path}")
