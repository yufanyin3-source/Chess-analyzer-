"""
pipeline.report
================================================================================
棋手综合分析报告：风格雷达+开局偏好图、能力雷达图、ACPL分布图、
百分位汇总CSV、风格诊断文本、可选ML特征导出。
从原 extra_style.py 的 main() 迁移为 run_style_report(...) 函数式调用，
并接入 configs/default.yaml 的 `report` 段（数据路径/棋手名可配置，
未提供 config 时使用与原脚本完全一致的默认路径）。
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

import pandas as pd

from chess_analyzer.core.color_utils import assign_tier
from chess_analyzer.core.config import load_config
from chess_analyzer.stats.bootstrap import (
    load_and_merge, compute_style_rates, compute_tier_percentiles,
    validate_direction_maps, ABILITY_GROUPS, ABILITY_DIRECTIONS,
    DEVIATION_HIGH, DEVIATION_LOW,
)
from chess_analyzer.viz.radar import plot_style_and_opening, plot_ability_radars
from chess_analyzer.viz.distribution import plot_acpl_distributions

logger = logging.getLogger(__name__)


def run_style_report(config: Optional[dict] = None, export_ml: bool = False) -> dict:
    """生成棋手综合分析报告。返回 {'style_percentiles': ..., 'ability_percentiles': ...}。

    与原 extra_style.py main() 逻辑 100% 一致，仅将模块级路径常量替换为
    configs/default.yaml 的 `report` 段（未提供 config 时使用相同的默认值）。
    """
    cfg = config if config is not None else load_config()
    report_cfg = cfg.get("report", {}) if hasattr(cfg, "get") else {}

    lichess_style_file = report_cfg.get("lichess_style_file", "baseline_sample_new/rapid_style_features.parquet")
    lichess_meta_file = report_cfg.get("lichess_meta_file", "baseline_sample_new/rapid.parquet")
    lichess_opening_file = report_cfg.get("lichess_opening_file", "baseline_sample_new/rapid_opening_detail.parquet")
    twic_phase_file = report_cfg.get("twic_phase_file", "twic_sample/classical_phase_ability_full.parquet")
    twic_meta_file = report_cfg.get("twic_meta_file", "twic_sample/classical.parquet")
    player_style_file = report_cfg.get("player_style_file", "Zhu_jiner_2547/classical_style_features.parquet")
    player_meta_file = report_cfg.get("player_meta_file", "Zhu_jiner_2547/classical.parquet")
    player_phase_file = report_cfg.get("player_phase_file", "Zhu_jiner_2547/classical_phase_ability_full.parquet")
    player_opening_file = report_cfg.get("player_opening_file", "Zhu_jiner_2547/classical_opening_detail.parquet")
    player_name = report_cfg.get("player_name", "Zhu jiner")
    output_dir_str = report_cfg.get("output_dir", "player_style")

    validate_direction_maps()

    out_dir = Path(output_dir_str)
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("棋手综合分析 - %s", player_name)
    logger.info("=" * 60)

    logger.info("📂 加载风格数据...")
    try:
        lichess_style = load_and_merge(lichess_style_file, lichess_meta_file)
        lichess_style = compute_style_rates(lichess_style)
        logger.info("   Lichess 风格样本: %d 盘", len(lichess_style))
    except Exception as e:
        logger.error("❌ 加载 Lichess 风格失败: %s", e)
        sys.exit(1)

    try:
        player_style = load_and_merge(player_style_file, player_meta_file)
        player_style = compute_style_rates(player_style)
        logger.info("   %s 风格样本: %d 盘", player_name, len(player_style))
    except Exception as e:
        logger.error("❌ 加载目标风格失败: %s", e)
        sys.exit(1)

    logger.info("📂 加载能力数据...")
    try:
        twic_phase = load_and_merge(twic_phase_file, twic_meta_file)
        logger.info("   TWIC 能力样本: %d 盘", len(twic_phase))
    except Exception as e:
        logger.error("❌ 加载 TWIC 能力失败: %s", e)
        sys.exit(1)

    try:
        player_phase = load_and_merge(player_phase_file, player_meta_file)
        logger.info("   %s 能力样本: %d 盘", player_name, len(player_phase))
    except Exception as e:
        logger.error("❌ 加载目标能力失败: %s", e)
        sys.exit(1)

    player_opening, lichess_opening = None, None
    if Path(player_opening_file).exists():
        try:
            player_opening = pd.read_parquet(player_opening_file)
            if 'tier' not in player_opening.columns:
                meta = pd.read_parquet(player_meta_file)
                meta['tier'] = meta['target_elo'].apply(assign_tier)
                player_opening = player_opening.merge(meta[['game_id', 'tier']], on='game_id', how='left')
        except Exception as e:
            logger.warning("⚠️ 加载目标开局失败: %s", e)
    if Path(lichess_opening_file).exists():
        try:
            lichess_opening = pd.read_parquet(lichess_opening_file)
            if 'tier' not in lichess_opening.columns:
                meta = pd.read_parquet(lichess_meta_file)
                meta['tier'] = meta['target_elo'].apply(assign_tier)
                lichess_opening = lichess_opening.merge(meta[['game_id', 'tier']], on='game_id', how='left')
        except Exception as e:
            logger.warning("⚠️ 加载 Lichess 开局失败: %s", e)

    # ---- 风格+开局合并图 ----
    style_pct_dict = plot_style_and_opening(
        player_style, lichess_style,
        player_opening, lichess_opening,
        player_name,
        out_dir / f"{player_name.replace(' ', '_')}_style_opening.png"
    )

    # ---- 三能力雷达图（用同一个 ABILITY_DIRECTIONS 计算，CSV导出复用同一份结果）----
    ability_radar_data = {}
    ability_pct_all = {}
    for group_name, group_info in ABILITY_GROUPS.items():
        feats = group_info['features']
        labels = group_info['labels']
        pct_dict = compute_tier_percentiles(player_phase, twic_phase, feats, direction_map=ABILITY_DIRECTIONS)
        ability_pct_all.update(pct_dict)  # 雷达图和CSV共用同一份已翻转的结果
        radar_entry = {}
        for feat, label in zip(feats, labels):
            if feat in pct_dict:
                radar_entry[label] = pct_dict[feat]
        ability_radar_data[group_name] = radar_entry
    plot_ability_radars(ability_radar_data, player_name,
                         out_dir / f"{player_name.replace(' ', '_')}_ability_radar.png")

    # ---- ACPL 分布图 ----
    plot_acpl_distributions(player_phase, twic_phase, player_name, out_dir)

    # ---- 汇总百分位表（含CI，风格和能力共用同一批 direction 修正过的数据）----
    all_percentiles = {}
    for feat, res in style_pct_dict.items():
        all_percentiles[f'style_{feat}_percentile'] = res['percentile']
        all_percentiles[f'style_{feat}_ci_low'] = res['ci_low']
        all_percentiles[f'style_{feat}_ci_high'] = res['ci_high']
        all_percentiles[f'style_{feat}_n_games'] = res['n_games']
    for feat, res in ability_pct_all.items():
        all_percentiles[f'ability_{feat}_percentile'] = res['percentile']
        all_percentiles[f'ability_{feat}_ci_low'] = res['ci_low']
        all_percentiles[f'ability_{feat}_ci_high'] = res['ci_high']
        all_percentiles[f'ability_{feat}_n_games'] = res['n_games']

    df_pct = pd.DataFrame([all_percentiles])
    df_pct.insert(0, 'player', player_name)
    csv_path = out_dir / f"{player_name.replace(' ', '_')}_percentiles.csv"
    df_pct.to_csv(csv_path, index=False, encoding='utf-8-sig')
    logger.info("✅ 百分位汇总表保存至: %s", csv_path)

    # ---- 风格诊断 ----
    if style_pct_dict:
        diag_lines = [f"【{player_name} 风格偏差诊断】", "=" * 60]
        for feat, res in style_pct_dict.items():
            val = res['percentile']
            if val >= 50 + DEVIATION_HIGH:
                direction_desc = "极度倾向"
            elif val <= 50 + DEVIATION_LOW:
                direction_desc = "极度回避"
            else:
                direction_desc = "适中"
            diag_lines.append(
                f"{feat:30s} 百分位: {val:5.1f}% [{res['ci_low']:.1f}%~{res['ci_high']:.1f}%]   {direction_desc}"
                f"  (基于{res['n_games']}盘)"
            )
        diag_lines.append("=" * 60)
        diag_lines.append(f"标准: ≥{50+DEVIATION_HIGH}% 极度倾向，≤{50+DEVIATION_LOW}% 极度回避")
        diag_lines.append("注：区间为95%置信区间，样本盘数越少区间越宽，属正常现象。")
        with open(out_dir / "style_diagnosis.txt", 'w', encoding='utf-8') as f:
            f.write('\n'.join(diag_lines))
        logger.info("✅ 诊断报告保存至: style_diagnosis.txt")

    # ---- ML特征导出（可选，导出原始均值供下游模型使用，与展示层的百分位相互独立）----
    if export_ml:
        logger.info("📊 导出 ML 特征向量...")
        ml_record = {'player_name': player_name, 'total_games': len(player_style)}
        style_rate_cols = [c for c in player_style.columns if c.endswith('_rate')]
        for col in style_rate_cols:
            ml_record[f'style_{col}'] = player_style[col].mean()
        all_ability_feats = list(ABILITY_DIRECTIONS.keys())
        for col in all_ability_feats:
            if col in player_phase.columns:
                ml_record[f'ability_{col}'] = player_phase[col].mean()
        mental_file = Path(str(player_phase_file).replace('phase_ability_full', 'mental_fatigue'))
        if mental_file.exists():
            try:
                player_mental = pd.read_parquet(mental_file)
                for col in ['fatigue_effect', 'resilience', 'acpl_vs_strong', 'acpl_vs_weak',
                            'tilt_effect', 'time_pressure_acpl']:
                    if col in player_mental.columns:
                        ml_record[f'mental_{col}'] = player_mental[col].mean()
            except Exception:
                pass
        ml_df = pd.DataFrame([ml_record])
        ml_path = out_dir / f"{player_name.replace(' ', '_')}_ml_features.parquet"
        ml_df.to_parquet(ml_path, index=False)
        logger.info("✅ ML特征向量保存至: %s", ml_path)

    logger.info("=" * 60)
    logger.info("✅ 全部完成！")
    logger.info("   输出目录: %s", out_dir)
    logger.info("=" * 60)

    return {'style_percentiles': style_pct_dict, 'ability_percentiles': ability_pct_all}


def main() -> None:
    """CLI 入口，等价于原 `python extra_style.py [--export-ml]`。"""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description='棋手综合分析')
    parser.add_argument('--export-ml', action='store_true', help='导出ML特征向量')
    args = parser.parse_args()
    run_style_report(export_ml=args.export_ml)


if __name__ == "__main__":
    main()
