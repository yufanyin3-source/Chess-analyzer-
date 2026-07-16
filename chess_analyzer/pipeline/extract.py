"""
pipeline.extract
================================================================================
特征提取流水线：合并版风格 / 分项能力 / 开局 / 心态特征生成。
从原 total.py 的 main() 迁移为 run_feature_pipeline(...) 函数式调用
（任务一要求），并接入 configs/default.yaml（任务二要求）。

输出文件名、列名、计算逻辑与原 total.py 100% 保持一致：
  - {category}_style_features.parquet
  - {category}_phase_ability_full.parquet
  - {category}_opening_detail.parquet
  - {category}_mental_fatigue.parquet
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

import chess
import chess.engine
import numpy as np
import pandas as pd

from chess_analyzer.core.board_utils import safe_eval_list
from chess_analyzer.core.config import load_config
from chess_analyzer.features.style import extract_style_features
from chess_analyzer.features.phase import aggregate_phase_ability
from chess_analyzer.features.opening import classify_opening
from chess_analyzer.features.mental import compute_mental_metrics

logger = logging.getLogger(__name__)

# 中文列名映射（仅用于日志摘要打印，不影响任何 DataFrame 的实际列名/输出文件）
_COL_NAME_MAP = {
    'game_id': '对局ID', 'target_color': '执子颜色', 'result': '结果',
    'win_path': '赢棋路径', 'effective_sacrifices': '有效弃子数',
    'pawn_storm_kingside': '王翼兵风暴', 'pawn_storm_queenside': '后翼兵风暴',
    'pawn_storm_center': '中心兵风暴', 'total_pawn_storm': '兵风暴总数',
    'heavy_invasion_count': '重子侵入次数', 'total_moves': '总步数',
    'opening_acpl': '开局平均ACPL', 'opening_deviation_rate': '开局偏离率',
    'developed_light_pieces': '轻子出动数', 'middlegame_acpl': '中局平均ACPL',
    'mid_peak_accuracy': '中局卓越率', 'middlegame_invasion_rate': '中局侵入率',
    'tactical_punish_rate': '战术警觉性', 'endgame_acpl': '残局平均ACPL',
    'valid_endgame_pool': '均势残局池', 'endgame_win_rate': '均势残局胜率',
    'transition_success': '过渡成功率', 'entry_endgame_eval': '残局入场评估',
    'wing': '翼', 'variation': '变例', 'label': '开局标签',
    'white_first': '白棋首步', 'is_win': '胜', 'is_draw': '和', 'is_loss': '负',
    'resilience': '逆转率', 'acpl_vs_strong': '遇强手偏差',
    'acpl_vs_weak': '遇弱手偏差', 'fatigue_effect': '体能衰减',
    'tilt_effect': '输棋后反弹', 'self_avg_acpl': '全局平均ACPL',
    'time_pressure_acpl': '时间压力ACPL', 'pressure_move_count': '时间压力步数',
    'self_mid_acpl': '中局平均ACPL', 'event': '赛事', 'round': '轮次',
    'date': '日期', 'elo_diff': '等级分差', 'was_behind': '曾落后'
}


def print_summary(df: pd.DataFrame, name: str, data_source: str = 'player') -> None:
    """打印DataFrame的基本信息和关键统计（中文版）。逻辑与原 total.py 完全一致。"""
    if df is None or len(df) == 0:
        logger.warning("%s 为空，无概要可打印。", name)
        return

    logger.info("📊 %s 概要:", name)
    logger.info("   - 行数: %d", len(df))
    cols_cn = [_COL_NAME_MAP.get(c, c) for c in df.columns]
    logger.info("   - 列: %s", ', '.join(cols_cn))

    num_cols = df.select_dtypes(include=[np.number]).columns
    if len(num_cols) > 0:
        logger.info("   - 主要数值指标均值（前5项）：")
        for col in num_cols[:5]:
            mean_val = df[col].mean()
            if pd.notna(mean_val):
                cn_name = _COL_NAME_MAP.get(col, col)
                logger.info("      %s: %.4f", cn_name, mean_val)

    if 'label' in df.columns:
        logger.info("   - 开局标签分布（前5项）：")
        for label, cnt in df['label'].value_counts().head(5).items():
            logger.info("      %s: %d", label, cnt)

    if 'resilience' in df.columns or 'time_pressure_acpl' in df.columns:
        logger.info("   - 心态/体力指标（有效样本均值）：")
        core_cols = ['resilience', 'acpl_vs_strong', 'acpl_vs_weak',
                     'fatigue_effect', 'tilt_effect',
                     'time_pressure_acpl', 'pressure_move_count']
        for col in core_cols:
            if col in df.columns:
                vals = df[col].dropna()
                cn_name = _COL_NAME_MAP.get(col, col)
                if len(vals) > 0:
                    logger.info("      %s: %.4f (样本数=%d)", cn_name, vals.mean(), len(vals))
                else:
                    reason = 'baseline模式跳过' if data_source == 'baseline' else '无有效数据'
                    logger.info("      %s: NaN（%s）", cn_name, reason)


def run_feature_pipeline(
    meta_path: Optional[str | Path] = None,
    step_path: Optional[str | Path] = None,
    agg_path: Optional[str | Path] = None,
    output_dir: Optional[str | Path] = None,
    config: Optional[dict] = None,
) -> dict:
    """合并版特征流水线：一次性生成风格 / 分项能力 / 开局 / 心态四张表。

    与原 total.py main() 的唯一区别是接受显式路径与 config 对象，
    而不是读取模块级全局变量；计算逻辑、输出列、输出文件名 100% 保持不变。

    参数缺省时，使用 configs/default.yaml 的 project.category /
    project.input_dir / project.output_dir 派生路径（与原脚本硬编码的
    CATEGORY="classical", INPUT_DIR=Path("classical") 行为完全一致）。

    返回: {'style': df, 'phase': df, 'opening': df, 'mental': df}（缺失的表为 None）
    """
    cfg = config if config is not None else load_config()
    project_cfg = cfg.get("project", {}) if hasattr(cfg, "get") else {}
    category = project_cfg.get("category", "classical")
    input_dir = Path(project_cfg.get("input_dir", category))
    default_output_dir = Path(project_cfg.get("output_dir", str(input_dir / "style")))

    data_source = project_cfg.get("data_source", "player")

    output_dir = Path(output_dir) if output_dir is not None else default_output_dir
    meta_path = Path(meta_path) if meta_path is not None else input_dir / f"{category}.parquet"
    step_path = Path(step_path) if step_path is not None else input_dir / f"{category}_gm_detailed_evals.parquet"
    agg_path = Path(agg_path) if agg_path is not None else input_dir / f"{category}_gm_aggregated_metrics.parquet"

    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("🚀 启动合并版特征流水线，分类：%s", category.upper())

    if not meta_path.exists():
        logger.error("❌ 元数据不存在: %s，请先运行 curation.py", meta_path)
        return {}
    if not step_path.exists() or not agg_path.exists():
        logger.error("❌ 评估数据不存在，请先运行 ACPL.py")
        return {}

    logger.info("📂 加载数据...")
    df_meta = pd.read_parquet(meta_path)
    df_steps = pd.read_parquet(step_path)
    df_agg = pd.read_parquet(agg_path)
    logger.info("   ✅ 元数据: %d 盘，步级明细: %d 步，汇总: %d 盘", len(df_meta), len(df_steps), len(df_agg))

    # ---- 新版 curation 字段兼容处理 ----
    if 'target_elo' not in df_meta.columns:
        logger.info("   ℹ️ 未找到 target_elo 列（旧版curation），有效弃子将使用默认阈值")
        df_meta['target_elo'] = None
    if 'time_weight' not in df_meta.columns:
        logger.info("   ℹ️ 未找到 time_weight 列（旧版curation），权重设为1.0")
        df_meta['time_weight'] = 1.0
    if 'career_phase' not in df_meta.columns:
        df_meta['career_phase'] = 'phase_1'

    # ---- 尝试复用 ACPL 里的引擎实例做有效弃子主动性检测 ----
    # 如果不想启动引擎可以设为 None，此时有效弃子退化为兜底逻辑
    _engine_for_sacrifice = None
    stockfish_cfg = cfg.get("stockfish", {}) if hasattr(cfg, "get") else {}
    _sf_path = stockfish_cfg.get("path", "/usr/local/bin/stockfish")
    _sf_memory = stockfish_cfg.get("memory", 128)
    if os.path.exists(_sf_path):
        try:
            _engine_for_sacrifice = chess.engine.SimpleEngine.popen_uci(_sf_path)
            _engine_for_sacrifice.configure({"Hash": _sf_memory})
            logger.info("   ✅ Stockfish 已启动（用于有效弃子主动性检测，深度8浅搜）")
        except Exception as e:
            logger.warning("   ⚠️ Stockfish 启动失败（%s），有效弃子将使用兜底逻辑", e)
    else:
        logger.info("   ℹ️ 未找到 Stockfish，有效弃子主动性检测将使用兜底逻辑")

    steps_by_game = {gid: group for gid, group in df_steps.groupby('game_id')}

    style_rows, phase_rows, opening_rows = [], [], []

    logger.info("🔄 逐盘提取风格、分项能力和开局...")

    for idx, (_, meta_row) in enumerate(df_meta.iterrows(), 1):
        gid = meta_row['game_id']
        step_df = steps_by_game.get(gid)
        if step_df is None or step_df.empty:
            continue

        step_dict = {row['move_number']: row for _, row in step_df.iterrows()}

        moves_uci = safe_eval_list(meta_row.get('moves_uci', []))
        color = chess.WHITE if meta_row['target_color'] == 'White' else chess.BLACK
        target_indices = []
        board = chess.Board()
        for uci_move in moves_uci:
            try:
                move = chess.Move.from_uci(uci_move)
                if board.turn == color:
                    target_indices.append(len(board.move_stack) + 1)
                board.push(move)
            except Exception:
                break

        if idx == 1:
            logger.debug("调试: %s target_indices前5个 = %s", gid, target_indices[:5])
            logger.debug("调试: step_dict keys 前5个 = %s", list(step_dict.keys())[:5])

        style = extract_style_features(meta_row, step_dict, target_indices,
                                        engine=_engine_for_sacrifice, config=cfg)
        if style is not None:
            style_rows.append(style)
        else:
            if idx <= 5:
                logger.warning("⚠️ 警告: %s 风格特征提取失败", gid)

        phase = aggregate_phase_ability(meta_row, step_df, config=cfg)
        if phase:
            phase_rows.append(phase)

        opening = classify_opening(meta_row, config=cfg)
        if opening:
            opening_rows.append(opening)

        if idx % 10 == 0:
            logger.info("   🏃 进度: %d/%d 盘", idx, len(df_meta))

    logger.info("✅ 提取完成：风格 %d 盘，分项 %d 盘，开局 %d 盘",
                len(style_rows), len(phase_rows), len(opening_rows))

    df_style = pd.DataFrame(style_rows) if style_rows else None
    df_phase = pd.DataFrame(phase_rows) if phase_rows else None
    df_opening = pd.DataFrame(opening_rows) if opening_rows else None
    df_mental = None

    if df_style is not None and not df_style.empty:
        style_path = output_dir / f"{category}_style_features.parquet"
        df_style.to_parquet(style_path, index=False)
        logger.info("💾 保存风格特征: %s", style_path)
        print_summary(df_style, "风格特征表", data_source)
    else:
        logger.warning("⚠️ 风格特征为空，未保存")

    if df_phase is not None and not df_phase.empty:
        phase_path = output_dir / f"{category}_phase_ability_full.parquet"
        df_phase.to_parquet(phase_path, index=False)
        logger.info("💾 保存分项能力: %s", phase_path)
        print_summary(df_phase, "分项能力表", data_source)
    else:
        logger.warning("⚠️ 分项能力为空，未保存")

    if df_opening is not None and not df_opening.empty:
        opening_path = output_dir / f"{category}_opening_detail.parquet"
        df_opening.to_parquet(opening_path, index=False)
        logger.info("💾 保存开局偏好: %s", opening_path)
        print_summary(df_opening, "开局偏好表", data_source)
    else:
        logger.warning("⚠️ 开局偏好为空，未保存")

    if df_phase is not None and not df_phase.empty:
        df_mental = compute_mental_metrics(df_meta, df_steps, df_agg, df_phase,
                                            data_source=data_source, config=cfg)
        if df_mental is not None and not df_mental.empty:
            mental_path = output_dir / f"{category}_mental_fatigue.parquet"
            df_mental.to_parquet(mental_path, index=False)
            logger.info("💾 保存心态/体力: %s", mental_path)
            print_summary(df_mental, "心态体力表", data_source)
        else:
            logger.warning("⚠️ 心态数据为空（可能缺少中局ACPL）")
    else:
        logger.warning("⚠️ 缺少分项数据，无法计算心态指标")

    if _engine_for_sacrifice is not None:
        try:
            _engine_for_sacrifice.quit()
        except Exception:
            pass

    logger.info("🎉 全部完成！合并版流水线执行成功。")

    return {'style': df_style, 'phase': df_phase, 'opening': df_opening, 'mental': df_mental}


def main() -> None:
    """CLI 入口，等价于原 `python total.py`。"""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    run_feature_pipeline()


if __name__ == "__main__":
    main()
