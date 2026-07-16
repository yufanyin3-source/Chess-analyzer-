"""
features.mental
================================================================================
心态与体力指标：逆转率、遇强弱手偏差、体能衰减、Tilt、时间压力ACPL。
从原 total.py 迁移（逻辑与输出 100% 保持不变）。
"""

import chess
import numpy as np
import pandas as pd

from chess_analyzer.core.board_utils import safe_eval_list
from chess_analyzer.core.color_utils import get_relative_score
from chess_analyzer.core.config import load_config


def extract_time_pressure_acpl(meta_row, step_df, pressure_threshold_seconds=30, config=None):
    """
    计算时间压力下的ACPL。
    从 meta_row 的 clock_times 列读时钟数据（separate_lichess.py格式：空格分隔，?表示缺失）。
    找出目标棋手剩余时间 < pressure_threshold_seconds 的走法，计算这些步的平均ACPL。
    无时钟数据或无压力时刻时返回 None。
    """
    clock_str = meta_row.get('clock_times', None)
    if not isinstance(clock_str, str) or not clock_str.strip():
        return None, None

    # 解析时钟序列
    parts = clock_str.strip().split()
    clock_vals = []
    for p in parts:
        if p == '?':
            clock_vals.append(None)
        else:
            try:
                clock_vals.append(float(p))
            except ValueError:
                clock_vals.append(None)

    if not clock_vals:
        return None, None

    target_color = meta_row['target_color']
    color = chess.WHITE if target_color == 'White' else chess.BLACK

    # 白方走法在偶数半步（ply 1,3,5...），黑方在奇数半步（ply 2,4,6...）
    # clock_vals 按全局半步顺序排列
    own_clock_indices = []
    moves_uci = safe_eval_list(meta_row.get('moves_uci', []))
    board_tmp = chess.Board()
    for i, uci in enumerate(moves_uci):
        try:
            move = chess.Move.from_uci(uci)
            if board_tmp.turn == color:
                own_clock_indices.append(i)
            board_tmp.push(move)
        except Exception:
            break

    # 找出时间压力步（目标棋手剩余时间 < threshold）
    pressure_move_numbers = []
    for idx in own_clock_indices:
        if idx < len(clock_vals) and clock_vals[idx] is not None:
            if clock_vals[idx] < pressure_threshold_seconds:
                pressure_move_numbers.append(idx + 1)  # move_number从1开始

    if not pressure_move_numbers:
        return None, 0  # 没有时间压力时刻，返回压力步数为0

    # 从 step_df 里取这些步的 loss_cp
    pressure_steps = step_df[step_df['move_number'].isin(pressure_move_numbers)]
    pressure_losses = pressure_steps['loss_cp'].dropna()

    if pressure_losses.empty:
        return None, len(pressure_move_numbers)

    return float(pressure_losses.mean()), len(pressure_move_numbers)


def compute_mental_metrics(df_meta, df_steps, df_agg, df_phase, data_source='player', config=None):
    """
    计算心态/体力指标。

    DATA_SOURCE == 'player'：
        正常计算全部指标（遇强弱手偏差、逆转率、体能衰减、Tilt、时间压力）
    DATA_SOURCE == 'baseline'：
        跨对局指标（遇强弱手偏差、逆转率、Tilt）输出 NaN——线A单盘抽样无棋手历史，
        这些指标没有统计意义。体能衰减和时间压力是单盘内指标，正常计算。
    """
    results = []
    detail_by_game = {gid: group for gid, group in df_steps.groupby('game_id')}
    agg_dict = {row['game_id']: row for _, row in df_agg.iterrows()}
    phase_dict = {row['game_id']: row for _, row in df_phase.iterrows()}

    all_mid_acpl = df_phase['middlegame_acpl'].dropna()
    if all_mid_acpl.empty:
        return pd.DataFrame()
    global_avg_mid_acpl = all_mid_acpl.mean()
    all_avg_acpl = df_agg['avg_acpl'].dropna()
    if all_avg_acpl.empty:
        return pd.DataFrame()

    _cfg = config if config is not None else load_config()
    CONFIG = _cfg.flat("thresholds") if hasattr(_cfg, "flat") else {}
    if not CONFIG:
        CONFIG = {"RESILIENCE_THRESHOLD": -80, "STRONG_ELO_DIFF": -100, "WEAK_ELO_DIFF": 100}

    is_player_mode = (data_source == 'player')

    for _, meta_row in df_meta.iterrows():
        gid = meta_row['game_id']
        target_color = meta_row['target_color']
        result = meta_row['result']
        elo_diff = meta_row.get('elo_diff', 0)
        event = meta_row.get('event', '')
        round_num = meta_row.get('round', '')
        date = meta_row.get('date', '')

        step_df = detail_by_game.get(gid)
        agg_row = agg_dict.get(gid)
        phase_row = phase_dict.get(gid)
        if step_df is None or agg_row is None or phase_row is None:
            continue

        # ---- 劣势逆转（仅 player 模式） ----
        was_behind = False
        resilience = None
        if is_player_mode:
            for _, step in step_df.iterrows():
                raw_eval = step.get('move_score_cp')
                if raw_eval is not None:
                    relative_eval = get_relative_score(raw_eval, target_color)
                    if relative_eval <= CONFIG["RESILIENCE_THRESHOLD"]:
                        was_behind = True
                        break
            if was_behind:
                is_win = (target_color == 'White' and result == '1-0') or \
                         (target_color == 'Black' and result == '0-1')
                is_draw = (result == '1/2-1/2')
                resilience = 1 if (is_win or is_draw) else 0

        # ---- 强/弱手局偏差（仅 player 模式） ----
        acpl_vs_strong = None
        acpl_vs_weak = None
        if is_player_mode:
            if elo_diff is not None and elo_diff <= CONFIG["STRONG_ELO_DIFF"]:
                mid_acpl = phase_row.get('middlegame_acpl')
                if mid_acpl is not None:
                    acpl_vs_strong = mid_acpl - global_avg_mid_acpl
            if elo_diff is not None and elo_diff >= CONFIG["WEAK_ELO_DIFF"]:
                mid_acpl = phase_row.get('middlegame_acpl')
                if mid_acpl is not None:
                    acpl_vs_weak = mid_acpl - global_avg_mid_acpl

        # ---- 体能衰减（单盘内指标，两种模式都算） ----
        fatigue_effect = None
        moves_uci = safe_eval_list(meta_row.get('moves_uci', []))
        color = chess.WHITE if target_color == 'White' else chess.BLACK
        target_moves = []
        board_temp = chess.Board()
        for move_num, uci in enumerate(moves_uci, start=1):
            try:
                move = chess.Move.from_uci(uci)
                if board_temp.turn == color:
                    target_moves.append(move_num)
                board_temp.push(move)
            except Exception:
                break
        if target_moves:
            own_losses = step_df[step_df['move_number'].isin(
                target_moves)]['loss_cp'].dropna().tolist()
            if len(own_losses) > 10:
                half = len(own_losses) // 2
                fatigue_effect = np.mean(own_losses[half:]) - np.mean(own_losses[:half])

        # ---- 时间压力ACPL（单盘内指标，两种模式都算，无时钟时输出NaN） ----
        time_pressure_acpl, pressure_move_count = extract_time_pressure_acpl(
            meta_row, step_df
        )

        results.append({
            'game_id':              gid,
            'target_color':         target_color,
            'result':               result,
            'event':                event,
            'round':                round_num,
            'date':                 date,
            'elo_diff':             elo_diff,
            'was_behind':           was_behind if is_player_mode else None,
            'resilience':           resilience,          # NaN if baseline
            'acpl_vs_strong':       acpl_vs_strong,      # NaN if baseline
            'acpl_vs_weak':         acpl_vs_weak,        # NaN if baseline
            'fatigue_effect':       fatigue_effect,
            'self_avg_acpl':        agg_row.get('avg_acpl'),
            'self_mid_acpl':        phase_row.get('middlegame_acpl'),
            'time_pressure_acpl':   time_pressure_acpl,  # NaN if no clock data
            'pressure_move_count':  pressure_move_count, # 发生时间压力的步数
        })

    df_out = pd.DataFrame(results)
    if df_out.empty:
        return df_out

    # ---- Tilt（仅 player 模式，需要跨对局排序） ----
    df_out['tilt_effect'] = None
    df_out['round_num'] = pd.to_numeric(df_out['round'], errors='coerce')
    df_out['date_parsed'] = pd.to_datetime(df_out['date'], errors='coerce')

    if is_player_mode:
        for event_name, group in df_out.groupby('event'):
            if len(group) < 2:
                continue
            event_avg_acpl = group['self_avg_acpl'].dropna().mean()
            if pd.isna(event_avg_acpl):
                continue
            group_sorted = group.sort_values(
                by=['round_num', 'date_parsed'],
                ascending=[True, True],
                na_position='last'
            )
            sorted_indices = group_sorted.index.tolist()
            for i in range(1, len(sorted_indices)):
                prev_idx = sorted_indices[i - 1]
                curr_idx = sorted_indices[i]
                prev_result = df_out.loc[prev_idx, 'result']
                prev_target = df_out.loc[prev_idx, 'target_color']
                is_prev_loss = (
                    (prev_target == 'White' and prev_result == '0-1') or
                    (prev_target == 'Black' and prev_result == '1-0')
                )
                if is_prev_loss:
                    curr_acpl = df_out.loc[curr_idx, 'self_avg_acpl']
                    if curr_acpl is not None and not pd.isna(curr_acpl):
                        df_out.loc[curr_idx, 'tilt_effect'] = (
                            curr_acpl - event_avg_acpl
                        )

    return df_out
