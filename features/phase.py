"""
features.phase
================================================================================
分项能力聚合：开局/中局/残局 ACPL、战术警觉性、均势残局池、过渡成功率。
从原 total.py 迁移（逻辑与输出 100% 保持不变）。
"""

import chess
import numpy as np

from chess_analyzer.core.board_utils import safe_eval_list, get_board_phase
from chess_analyzer.core.color_utils import get_relative_score
from chess_analyzer.core.config import load_config


def aggregate_phase_ability(game_meta, step_df, config=None):
    """
    聚合开局/中局/残局能力（含战术警觉性）
    - 修复：阶段 loss 仅统计目标方走法
    - 修复：均势残局池使用目标视角评估
    - 修复：过渡成功率只取目标方步数
    - 修复：战术警觉性方向（转换为目标视角比较）
    - 修复：开局偏离率分母为目标方开局步数
    - 修复：board_phase 未初始化
    - 【2026 新增修复】开局偏离率：用目标棋手第一步后的评估作为初始基准（兼容黑方）
    - 【2026 新增修复】战术警觉性：跳过 loss_cp 为 None 的步数
    - 【2026 最新修复】战术警觉性：索引由 idx+2 修正为 idx+1，正确检测目标棋手回应
    """
    moves_uci = safe_eval_list(game_meta.get('moves_uci', []))
    if not moves_uci or len(moves_uci) < 5:
        return None

    _cfg = config if config is not None else load_config()
    CONFIG = _cfg.flat("thresholds") if hasattr(_cfg, "flat") else {}
    if not CONFIG:
        CONFIG = {
            "OPENING_MOVES": 12, "DRAWN_EVAL_LIMIT": 150, "EXCELLENT_LOSS": 10,
            "PUNISH_THRESHOLD": 30, "OPPONENT_BLUNDER_DROP": 100,
        }

    target_color = game_meta['target_color']
    color = chess.WHITE if target_color == 'White' else chess.BLACK
    result = game_meta['result']
    game_id = game_meta['game_id']

    step_dict = {row['move_number']: row for _, row in step_df.iterrows()}

    opening_losses, middlegame_losses, endgame_losses = [], [], []
    mid_excellent_count, mid_total_count = 0, 0
    heavy_invasion_mid, midgame_move_count = 0, 0

    # 【修复】初始评估将记录为目标棋手第一步走完后的评估，不再固定为全局第1步
    target_initial_eval = None
    target_first_move_set = False   # 标记是否已记录目标方第一步

    opening_deviation_count = 0

    opp_opportunities = 0
    punish_success = 0
    eval_list = []

    endgame_entry_eval = None          # 白方视角原始值
    endgame_entry_move = None

    if color == chess.WHITE:
        initial_light_squares = [chess.B1, chess.G1, chess.C1, chess.F1]
    else:
        initial_light_squares = [chess.B8, chess.G8, chess.C8, chess.F8]
    developed_light_pieces = 0

    board = chess.Board()

    # 预先收集目标方 move_number
    target_moves_set = set()
    board_temp = chess.Board()
    for i, uci_move in enumerate(moves_uci, start=1):
        try:
            move = chess.Move.from_uci(uci_move)
            if board_temp.turn == color:
                target_moves_set.add(i)
            board_temp.push(move)
        except:
            break

    # 【修复】显式初始化 board_phase
    board_phase = None

    for i, uci_move in enumerate(moves_uci, start=1):
        try:
            move = chess.Move.from_uci(uci_move)
            from_sq, to_sq = move.from_square, move.to_square
            step_info = step_dict.get(i)
            if step_info is None:
                board.push(move)
                continue

            loss_cp = step_info.get('loss_cp')
            rank = step_info.get('rank_in_multipv')
            move_score = step_info.get('move_score_cp')
            is_capture = board.is_capture(move)

            # ---- 阶段分类 ----
            if i <= CONFIG["OPENING_MOVES"]:
                # 开局 loss 仅目标方
                if i in target_moves_set and loss_cp is not None:
                    opening_losses.append(loss_cp)

                # 【修复】开局偏离：使用目标方第一步后的评估作为基准
                if board.turn == color:
                    # 记录目标方第一步走完后的评估（仅记录一次）
                    if not target_first_move_set and move_score is not None:
                        target_initial_eval = get_relative_score(move_score, target_color)
                        target_first_move_set = True
                    # 后续目标方走法（非吃子）与基准比较
                    if target_initial_eval is not None and move_score is not None and not is_capture:
                        target_current_eval = get_relative_score(move_score, target_color)
                        if (target_initial_eval - target_current_eval) > 50:
                            opening_deviation_count += 1

                # ---- 开局轻子出动（在push前读棋盘） ----
                moving_piece_open = board.piece_at(from_sq)
                if (moving_piece_open and moving_piece_open.piece_type in [chess.KNIGHT, chess.BISHOP]
                        and moving_piece_open.color == color):
                    if from_sq in initial_light_squares:
                        developed_light_pieces += 1
                        initial_light_squares.remove(from_sq)

                # 记录评估值供战术警觉性使用（push前记录turn，push后eval仍有效）
                if move_score is not None:
                    eval_list.append((i, board.turn, move_score))

                board.push(move)

            else:
                # ---- 中残局：持续推进棋盘，不pop ----
                # 先读走棋前的棋盘状态（用于侵入检测）
                moving_piece_mid = board.piece_at(from_sq)
                current_turn_before_push = board.turn

                # 记录评估值（push前记录turn）
                if move_score is not None:
                    eval_list.append((i, board.turn, move_score))

                board.push(move)
                board_phase = get_board_phase(board)

                if board_phase == 'midgame':
                    # 中局仅目标方
                    if i in target_moves_set:
                        midgame_move_count += 1
                        if loss_cp is not None:
                            middlegame_losses.append(loss_cp)
                            mid_total_count += 1
                            # 【修复】从 step_dict 读动态阈值，而非硬编码 CONFIG["EXCELLENT_LOSS"]
                            excellent_threshold = step_info.get(
                                'excellent_threshold_cp', CONFIG["EXCELLENT_LOSS"]
                            )
                            if rank == 1 and loss_cp <= excellent_threshold:
                                mid_excellent_count += 1

                    # ---- 中局重子侵入（修复：深入第6/7线，rank>=5 或 rank<=2） ----
                    if (moving_piece_mid
                            and moving_piece_mid.piece_type in [chess.ROOK, chess.QUEEN]
                            and moving_piece_mid.color == color):
                        rank_sq = chess.square_rank(to_sq)
                        if (color == chess.WHITE and rank_sq >= 5) or (color == chess.BLACK and rank_sq <= 2):
                            heavy_invasion_mid += 1

                else:
                    # 残局仅目标方
                    if i in target_moves_set and loss_cp is not None:
                        endgame_losses.append(loss_cp)
                    if endgame_entry_eval is None and move_score is not None:
                        endgame_entry_eval = move_score
                        endgame_entry_move = i

        except Exception:
            try:
                board.push(chess.Move.from_uci(uci_move))
            except:
                break
            continue

    # ---- 战术警觉性（选项C：用局面评估暴跌判断对手失误，不依赖对手等级阈值） ----
    # eval_list 格式：(move_number, board.turn_before_push, move_score_cp)
    # 逻辑：
    #   找到"对手刚走完、目标棋手视角评估大幅提升"的时刻（对手出现大失误）
    #   判断目标棋手紧接着的回应是否进一步扩大优势
    #
    # OPP_BLUNDER_DROP：目标棋手视角评估在对手走完后提升超过此值，视为对手出现机会
    # 不依赖对手等级，只看局面评估变化，100cp是一个合理的"明显失误"门槛
    OPP_BLUNDER_DROP = CONFIG.get("OPPONENT_BLUNDER_DROP", 100)

    for idx in range(len(eval_list) - 1):
        num_cur,  turn_cur,  eval_cur  = eval_list[idx]
        num_next, turn_next, eval_next = eval_list[idx + 1]

        # 当前这步是对手走的（turn_cur != color），下一步是目标棋手走的
        if turn_cur == color:
            continue

        # 从目标棋手视角：对手走完后评估 vs 对手走之前评估
        # eval_list 存的是走完后的 move_score_cp（白方原始视角）
        # 需要找对手走之前的评估，即 eval_list[idx-1] 的值（如果存在）
        if idx == 0:
            continue
        _, _, eval_before_opp = eval_list[idx - 1]

        rel_before_opp = get_relative_score(eval_before_opp, target_color)
        rel_after_opp  = get_relative_score(eval_cur, target_color)

        # 目标棋手视角提升超过 OPP_BLUNDER_DROP，说明对手走砸了
        if (rel_after_opp - rel_before_opp) >= OPP_BLUNDER_DROP:
            opp_opportunities += 1
            # 目标棋手的回应：eval_next
            rel_response = get_relative_score(eval_next, target_color)
            # 回应后进一步扩大优势，视为成功把握机会
            if (rel_response - rel_after_opp) > CONFIG["PUNISH_THRESHOLD"]:
                punish_success += 1

    tactical_punish_rate = punish_success / opp_opportunities if opp_opportunities > 0 else None

    # ---- 聚合指标 ----
    opening_acpl = np.mean(opening_losses) if opening_losses else None
    middlegame_acpl = np.mean(middlegame_losses) if middlegame_losses else None
    endgame_acpl = np.mean(endgame_losses) if endgame_losses else None

    mid_peak_accuracy = mid_excellent_count / mid_total_count if mid_total_count > 0 else 0.0
    mid_invasion_rate = heavy_invasion_mid / midgame_move_count if midgame_move_count > 0 else 0.0

    # ----- 开局偏离率（分母为目标方开局步数） -----
    opening_moves_limit = min(len(moves_uci), CONFIG["OPENING_MOVES"])
    opening_target_moves = sum(1 for m in range(1, opening_moves_limit + 1) if m in target_moves_set)
    opening_deviation_rate = opening_deviation_count / opening_target_moves if opening_target_moves > 0 else 0.0

    # ----- 均势残局池（关键修复：转换为目标视角） -----
    valid_endgame_pool = False
    endgame_win_rate = None
    if endgame_entry_eval is not None:
        rel_entry_eval = get_relative_score(endgame_entry_eval, target_color)
        if abs(rel_entry_eval) <= CONFIG["DRAWN_EVAL_LIMIT"]:
            valid_endgame_pool = True
            if ((target_color == 'White' and result == '1-0') or
                (target_color == 'Black' and result == '0-1')):
                endgame_win_rate = 1.0
            else:
                endgame_win_rate = 0.0
    else:
        rel_entry_eval = None

    # ----- 中局→残局过渡成功率（只取目标方步数） -----
    transition_success = 0
    if endgame_entry_move is not None and endgame_entry_move > 3:
        pre_moves = [m for m in range(max(1, endgame_entry_move - 3), endgame_entry_move) if m in target_moves_set]
        pre_losses = []
        for m in pre_moves:
            info = step_dict.get(m)
            if info is not None and info.get('loss_cp') is not None:
                pre_losses.append(info['loss_cp'])
        if pre_losses and endgame_acpl is not None:
            if np.mean(pre_losses) <= endgame_acpl:
                transition_success = 1

    return {
        'game_id': game_id,
        'target_color': target_color,
        'result': result,
        'total_moves': len(moves_uci),
        'opening_acpl': opening_acpl,
        'opening_deviation_rate': opening_deviation_rate,
        'developed_light_pieces': developed_light_pieces,
        'middlegame_acpl': middlegame_acpl,
        'mid_peak_accuracy': mid_peak_accuracy,
        'middlegame_invasion_rate': mid_invasion_rate,
        'tactical_punish_rate': tactical_punish_rate,
        'endgame_acpl': endgame_acpl,
        'valid_endgame_pool': valid_endgame_pool,
        'endgame_win_rate': endgame_win_rate,
        'transition_success': transition_success,
        'entry_endgame_eval': rel_entry_eval,  # 目标视角
    }
