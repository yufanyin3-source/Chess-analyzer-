"""
features.style
================================================================================
风格特征提取：有效弃子检测、兵风暴、重子侵入、赢棋路径判定。
从原 total.py 迁移（逻辑与输出 100% 保持不变）。
"""

import chess

from chess_analyzer.core.board_utils import safe_eval_list, piece_value, get_board_phase
from chess_analyzer.core.color_utils import get_relative_score
from chess_analyzer.core.config import load_config


def get_sacrifice_drop_threshold(target_elo, config=None):
    """
    根据棋手等级分返回有效弃子的评估暴跌容忍阈值（cp）。
    等级越高，对"弃子后局面不能太差"的要求越严格。

    阈值分段现在从 configs/default.yaml 的 `sacrifice_thresholds` 读取
    （未提供 config 时使用与原脚本完全一致的硬编码值：120 默认，
    1600/1800/2000/2200/2400/9999 分段对应 180/150/130/110/90/70）。
    """
    cfg = config if config is not None else load_config()
    thresholds = cfg.get("sacrifice_thresholds") if isinstance(cfg, dict) else None
    if not thresholds:
        thresholds = {1600: 180, 1800: 150, 2000: 130, 2200: 110, 2400: 90, 9999: 70}

    if target_elo is None:
        return 120   # 默认中等水平
    try:
        elo = int(target_elo)
    except (ValueError, TypeError):
        return 120

    for upper_bound in sorted(thresholds.keys(), key=int):
        if elo < int(upper_bound):
            return thresholds[upper_bound]
    last_key = sorted(thresholds.keys(), key=int)[-1]
    return thresholds[last_key]

def is_effective_sacrifice(board, move, eval_before, next_evals_by_target, target_color,
                           target_elo=None, engine=None, depth=8):
    """
    有效弃子检测（选项B升级版）。

    判定条件（三项全部满足才算有效弃子）：
    1. 子力价值：送出去的子比吃回来的贵（moving_val > captured_val）
    2. 主动性：弃子后对方并非被迫吃子——
         有引擎时：引擎Top1走法不包含吃这颗子（对方可以不吃），才算主动弃子
         无引擎时：退化为检测落点是否是"空格"（走到空格=对方不一定能吃，兜底逻辑）
    3. 效果性：目标棋手视角，3步内的最低评估点不低于弃子前超过 drop_threshold cp
               （看3步内最低点，而非第3步终点）

    参数：
      next_evals_by_target : 目标棋手接下来3步的 move_score_cp 列表（可以不足3步）
      engine               : chess.engine.SimpleEngine 实例，可为 None（降级运行）
      depth                : 主动性检测引擎深度（默认浅搜8，速度快）
    """
    from_sq = move.from_square
    to_sq   = move.to_square

    moving_piece = board.piece_at(from_sq)
    if moving_piece is None or moving_piece.color != (chess.WHITE if target_color == 'White' else chess.BLACK):
        return False

    captured_piece = board.piece_at(to_sq)
    moving_val   = piece_value(moving_piece)
    captured_val = piece_value(captured_piece) if captured_piece else 0

    # ---- 条件1：子力价值 ----
    if moving_val <= captured_val:
        return False

    # ---- 条件2：主动性检测 ----
    # 走完这步之后，轮到对方走。检查对方是否被迫吃子。
    board_after = board.copy()
    board_after.push(move)

    opponent_must_capture = False  # 默认假设对方不是强制吃

    if engine is not None:
        try:
            info = engine.analyse(board_after, chess.engine.Limit(depth=depth), multipv=1)
            if info and len(info) > 0:
                pv = info[0].get('pv', [])
                if pv:
                    opp_top1 = pv[0]
                    # 对方引擎首选走法如果是吃掉我们刚走到 to_sq 的子，说明强制吃
                    if opp_top1.to_square == to_sq and board_after.piece_at(to_sq) is not None:
                        opponent_must_capture = True
        except Exception:
            pass  # 引擎失败则退化，不判定为强制
    else:
        # 无引擎兜底：如果落点有对方棋子（即是吃子走法），对方刚才已经被我们吃了，
        # 真正的"送吃"是落在空格让对方可以选择是否来吃。
        # 落点无子 = 送吃（对方可以不吃），落点有子 = 普通兑换，不算弃子
        if captured_piece is not None:
            return False  # 普通兑换，已在条件1过滤；此处二次保障

    if opponent_must_capture:
        return False   # 对方被迫吃，这不是主动弃子，是走漏或强制交换

    # ---- 条件3：效果性（3步内最低点，非第3步终点） ----
    drop_threshold = get_sacrifice_drop_threshold(target_elo)
    rel_before = get_relative_score(eval_before, target_color)
    if rel_before is None:
        return False

    if not next_evals_by_target:
        # 没有后续评估数据，无法判断效果，保守处理为False
        return False

    rel_next = [get_relative_score(e, target_color) for e in next_evals_by_target if e is not None]
    if not rel_next:
        return False

    min_eval_after = min(rel_next)   # 3步内最低点
    if (min_eval_after - rel_before) < -drop_threshold:
        return False

    return True

def extract_style_features(game_meta, step_dict, target_move_indices, engine=None, config=None):
    """
    提取风格特征——使用独立版完整逻辑（仅调整输入参数）
    已修复：steps_phase 现在记录走棋后的阶段，而非走棋前
    【2026修复】有效弃子升级为选项B（主动性检测+3步最低点+等级分阈值）
    【2026修复】兵风暴改为判断对方王所在翼，而非己方易位翼
    【2026修复】重子侵入阈值：过中线→深入第6/7线（rank>=5 或 rank<=2）
    """
    moves_uci = safe_eval_list(game_meta.get('moves_uci', []))
    if not moves_uci:
        return None

    _cfg = config if config is not None else load_config()
    _cfg_thresholds = _cfg.get("thresholds", {}) if isinstance(_cfg, dict) else {}

    target_color = game_meta['target_color']
    color = chess.WHITE if target_color == 'White' else chess.BLACK
    result = game_meta['result']
    target_elo = game_meta.get('target_elo', None)

    board = chess.Board()
    effective_sacrifices = 0
    pawn_storm_kingside = 0
    pawn_storm_queenside = 0
    pawn_storm_center = 0
    heavy_invasion_count = 0

    steps_eval = []      # 目标棋手走后的评估值（已转换为目标视角）
    steps_phase = []     # 对应阶段（走棋后）

    if not target_move_indices:
        return None

    board.reset()
    target_move_counter = 0
    opp_color = chess.BLACK if color == chess.WHITE else chess.WHITE

    for idx, uci_move in enumerate(moves_uci):
        try:
            move = chess.Move.from_uci(uci_move)
            current_turn = board.turn

            # 只处理目标棋手的走法
            if current_turn == color:
                move_number = target_move_indices[target_move_counter]
                eval_row = step_dict.get(move_number)
                if eval_row is None:
                    board.push(move)
                    target_move_counter += 1
                    continue

                eval_before = eval_row.get('top1_score_cp')
                eval_after = eval_row.get('move_score_cp')

                # ----- 有效弃子检测（选项B升级版） -----
                # 收集目标棋手接下来3步的 move_score_cp（最低点判断用）
                cur_pos = target_move_counter
                next_evals = []
                for offset in range(1, 4):
                    if cur_pos + offset < len(target_move_indices):
                        next_mn = target_move_indices[cur_pos + offset]
                        next_row = step_dict.get(next_mn)
                        if next_row is not None:
                            v = next_row.get('move_score_cp')
                            if v is not None:
                                next_evals.append(v)

                if eval_before is not None:
                    if is_effective_sacrifice(
                        board, move, eval_before, next_evals,
                        target_color, target_elo=target_elo,
                        engine=engine, depth=8
                    ):
                        effective_sacrifices += 1

                # ----- 兵风暴（修复：按对方王所在翼判断，而非己方易位翼） -----
                # 只在己方已经易位后才统计（己方王不在初始格）
                own_king_sq = board.king(color)
                opp_king_sq = board.king(opp_color)
                own_castled = False
                if color == chess.WHITE and own_king_sq in [chess.G1, chess.C1]:
                    own_castled = True
                elif color == chess.BLACK and own_king_sq in [chess.G8, chess.C8]:
                    own_castled = True

                if own_castled and opp_king_sq is not None:
                    moving_piece = board.piece_at(move.from_square)
                    if (moving_piece and moving_piece.piece_type == chess.PAWN
                            and moving_piece.color == color):
                        to_rank = chess.square_rank(move.to_square)
                        # 推进方向：白方兵向前（rank增大），黑方兵向前（rank减小）
                        is_advancing = (
                            (color == chess.WHITE and to_rank >= 4) or
                            (color == chess.BLACK and to_rank <= 3)
                        )
                        if is_advancing:
                            # 按对方王所在列判断攻击翼
                            opp_king_file = chess.square_file(opp_king_sq)
                            pawn_file = chess.square_file(move.to_square)
                            # 兵推进的列与对方王的列距离<=2，才算朝对方王方向冲
                            if abs(pawn_file - opp_king_file) <= 2:
                                # 再细分翼别（以对方王位置为准）
                                if opp_king_file in [0, 1, 2]:
                                    pawn_storm_queenside += 1
                                elif opp_king_file in [3, 4]:
                                    pawn_storm_center += 1
                                else:
                                    pawn_storm_kingside += 1

                # ----- 重子侵入（修复：深入第6/7线，而非过中线） -----
                # 白方：rank >= 5（第6横线，0-indexed）
                # 黑方：rank <= 2（第3横线，0-indexed）
                moving_piece = board.piece_at(move.from_square)
                if (moving_piece and moving_piece.color == color
                        and moving_piece.piece_type in [chess.ROOK, chess.QUEEN]):
                    to_rank = chess.square_rank(move.to_square)
                    if (color == chess.WHITE and to_rank >= 5) or (color == chess.BLACK and to_rank <= 2):
                        heavy_invasion_count += 1

                # 先推棋，再记录走棋后的阶段
                board.push(move)
                if eval_after is not None:
                    steps_eval.append(get_relative_score(eval_after, target_color))
                    steps_phase.append(get_board_phase(board))

                target_move_counter += 1
            else:
                board.push(move)

        except Exception:
            try:
                board.push(chess.Move.from_uci(uci_move))
            except Exception:
                break
            continue

    if not steps_eval:
        return None

    # ----- 赢棋路径判定（修复：从后往前找最早稳定点） -----
    # 定义：从某步起，目标棋手视角评估始终 >= STABLE_THRESHOLD，
    # 找到这个最早的步数，判断它处于中局还是残局。
    win_path = None
    is_win = (target_color == 'White' and result == '1-0') or (target_color == 'Black' and result == '0-1')
    if is_win and len(steps_eval) > 0:
        STABLE_THRESHOLD = _cfg_thresholds.get("win_path_stable_threshold", 150)   # 稳定优势线
        DECISIVE_THRESHOLD = _cfg_thresholds.get("win_path_threshold", 200) # 最低需要达到过的优势

        # 必须至少达到过 DECISIVE_THRESHOLD，否则属于对手走输的比赛，不算赢棋路径
        if max(steps_eval) >= DECISIVE_THRESHOLD:
            # 从后往前扫描：找到最早的一步，使得从该步到终局评估始终 >= STABLE_THRESHOLD
            earliest_stable = None
            for i in range(len(steps_eval) - 1, -1, -1):
                if steps_eval[i] >= STABLE_THRESHOLD:
                    earliest_stable = i
                else:
                    break   # 一旦找到不满足的步，停止向前扩展

            if earliest_stable is not None and earliest_stable < len(steps_phase):
                win_path = steps_phase[earliest_stable]
            else:
                # 退化兜底：看最后5步的阶段众数
                if len(steps_phase) > 5:
                    last_phases = steps_phase[-5:]
                    win_path = 'endgame' if last_phases.count('endgame') >= 3 else 'midgame'
                elif steps_phase:
                    win_path = steps_phase[-1]
                else:
                    win_path = 'unknown'
        else:
            win_path = 'unknown'  # 未形成决定性优势，无法判断赢棋路径

    result_dict = {
        'game_id': game_meta['game_id'],
        'target_color': target_color,
        'result': result,
        'win_path': win_path,
        'effective_sacrifices': effective_sacrifices,
        'pawn_storm_kingside': pawn_storm_kingside,
        'pawn_storm_queenside': pawn_storm_queenside,
        'pawn_storm_center': pawn_storm_center,
        'total_pawn_storm': pawn_storm_kingside + pawn_storm_queenside + pawn_storm_center,
        'heavy_invasion_count': heavy_invasion_count,
        'total_moves': len(steps_eval),
    }
    return result_dict
