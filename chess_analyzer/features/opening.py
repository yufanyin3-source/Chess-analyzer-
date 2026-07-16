"""
features.opening
================================================================================
开局分类：白方/黑方开局细分与标签映射。
从原 total.py 迁移（逻辑与输出 100% 保持不变，含全部 ECO 分支判定）。
"""

import chess

from chess_analyzer.core.board_utils import safe_eval_list
from chess_analyzer.core.config import load_config


def get_board_after_n_moves(moves_uci, n=12):
    board = chess.Board()
    max_moves = min(n, len(moves_uci))
    try:
        for i in range(max_moves):
            board.push(chess.Move.from_uci(moves_uci[i]))
        return board
    except Exception:
        return None

def get_color_moves(moves_uci, n=12):
    """返回 (white_moves, black_moves) 前 n 步中的 Move 对象列表"""
    board = get_board_after_n_moves(moves_uci, n)
    if board is None:
        return [], []
    white_moves = []
    black_moves = []
    for idx, move in enumerate(board.move_stack):
        if idx % 2 == 0:
            white_moves.append(move)
        else:
            black_moves.append(move)
    return white_moves, black_moves

def classify_white_opening_detail(moves_uci, max_moves=12):
    """
    返回：(wing, opening_variation, variation_label)
    """
    if len(moves_uci) < 4:
        return 'unknown', 'unknown', '未知'
    white_moves, black_moves = get_color_moves(moves_uci, max_moves)
    if len(white_moves) < 2 or len(black_moves) < 1:
        return 'unknown', 'unknown', '未知'

    w1 = white_moves[0]
    b1 = black_moves[0]

    is_e4 = (w1.from_square == chess.E2 and w1.to_square == chess.E4)
    is_d4 = (w1.from_square == chess.D2 and w1.to_square == chess.D4)
    is_c4 = (w1.from_square == chess.C2 and w1.to_square == chess.C4)
    is_Nf3 = (w1.from_square == chess.G1 and w1.to_square == chess.F3)

    # e4 系统
    if is_e4:
        if b1.from_square == chess.E7 and b1.to_square == chess.E5:
            has_spanish = any(
                (m.from_square == chess.F1 and m.to_square == chess.B5)
                for m in white_moves
            )
            if has_spanish:
                return 'kingside', 'spanish_ruy_lopez', '西班牙开局'
            has_italian = any(
                (m.from_square == chess.F1 and m.to_square == chess.C4)
                for m in white_moves
            )
            if has_italian:
                return 'kingside', 'italian_game', '意大利开局'
            has_scotch = any(m.from_square == chess.D2 and m.to_square == chess.D4 for m in white_moves)
            if has_scotch:
                return 'kingside', 'scotch_game', '苏格兰开局'
            return 'kingside', 'e4_e5_other', '其他e4-e5开局'

        if b1.from_square == chess.C7 and b1.to_square == chess.C5:
            has_Nf3 = any(m.from_square == chess.G1 and m.to_square == chess.F3 for m in white_moves)
            has_d4 = any(m.from_square == chess.D2 and m.to_square == chess.D4 for m in white_moves)
            has_Bb5 = any((m.from_square == chess.F1 and m.to_square == chess.B5) for m in white_moves)
            has_g3 = any(m.from_square == chess.G2 and m.to_square == chess.G3 for m in white_moves)

            if has_Nf3 and has_d4:
                return 'kingside', 'sicilian_open', '西西里(开放式)'
            elif has_Nf3 and has_Bb5:
                return 'kingside', 'sicilian_rossolimo', '西西里罗索里莫'
            elif has_g3:
                return 'kingside', 'sicilian_closed', '西西里封闭式'
            else:
                return 'kingside', 'sicilian_other', '西西里(其他)'

        if b1.from_square == chess.E7 and b1.to_square == chess.E6:
            return 'kingside', 'french_defense', '法兰西防御'
        if b1.from_square == chess.C7 and b1.to_square == chess.C6:
            return 'kingside', 'caro_kann', '卡罗康防御'
        if b1.from_square == chess.D7 and b1.to_square == chess.D6:
            return 'kingside', 'pirc_defense', '皮尔茨防御'
        if b1.from_square == chess.G8 and b1.to_square == chess.F6:
            return 'kingside', 'alekhine', '阿廖欣防御'
        return 'kingside', 'e4_other', '其他e4开局'

    # d4 系统
    if is_d4:
        if len(black_moves) < 1:
            return 'queenside', 'd4_unknown', 'd4体系'
        b1 = black_moves[0]
        if b1.from_square == chess.D7 and b1.to_square == chess.D5:
            has_c4 = any(m.from_square == chess.C2 and m.to_square == chess.C4 for m in white_moves)
            if has_c4:
                if len(black_moves) > 1:
                    b2 = black_moves[1]
                    if b2.from_square == chess.C7 and b2.to_square == chess.C6:
                        return 'queenside', 'slav_defense', '斯拉夫防御'
                    if b2.from_square == chess.E7 and b2.to_square == chess.E6:
                        return 'queenside', 'qgd_orthodox', '后翼弃兵正统'
                return 'queenside', 'queens_gambit', '后翼弃兵(其他)'
            return 'queenside', 'd4_d5_other', 'd4-d5其他'

        if b1.from_square == chess.G8 and b1.to_square == chess.F6:
            has_c4 = any(m.from_square == chess.C2 and m.to_square == chess.C4 for m in white_moves)
            if has_c4:
                if len(black_moves) > 1:
                    b2 = black_moves[1]
                    if b2.from_square == chess.E7 and b2.to_square == chess.E6:
                        if len(black_moves) > 2:
                            b3 = black_moves[2]
                            if b3.from_square == chess.F8 and b3.to_square == chess.B4:
                                return 'queenside', 'nimzo_indian', '尼姆佐印度'
                        return 'queenside', 'queens_indian', '后印度防御'
                    if b2.from_square == chess.G7 and b2.to_square == chess.G6:
                        if len(black_moves) > 2:
                            b3 = black_moves[2]
                            if b3.from_square == chess.D7 and b3.to_square == chess.D5:
                                return 'queenside', 'grunfeld', '格林菲尔德'
                        return 'queenside', 'king_indian', '古印度防御'
                    if b2.from_square == chess.C7 and b2.to_square == chess.C5:
                        return 'queenside', 'benoni', '别诺尼防御'
                return 'queenside', 'indian_other', '其他印度防御'
            return 'queenside', 'd4_Nf6_other', '对d4 Nf6其他'

        if b1.from_square == chess.C7 and b1.to_square == chess.C5:
            return 'queenside', 'benoni', '别诺尼防御'
        return 'queenside', 'd4_other', '其他d4开局'

    # c4 英国式
    if is_c4:
        has_d4 = any(m.from_square == chess.D2 and m.to_square == chess.D4 for m in white_moves)
        if has_d4:
            return 'queenside', 'english_to_d4', '英国式转d4'
        return 'queenside', 'english_opening', '英国式'

    # Nf3 列蒂
    if is_Nf3:
        has_g3 = any(m.from_square == chess.G2 and m.to_square == chess.G3 for m in white_moves)
        has_d4 = any(m.from_square == chess.D2 and m.to_square == chess.D4 for m in white_moves)
        has_c4 = any(m.from_square == chess.C2 and m.to_square == chess.C4 for m in white_moves)
        if has_g3:
            return 'kingside', 'reti_kingside', '列蒂王翼'
        elif has_d4 or has_c4:
            return 'queenside', 'reti_queenside', '列蒂后翼'
        return 'queenside', 'reti_other', '列蒂'

    return 'other', 'uncommon_opening', '冷门开局'

def classify_black_opening_detail(moves_uci, max_moves=12):
    """
    返回：(white_first_move, black_defense, defense_label)
    """
    if len(moves_uci) < 4:
        return 'unknown', 'unknown', '未知'
    white_moves, black_moves = get_color_moves(moves_uci, max_moves)
    if len(white_moves) < 1 or len(black_moves) < 1:
        return 'unknown', 'unknown', '未知'

    w1 = white_moves[0]
    b1 = black_moves[0]

    if w1.from_square == chess.E2 and w1.to_square == chess.E4:
        white_first = 'e4'
    elif w1.from_square == chess.D2 and w1.to_square == chess.D4:
        white_first = 'd4'
    elif w1.from_square == chess.C2 and w1.to_square == chess.C4:
        white_first = 'c4'
    elif w1.from_square == chess.G1 and w1.to_square == chess.F3:
        white_first = 'Nf3'
    else:
        white_first = 'other'

    if white_first == 'e4':
        if b1.from_square == chess.E7 and b1.to_square == chess.E5:
            return 'e4', 'e5_response', '对e4-e5'
        if b1.from_square == chess.C7 and b1.to_square == chess.C5:
            if len(black_moves) > 1:
                b2 = black_moves[1]
                if b2.from_square == chess.D7 and b2.to_square == chess.D6:
                    if len(black_moves) > 2:
                        b3 = black_moves[2]
                        if b3.from_square == chess.G7 and b3.to_square == chess.G6:
                            return 'e4', 'sicilian_dragon', '西西里龙式'
                    return 'e4', 'sicilian_najdorf', '西西里纳道尔夫'
                if b2.from_square == chess.B8 and b2.to_square == chess.C6:
                    if len(black_moves) > 2:
                        b3 = black_moves[2]
                        if b3.from_square == chess.D7 and b3.to_square == chess.D6:
                            return 'e4', 'sicilian_classical', '西西里古典'
                        if b3.from_square == chess.E7 and b3.to_square == chess.E5:
                            return 'e4', 'sicilian_sveshnikov', '西西里斯韦什尼科夫'
                    return 'e4', 'sicilian_other', '西西里其他'
                if b2.from_square == chess.E7 and b2.to_square == chess.E6:
                    return 'e4', 'sicilian_scheveningen', '西西里舍维宁根'
            return 'e4', 'sicilian_other', '西西里防御'
        if b1.from_square == chess.E7 and b1.to_square == chess.E6:
            return 'e4', 'french_defense', '法兰西防御'
        if b1.from_square == chess.C7 and b1.to_square == chess.C6:
            return 'e4', 'caro_kann', '卡罗康防御'
        if b1.from_square == chess.D7 and b1.to_square == chess.D6:
            return 'e4', 'pirc_defense', '皮尔茨防御'
        if b1.from_square == chess.G8 and b1.to_square == chess.F6:
            return 'e4', 'alekhine', '阿廖欣防御'
        return 'e4', 'e4_other', '对e4其他'

    if white_first == 'd4':
        if b1.from_square == chess.D7 and b1.to_square == chess.D5:
            if len(black_moves) > 1:
                b2 = black_moves[1]
                if b2.from_square == chess.C7 and b2.to_square == chess.C6:
                    return 'd4', 'slav_defense', '斯拉夫防御'
                if b2.from_square == chess.E7 and b2.to_square == chess.E6:
                    return 'd4', 'qgd_orthodox', '后翼弃兵正统'
            return 'd4', 'd4_d5_other', '对d4-d5其他'
        if b1.from_square == chess.G8 and b1.to_square == chess.F6:
            if len(black_moves) > 1:
                b2 = black_moves[1]
                if b2.from_square == chess.E7 and b2.to_square == chess.E6:
                    if len(black_moves) > 2:
                        b3 = black_moves[2]
                        if b3.from_square == chess.F8 and b3.to_square == chess.B4:
                            return 'd4', 'nimzo_indian', '尼姆佐印度'
                    return 'd4', 'queens_indian', '后印度防御'
                if b2.from_square == chess.G7 and b2.to_square == chess.G6:
                    if len(black_moves) > 2:
                        b3 = black_moves[2]
                        if b3.from_square == chess.D7 and b3.to_square == chess.D5:
                            return 'd4', 'grunfeld', '格林菲尔德'
                    return 'd4', 'king_indian', '古印度防御'
                if b2.from_square == chess.C7 and b2.to_square == chess.C5:
                    return 'd4', 'benoni', '别诺尼防御'
            return 'd4', 'indian_other', '其他印度防御'
        if b1.from_square == chess.C7 and b1.to_square == chess.C5:
            return 'd4', 'benoni', '别诺尼防御'
        return 'd4', 'd4_other', '对d4其他'

    if white_first in ['c4', 'Nf3']:
        if b1.from_square == chess.D7 and b1.to_square == chess.D5:
            return white_first, 'flank_d5', '对侧翼走d5'
        if b1.from_square == chess.G8 and b1.to_square == chess.F6:
            return white_first, 'flank_Nf6', '对侧翼走Nf6'
        return white_first, 'flank_other', '对侧翼其他'

    return 'other', 'other_response', '其他'

def classify_opening(game_meta, config=None):
    """开局分类主函数"""
    moves_uci = safe_eval_list(game_meta.get('moves_uci', []))
    if len(moves_uci) < 4:
        return None

    _cfg = config if config is not None else load_config()
    CONFIG = _cfg.flat("thresholds") if hasattr(_cfg, "flat") else {}
    if not CONFIG:
        CONFIG = {"OPENING_MOVES": 12}

    target_color = game_meta['target_color']
    result = game_meta['result']
    game_id = game_meta['game_id']

    if target_color == 'White':
        wing, variation, label = classify_white_opening_detail(moves_uci, CONFIG["OPENING_MOVES"])
        return {
            'game_id': game_id,
            'target_color': 'White',
            'result': result,
            'wing': wing,
            'variation': variation,
            'label': label,
            'is_win': 1 if result == '1-0' else 0,
            'is_draw': 1 if result == '1/2-1/2' else 0,
            'is_loss': 1 if result == '0-1' else 0,
        }
    else:
        white_first, variation, label = classify_black_opening_detail(moves_uci, CONFIG["OPENING_MOVES"])
        return {
            'game_id': game_id,
            'target_color': 'Black',
            'result': result,
            'white_first': white_first,
            'variation': variation,
            'label': label,
            'is_win': 1 if result == '0-1' else 0,
            'is_draw': 1 if result == '1/2-1/2' else 0,
            'is_loss': 1 if result == '1-0' else 0,
        }
