"""
core.board_utils
================================================================================
棋盘通用工具函数：安全解析、子力价值、阶段判定。
从原 total.py 迁移（逻辑与输出 100% 保持不变）。
"""

import ast

import chess


def safe_eval_list(x):
    """安全地将数据转换为 Python 列表"""
    if isinstance(x, list):
        return x
    if hasattr(x, 'tolist'):
        return x.tolist()
    if isinstance(x, str):
        try:
            result = ast.literal_eval(x)
            if isinstance(result, list):
                return result
        except:
            pass
    return []

def piece_value(piece):
    """返回棋子价值"""
    if piece is None:
        return 0
    if piece.piece_type == chess.PAWN:
        return 1
    elif piece.piece_type in [chess.KNIGHT, chess.BISHOP]:
        return 3
    elif piece.piece_type == chess.ROOK:
        return 5
    elif piece.piece_type == chess.QUEEN:
        return 9
    return 0

def get_board_phase(board):
    """判定当前局面是中局还是残局"""
    total_value, non_pawn_cnt, queen_exists = 0, 0, False
    for sq in chess.SQUARES:
        piece = board.piece_at(sq)
        if piece and piece.piece_type != chess.KING:
            val = piece_value(piece)
            total_value += val
            if val > 0:
                non_pawn_cnt += 1
            if piece.piece_type == chess.QUEEN:
                queen_exists = True
    if total_value <= 24:
        return 'endgame'
    if not queen_exists and non_pawn_cnt <= 4:
        return 'endgame'
    return 'midgame'
