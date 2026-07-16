"""
tests.test_core
================================================================================
针对纯函数（core 模块）的单元测试。
"""

import chess
import pytest

from chess_analyzer.core.board_utils import safe_eval_list, piece_value, get_board_phase
from chess_analyzer.core.color_utils import get_relative_score, assign_tier


def test_safe_eval_list_handles_list():
    assert safe_eval_list([1, 2, 3]) == [1, 2, 3]


def test_safe_eval_list_handles_string_repr():
    assert safe_eval_list("['e2e4', 'e7e5']") == ['e2e4', 'e7e5']


def test_safe_eval_list_handles_garbage():
    assert safe_eval_list(None) == []
    assert safe_eval_list(42) == []
    assert safe_eval_list("not a list") == []


def test_piece_value():
    assert piece_value(chess.Piece(chess.PAWN, chess.WHITE)) == 1
    assert piece_value(chess.Piece(chess.KNIGHT, chess.WHITE)) == 3
    assert piece_value(chess.Piece(chess.BISHOP, chess.BLACK)) == 3
    assert piece_value(chess.Piece(chess.ROOK, chess.WHITE)) == 5
    assert piece_value(chess.Piece(chess.QUEEN, chess.BLACK)) == 9
    assert piece_value(chess.Piece(chess.KING, chess.WHITE)) == 0
    assert piece_value(None) == 0


def test_get_board_phase_start_position_is_midgame():
    board = chess.Board()
    assert get_board_phase(board) == 'midgame'


def test_get_board_phase_bare_kings_is_endgame():
    board = chess.Board(None)
    board.set_piece_at(chess.E1, chess.Piece(chess.KING, chess.WHITE))
    board.set_piece_at(chess.E8, chess.Piece(chess.KING, chess.BLACK))
    assert get_board_phase(board) == 'endgame'


@pytest.mark.parametrize("score,color,expected", [
    (50, 'White', 50),
    (50, 'Black', -50),
    (None, 'Black', None),
    (-30, 'White', -30),
])
def test_get_relative_score(score, color, expected):
    assert get_relative_score(score, color) == expected


@pytest.mark.parametrize("rating,expected_tier", [
    (1300, None),
    (1500, "Tier0_1400_1599"),
    (1750, "Tier1_1600_1799"),
    (1950, "Tier2_1800_1999"),
    (2150, "Tier3_2000_2199"),
    (2350, "Tier4_2200_2399"),
    (2550, "Tier5_2400_2599"),
    (2700, "Tier6_2600plus"),
    (None, None),
])
def test_assign_tier(rating, expected_tier):
    assert assign_tier(rating) == expected_tier
