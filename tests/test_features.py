"""
tests.test_features
================================================================================
针对特征提取函数的回归测试。这些测试用例（合成的一局西班牙开局对局）
已经过与原始 total.py 的逐字段比对，确保重构前后输出 100% 一致
（见 README.md「零功能回归验证」一节）。
"""

import chess
import pandas as pd
import pytest

from chess_analyzer.features.opening import classify_opening
from chess_analyzer.features.style import extract_style_features, get_sacrifice_drop_threshold
from chess_analyzer.features.phase import aggregate_phase_ability


def _build_synthetic_game():
    moves_san = ["e4", "e5", "Nf3", "Nc6", "Bb5", "a6", "Ba4", "Nf6",
                 "O-O", "Be7", "Re1", "b5", "Bb3", "d6"]
    board = chess.Board()
    moves_uci = []
    for san in moves_san:
        mv = board.parse_san(san)
        moves_uci.append(mv.uci())
        board.push(mv)

    meta = {
        "game_id": "test1",
        "target_color": "White",
        "result": "1-0",
        "moves_uci": moves_uci,
        "target_elo": 2200,
    }

    step_dict = {}
    target_indices = []
    b2 = chess.Board()
    for i, uci in enumerate(moves_uci, start=1):
        mv = chess.Move.from_uci(uci)
        if b2.turn == chess.WHITE:
            target_indices.append(i)
            step_dict[i] = {
                "top1_score_cp": 20, "move_score_cp": 15,
                "loss_cp": 5, "rank_in_multipv": 1,
                "excellent_threshold_cp": 10,
            }
        b2.push(mv)
    return meta, step_dict, target_indices


def test_classify_opening_ruy_lopez():
    meta, _, _ = _build_synthetic_game()
    result = classify_opening(meta)
    assert result["label"] == "西班牙开局"
    assert result["variation"] == "spanish_ruy_lopez"
    assert result["wing"] == "kingside"
    assert result["is_win"] == 1


def test_extract_style_features_matches_expected_snapshot():
    meta, step_dict, target_indices = _build_synthetic_game()
    style = extract_style_features(meta, step_dict, target_indices)
    assert style["game_id"] == "test1"
    assert style["total_moves"] == 7
    assert style["pawn_storm_kingside"] == 0
    assert style["heavy_invasion_count"] == 0


def test_aggregate_phase_ability_matches_expected_snapshot():
    meta, step_dict, _ = _build_synthetic_game()
    step_df = pd.DataFrame([{**v, "move_number": k} for k, v in step_dict.items()])
    phase = aggregate_phase_ability(meta, step_df)
    assert phase["total_moves"] == 14
    assert phase["opening_acpl"] == pytest.approx(5.0)
    assert phase["mid_peak_accuracy"] == pytest.approx(1.0)


@pytest.mark.parametrize("elo,expected", [
    (None, 120),
    (1500, 180),
    (1700, 150),
    (1900, 130),
    (2100, 110),
    (2300, 90),
    (2500, 70),
])
def test_get_sacrifice_drop_threshold(elo, expected):
    assert get_sacrifice_drop_threshold(elo) == expected
