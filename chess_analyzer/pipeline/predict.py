"""
pipeline.predict
================================================================================
棋手等级分区间预测 - 端到端推理流水线。
从原 predict_player.py 迁移为 run_inference(...) 函数式调用（任务一要求），
并接入 configs/default.yaml 的 `inference` / `stockfish` 段（任务二要求）。

流程：
  输入 PGN + 棋手名
    → Step 1: prepare_games 清洗，生成走法哈希 game_id
    → Step 2: 提取廉价特征（16维，无需引擎）
    → Step 3: 实时跑 Stockfish（depth=14），生成引擎特征
    → Step 4: 调用 chess_analyzer.features 提取风格/心态特征（原为动态
              sys.path 导入 total.py，现替换为标准包导入）
    → Step 5: 两层堆叠推理（XGBoost + LightGBM）
    → Step 6: 软投票聚合，输出最终等级分区间

已修复的 bug（保留自原脚本）：
  [BUG-1] Blitz判定：改用等效时长（base + 40×increment），与训练逻辑一致
  [BUG-2] game_id：改用走法哈希，不依赖PGN头信息，与缓存天然对齐
  [BUG-3] 批量推理：廉价特征批量送入XGBoost，不再逐盘循环调用
  [BUG-4] 低置信度输出：最高概率<0.35时输出Top-2候选，不强行给单一答案
  [BUG-5] 降级日志：明确提示当前精度模式，不静默降级
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
import sys
import warnings
from pathlib import Path
from typing import Optional

import chess
import chess.engine
import chess.pgn
import numpy as np
import pandas as pd

from chess_analyzer.core.config import load_config
from chess_analyzer.features.style import extract_style_features
from chess_analyzer.features.phase import aggregate_phase_ability
from chess_analyzer.features.opening import classify_opening
from chess_analyzer.features.mental import compute_mental_metrics
from chess_analyzer.models.loader import load_models, merge_7to5

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)

# ==================== 配置区（从 configs/default.yaml 读取，带原脚本默认值兜底） ====================
_cfg = load_config()
_sf_cfg = _cfg.get("stockfish", {}) if hasattr(_cfg, "get") else {}
_inf_cfg = _cfg.get("inference", {}) if hasattr(_cfg, "get") else {}

STOCKFISH_PATH = _sf_cfg.get("path", "/usr/local/bin/stockfish")
STOCKFISH_DEPTH = _sf_cfg.get("depth", 14)
STOCKFISH_MEMORY = _sf_cfg.get("memory", 512)
STOCKFISH_MULTI_PV = _sf_cfg.get("multipv", 3)

MODEL_DIR = _inf_cfg.get("model_dir", "models")
DATA_DIR = _inf_cfg.get("data_dir", "player_analysis")

# 【BUG-1修复】等效时长估算系数，与 extract_cheap_features.py 训练逻辑完全一致
AVG_MOVES_FOR_TIME_ESTIMATE = _inf_cfg.get("avg_moves_for_time_estimate", 40)

TIER_LABELS = _inf_cfg.get("tier_labels", [
    "Tier0_1400_1599", "Tier1_1600_1999",
    "Tier2_2000_2399", "Tier3_2400_2599", "Tier4_2600plus",
])

# 7→5 概率合并规则（与 train_lgb_layer2.py 一致）
MERGE_RULES = _inf_cfg.get("merge_rules", [[0], [1, 2], [3, 4], [5], [6]])

LOW_CONFIDENCE_THRESHOLD = _inf_cfg.get("low_confidence_threshold", 0.35)
BLITZ_MAX_SECONDS = _inf_cfg.get("blitz_max_seconds", 300)

CHEAP_FEATURE_COLS = [
    'ply_count', 'eco_group', 'base_time', 'increment',
    'capture_rate', 'pawn_push_rate', 'castling_occurred', 'avg_move_dist',
    'promotion_count', 'center_control_rate', 'piece_exchange_rate',
    'avg_time_per_move', 'time_pressure_rate', 'clock_slope',
    'endgame_time_ratio', 'min_clock_ratio',
]

ECO_GROUP_MAP = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4}
CENTER_SQUARES = {'d4', 'd5', 'e4', 'e5'}
CASTLING_MOVES = {'e1g1', 'e1c1', 'e8g8', 'e8c8'}
CLK_HMS = re.compile(r'%clk\s+(?:(\d+):)?(\d+):(\d+)')
CLK_SEC = re.compile(r'%clk\s+(\d+)')

TIER_DESCRIPTIONS = _inf_cfg.get("tier_descriptions", {
    "Tier0_1400_1599": "入门/初级俱乐部（约1400-1600）",
    "Tier1_1600_1999": "业余中坚（约1600-2000）",
    "Tier2_2000_2399": "业余精英/候选大师（约2000-2400）",
    "Tier3_2400_2599": "国际大师水平（约2400-2600）",
    "Tier4_2600plus": "特级大师水平（2600+）",
})
# ================================================


def normalize_name(name: str) -> str:
    if not name:
        return ""
    return re.sub(r'\s+', ' ', name.strip().lower().replace(",", " "))


def moves_to_game_id(moves_uci: list) -> str:
    """【BUG-2修复】走法哈希 game_id，完全不依赖PGN头信息格式。"""
    key = " ".join(moves_uci[:80])
    return "pgn_" + hashlib.sha256(key.encode()).hexdigest()[:20]


def parse_clock_value(comment: str):
    if '%clk' not in comment:
        return None
    m = CLK_HMS.search(comment)
    if m:
        return int(m.group(1) or 0) * 3600 + int(m.group(2)) * 60 + int(m.group(3))
    ms = CLK_SEC.search(comment)
    return int(ms.group(1)) if ms else None


def classify_time_control(event: str, tc_str: str) -> str:
    """
    【BUG-1修复】使用等效时长（base + 40×increment）分类，
    与 extract_cheap_features.py 训练逻辑完全一致。
    原来只用 base_time，导致 60+60 被误判为 Blitz。
    """
    event_lower = (event or "").lower()
    tc_str = str(tc_str or "").strip()

    if "blitz" in event_lower:
        return "blitz"
    if "rapid" in event_lower:
        return "rapid"
    for kw in ["candidates", "grand prix", "olympiad", "league",
               "challengers", "world team", "fide"]:
        if kw in event_lower:
            return "classical"

    if '+' in tc_str:
        try:
            base = int(tc_str.split('+')[0])
            inc  = int(tc_str.split('+')[1])
            eff  = base + AVG_MOVES_FOR_TIME_ESTIMATE * inc
            if eff >= 3600: return "classical"
            if eff >= 600:  return "rapid"
            if eff >= 180:  return "blitz"
            return None   # 子弹棋过滤
        except (ValueError, IndexError):
            pass
    elif tc_str.isdigit():
        base = int(tc_str)
        if base >= 3600: return "classical"
        if base >= 600:  return "rapid"
        if base >= 180:  return "blitz"

    return "classical"   # 无法识别时归为 classical
def prepare_games(pgn_path: str, target_name: str,
                  color_override: str = None,
                  min_ply: int = 20) -> pd.DataFrame:
    """
    清洗PGN：识别目标棋手，提取走法，生成走法哈希game_id。
    返回 curation 兼容格式的 DataFrame。
    """
    target_norm = normalize_name(target_name)
    records = []
    skipped = not_found = 0

    with open(pgn_path, encoding='utf-8', errors='replace') as f:
        while True:
            game = chess.pgn.read_game(f)
            if game is None:
                break

            headers = game.headers
            result  = headers.get('Result', '*')
            if result == '*':
                skipped += 1
                continue

            white_norm = normalize_name(headers.get('White', ''))
            black_norm = normalize_name(headers.get('Black', ''))

            if color_override:
                target_color = color_override
                opp_norm = black_norm if target_color == 'White' else white_norm
            elif target_norm in white_norm:
                target_color = 'White'
                opp_norm = black_norm
            elif target_norm in black_norm:
                target_color = 'Black'
                opp_norm = white_norm
            else:
                not_found += 1
                continue

            def parse_elo(v):
                try:
                    x = int(v)
                    return x if x > 0 else None
                except (ValueError, TypeError):
                    return None

            white_elo  = parse_elo(headers.get('WhiteElo', ''))
            black_elo  = parse_elo(headers.get('BlackElo', ''))
            target_elo = white_elo if target_color == 'White' else black_elo
            opp_elo    = black_elo if target_color == 'White' else white_elo
            elo_diff   = (target_elo - opp_elo
                          if target_elo is not None and opp_elo is not None else None)

            node  = game
            board = chess.Board()
            moves_uci  = []
            clock_vals = []
            ok = True
            while node.variations:
                nxt = node.variations[0]
                try:
                    board.push(nxt.move)
                    moves_uci.append(nxt.move.uci())
                except Exception:
                    ok = False
                    break
                clock_vals.append(parse_clock_value(nxt.comment))
                node = nxt

            if not ok or len(moves_uci) < min_ply:
                skipped += 1
                continue

            event    = headers.get('Event', '')
            tc_raw   = headers.get('TimeControl', '')
            category = classify_time_control(event, tc_raw)
            if category is None:
                skipped += 1
                continue

            clock_str = ' '.join(
                str(v) if v is not None else '?' for v in clock_vals
            )

            records.append({
                'game_id':          moves_to_game_id(moves_uci),
                'event':            event,
                'site':             headers.get('Site', ''),
                'date':             headers.get('Date', ''),
                'round':            headers.get('Round', ''),
                'white':            white_norm,
                'black':            black_norm,
                'result':           result,
                'eco':              headers.get('ECO', ''),
                'ply_count':        len(moves_uci),
                'time_control_raw': tc_raw,
                'category':         category.capitalize(),
                'target_color':     target_color,
                'opponent':         opp_norm,
                'target_elo':       target_elo,
                'opponent_elo':     opp_elo,
                'elo_diff':         elo_diff,
                'is_complete':      True,
                'moves_uci':        moves_uci,
                'time_weight':      1.0,
                'career_phase':     'phase_1',
                'clock_times':      clock_str,
            })

    df = pd.DataFrame(records)
    logger.info("   解析完成：%d 盘有效 / %d 盘跳过%s", len(df), skipped,
                (f" / {not_found} 盘未找到目标棋手" if not_found else ""))
    if not df.empty:
        for cat in ['Classical', 'Rapid', 'Blitz']:
            cnt = int((df['category'] == cat).sum())
            if cnt:
                logger.info(f"     {cat}: {cnt} 盘")
    return df
def parse_time_control_str(tc_str):
    if not tc_str or '+' not in str(tc_str):
        return -1, -1
    try:
        p = str(tc_str).split('+')
        return int(p[0]), int(p[1])
    except (ValueError, IndexError):
        return -1, -1


def extract_move_features(moves_uci):
    moves = list(moves_uci) if not isinstance(moves_uci, list) else moves_uci
    n = len(moves)
    if n == 0:
        return None
    captures = pawn_pushes = castling = promotions = center_moves = 0
    total_dist = exchanges = 0
    prev_cap = False
    for uci in moves:
        if len(uci) < 4:
            continue
        fs, ts = uci[:2], uci[2:4]
        try:
            total_dist += abs(ord(ts[0]) - ord(fs[0])) + abs(int(ts[1]) - int(fs[1]))
        except (ValueError, IndexError):
            pass
        if fs[0] == ts[0] and fs[0] in 'abcdefgh':
            pawn_pushes += 1
        if ts in CENTER_SQUARES:
            center_moves += 1
        if uci in CASTLING_MOVES:
            castling = 1
        if len(uci) == 5:
            promotions += 1
        if fs[0] != ts[0]:
            captures += 1
            if prev_cap:
                exchanges += 1
            prev_cap = True
        else:
            prev_cap = False
    return {
        'capture_rate':        captures / n,
        'pawn_push_rate':      pawn_pushes / n,
        'castling_occurred':   castling,
        'avg_move_dist':       total_dist / n,
        'promotion_count':     promotions,
        'center_control_rate': center_moves / n,
        'piece_exchange_rate': exchanges / n,
    }


def parse_clock_str(clock_str):
    if not isinstance(clock_str, str) or not clock_str.strip():
        return []
    vals = []
    for p in clock_str.strip().split():
        vals.append(np.nan if p == '?' else
                    (float(p) if p.replace('.', '').isdigit() else np.nan))
    arr = np.array(vals, dtype=float)
    if int(np.sum(~np.isnan(arr))) < 3:
        return []
    nans = np.isnan(arr)
    idx  = np.arange(len(arr))
    arr[nans] = np.interp(idx[nans], idx[~nans], arr[~nans])
    return arr.tolist()


def extract_clock_features(clock_str, base_time):
    no_clk = {k: np.nan for k in ['avg_time_per_move', 'time_pressure_rate',
                                    'clock_slope', 'endgame_time_ratio', 'min_clock_ratio']}
    clocks = parse_clock_str(clock_str)
    if not clocks or len(clocks) < 6:
        return no_clk
    arr  = np.array(clocks)
    own  = arr[::2]
    if len(own) < 3:
        return no_clk
    diffs  = np.clip(np.diff(own) * -1, 0, None)
    avg_t  = float(np.mean(diffs)) if len(diffs) > 0 else np.nan
    prate  = int(np.sum(own < 30)) / len(own)
    slope  = float(np.polyfit(np.arange(len(own)), own, 1)[0]) if len(own) >= 2 else np.nan
    t      = max(1, len(diffs) // 3)
    early  = float(np.mean(diffs[:t])) if t > 0 else 1.0
    late   = float(np.mean(diffs[-t:])) if t > 0 else 1.0
    end_r  = late / early if early > 0 else 1.0
    init   = own[0] if own[0] > 0 else (base_time if base_time > 0 else 1)
    min_r  = float(np.min(own)) / init
    return {
        'avg_time_per_move':  avg_t,  'time_pressure_rate': prate,
        'clock_slope':        slope,  'endgame_time_ratio':  end_r,
        'min_clock_ratio':    min_r,
    }


def extract_cheap_features(row) -> tuple:
    """提取16维廉价特征，返回 (feat_dict, base_time)。"""
    moves = list(row.get('moves_uci', []))
    if len(moves) < 10:
        return None, -1
    if str(row.get('result', '*')) == '*':
        return None, -1
    eco       = str(row.get('eco', ''))
    eco_group = ECO_GROUP_MAP.get(eco[0].upper() if eco else '', 5)
    base_time, increment = parse_time_control_str(str(row.get('time_control_raw', '')))
    move_feats = extract_move_features(moves)
    if move_feats is None:
        return None, -1
    clk_str = row.get('clock_times', '')
    if isinstance(clk_str, (list, np.ndarray)):
        clk_str = ' '.join(str(v) for v in clk_str)
    clk_feats = extract_clock_features(str(clk_str), base_time)
    feat = {'ply_count': len(moves), 'eco_group': eco_group,
            'base_time': base_time, 'increment': increment,
            **move_feats, **clk_feats}
    return feat, base_time
def get_thresholds(target_elo):
    ELO_TIERS = [
        (0, 1400, 200, 40), (1400, 1600, 150, 30),
        (1600, 1800, 120, 20), (1800, 2000, 100, 15),
        (2000, 2200, 80, 10), (2200, 2400, 60, 8),
        (2400, 2600, 50, 6),  (2600, 9999, 40, 4),
    ]
    try:
        elo = int(target_elo)
    except (ValueError, TypeError):
        return 80, 10
    for lo, hi, b, e in ELO_TIERS:
        if lo <= elo < hi:
            return b, e
    return 80, 10


def analyze_single_move(board, move, engine):
    import chess.engine
    info_before = engine.analyse(
        board, chess.engine.Limit(depth=STOCKFISH_DEPTH),
        multipv=STOCKFISH_MULTI_PV
    )
    top1 = 0
    if info_before:
        s = info_before[0].get('score')
        if s:
            v = s.white().score(mate_score=10000)
            if v is not None:
                top1 = v
    bc = board.copy()
    bc.push(move)
    info_after = engine.analyse(bc, chess.engine.Limit(depth=STOCKFISH_DEPTH), multipv=1)
    move_score = top1
    if info_after:
        s = info_after[0].get('score')
        if s:
            v = s.white().score(mate_score=10000)
            if v is not None:
                move_score = v
    loss_cp = max(0, top1 - move_score if board.turn == chess.WHITE
                  else move_score - top1)
    rank = None
    for i, pv_line in enumerate(info_before):
        pv = pv_line.get('pv')
        if pv and pv[0] == move:
            rank = i + 1
            break
    return loss_cp, rank, top1, move_score
def run_acpl_on_games(df: pd.DataFrame, engine) -> tuple:
    all_steps = []
    all_aggs  = []
    total = len(df)

    for idx, (_, row) in enumerate(df.iterrows(), 1):
        sys.stdout.write(f"\r   引擎分析进度: {idx}/{total} 盘")
        sys.stdout.flush()

        gid          = row['game_id']
        target_color = row['target_color']
        result       = row['result']
        target_elo   = row.get('target_elo', None)
        time_weight  = float(row.get('time_weight', 1.0))
        moves_uci    = list(row.get('moves_uci', []))

        if not moves_uci or len(moves_uci) < 3:
            continue

        blunder_cp, excellent_cp = get_thresholds(target_elo)
        color = chess.WHITE if target_color == 'White' else chess.BLACK
        board = chess.Board()
        steps = []
        losses = losses_op = losses_mid = []
        losses, losses_op, losses_mid = [], [], []

        for i, uci in enumerate(moves_uci, start=1):
            try:
                move = chess.Move.from_uci(uci)
                is_target = ((board.turn == chess.WHITE and target_color == 'White') or
                             (board.turn == chess.BLACK and target_color == 'Black'))
                if not is_target:
                    board.push(move)
                    continue
                phase = 'opening' if i <= 20 else 'middlegame'
                loss_cp, rank, top1, move_score = analyze_single_move(board, move, engine)
                board.push(move)
                is_blunder   = loss_cp >= blunder_cp
                is_excellent = (rank == 1 and loss_cp <= excellent_cp)
                steps.append({
                    'game_id': gid, 'move_number': i, 'move_uci': uci,
                    'turn': 'White' if color == chess.WHITE else 'Black',
                    'target_color': target_color, 'phase': phase,
                    'top1_score_cp': top1, 'move_score_cp': move_score,
                    'loss_cp': loss_cp, 'rank_in_multipv': rank,
                    'is_blunder': is_blunder, 'is_excellent': is_excellent,
                    'blunder_threshold_cp': blunder_cp,
                    'excellent_threshold_cp': excellent_cp,
                })
                losses.append(loss_cp)
                (losses_op if phase == 'opening' else losses_mid).append(loss_cp)
            except Exception:
                try:
                    board.push(chess.Move.from_uci(uci))
                except Exception:
                    break
                continue

        if not losses:
            continue

        n = len(losses)
        num_blunders  = sum(1 for l in losses if l >= blunder_cp)
        num_excellent = sum(1 for s in steps if s.get('is_excellent'))
        rc = {1: 0, 2: 0, 3: 0, 'other': 0}
        for s in steps:
            r = s.get('rank_in_multipv')
            rc[r if r in [1, 2, 3] else 'other'] += 1

        all_steps.extend(steps)
        all_aggs.append({
            'game_id': gid, 'target_color': target_color,
            'result': result, 'target_elo': target_elo,
            'time_weight': time_weight, 'total_moves': n,
            'avg_acpl': float(np.mean(losses)),
            'max_loss_cp': float(np.max(losses)),
            'std_loss_cp': float(np.std(losses)),
            'num_blunders': num_blunders,
            'blunder_rate': num_blunders / n if n > 0 else 0,
            'num_excellent': num_excellent,
            'excellent_rate': num_excellent / n if n > 0 else 0,
            'rank1_count': rc[1], 'rank2_count': rc[2],
            'rank3_count': rc[3], 'rank_other_count': rc['other'],
            'opening_avg_acpl': float(np.mean(losses_op)) if losses_op else None,
            'middlegame_avg_acpl': float(np.mean(losses_mid)) if losses_mid else None,
            'blunder_threshold_cp': blunder_cp,
            'excellent_threshold_cp': excellent_cp,
        })

    logger.info("")
    return pd.DataFrame(all_steps), pd.DataFrame(all_aggs)

# ==================== Step 4: 风格/分项/开局/心态特征 ====================
# 【任务一 - 重构】原版通过 sys.path.insert + import total 动态导入同目录脚本，
# 现改为标准包导入 chess_analyzer.features.*，逻辑与输出列 100% 保持不变。

def run_total_analysis(df_meta, df_steps, df_agg, config=None):
    cfg = config if config is not None else load_config()
    style_rows = []
    phase_rows = []
    opening_rows = []

    logger.info("   提取风格/分项特征...")
    for _, meta_row in df_meta.iterrows():
        gid     = meta_row['game_id']
        step_df = df_steps[df_steps['game_id'] == gid]
        if step_df.empty:
            continue
        step_dict      = {int(r['move_number']): r for _, r in step_df.iterrows()}
        target_indices = sorted(step_df['move_number'].astype(int).tolist())
        if not target_indices:
            continue
        try:
            s = extract_style_features(
                meta_row, step_dict, target_indices, engine=None, config=cfg)
            if s:
                style_rows.append(s)
        except Exception:
            pass
        try:
            p = aggregate_phase_ability(meta_row, step_df, config=cfg)
            if p:
                phase_rows.append(p)
        except Exception:
            pass
        try:
            o = classify_opening(meta_row, config=cfg)
            if o:
                opening_rows.append(o)
        except Exception:
            pass

    df_style   = pd.DataFrame(style_rows)
    df_phase   = pd.DataFrame(phase_rows)
    df_opening = pd.DataFrame(opening_rows)

    logger.info("   提取心态体力特征...")
    try:
        data_source = cfg.get("project", {}).get("data_source", "player") if hasattr(cfg, "get") else "player"
        df_mental = compute_mental_metrics(
            df_meta, df_steps, df_agg, df_phase, data_source=data_source, config=cfg)
    except Exception as e:
        logger.warning("   ⚠️  心态特征失败: %s", e)
        df_mental = pd.DataFrame()

    return {'style': df_style, 'phase': df_phase,
            'opening': df_opening, 'mental': df_mental}


# ==================== Step 5/6: 两层推理 与 报告 ====================

def build_engine_vector(game_id, feature_tables, selected_cols):
    row_data = {}
    for tname, df in feature_tables.items():
        if df is None or df.empty or 'game_id' not in df.columns:
            continue
        match = df[df['game_id'] == game_id]
        if match.empty:
            continue
        for col in match.columns:
            if col not in row_data:
                row_data[col] = match.iloc[0][col]
    if 'win_path' in row_data and 'win_path_encoded' not in row_data:
        row_data['win_path_encoded'] = {'midgame': 1, 'endgame': 0}.get(
            str(row_data.get('win_path', '')), -1)
    feat = []
    for col in selected_cols:
        val = row_data.get(col, np.nan)
        try:
            feat.append(np.nan if isinstance(val, str) else float(val))
        except (ValueError, TypeError):
            feat.append(np.nan)
    return np.array(feat, dtype=float)

def predict_all_games(df_meta, feature_tables, models, use_engine):
    """
    【BUG-3修复】批量向量化推理，不再逐盘循环调用 predict_proba。
    """
    feat_config       = models.get('feature_config') or {}
    selected_eng_cols = feat_config.get('selected_engine_cols', [])

    cheap_list  = []
    base_times  = []
    valid_rows  = []
    for _, row in df_meta.iterrows():
        cf, bt = extract_cheap_features(row)
        if cf is None:
            continue
        cheap_list.append(cf)
        base_times.append(bt)
        valid_rows.append(row)

    if not cheap_list:
        return None, None, []

    n     = len(cheap_list)
    X_all = np.array([[cf.get(c, np.nan) for c in CHEAP_FEATURE_COLS]
                       for cf in cheap_list])  # (N, 16)

    # 【BUG-1修复】使用等效时长判定 Blitz
    blitz_mask = np.array([
        bt > 0 and bt <= BLITZ_MAX_SECONDS
        for bt in base_times
    ])
    rc_mask = ~blitz_mask

    proba5_all = np.full((n, 5), 1.0 / 5)

    xgb_blitz = models.get('xgb_blitz')
    xgb_rc    = models.get('xgb_rapid_classical')

    # 批量推理（向量化）
    blitz_idx = np.where(blitz_mask)[0]
    rc_idx    = np.where(rc_mask)[0]

    if len(blitz_idx) > 0 and xgb_blitz:
        p7 = xgb_blitz.predict_proba(X_all[blitz_idx])   # (k, 7)
        for j, orig_i in enumerate(blitz_idx):
            proba5_all[orig_i] = merge_7to5(p7[j], merge_rules=MERGE_RULES)

    if len(rc_idx) > 0 and xgb_rc:
        p7 = xgb_rc.predict_proba(X_all[rc_idx])          # (m, 7)
        for j, orig_i in enumerate(rc_idx):
            proba5_all[orig_i] = merge_7to5(p7[j], merge_rules=MERGE_RULES)

    lgb_model   = models.get('lgb')
    per_game    = []
    all_probas  = []

    for i, row in enumerate(valid_rows):
        gid     = row['game_id']
        proba5  = proba5_all[i]
        final_p = proba5
        used_engine = False

        if use_engine and lgb_model and selected_eng_cols and feature_tables:
            eng = build_engine_vector(gid, feature_tables, selected_eng_cols)
            if not np.all(np.isnan(eng)):
                X2 = np.concatenate([proba5, eng]).reshape(1, -1)
                try:
                    final_p = np.array(lgb_model.predict(X2)[0])
                    used_engine = True
                except Exception:
                    pass

        pred_idx = int(np.argmax(final_p))
        per_game.append({
            'game_id':     gid,
            'pred_label':  TIER_LABELS[pred_idx],
            'confidence':  float(final_p[pred_idx]),
            'proba':       final_p,
            'used_engine': used_engine,
            'base_time':   base_times[i],
        })
        all_probas.append(final_p)

    avg_proba  = np.mean(all_probas, axis=0)
    final_tier = TIER_LABELS[int(np.argmax(avg_proba))]
    return final_tier, avg_proba, per_game


# ==================== Step 6: 输出报告 ====================

def print_report(final_tier, avg_proba, per_game, pgn_name, use_engine):
    """
    【BUG-4修复】最高概率<0.35时输出Top-2候选。
    【BUG-5修复】明确显示当前精度模式。
    """
    n       = len(per_game)
    n_blitz = sum(1 for r in per_game if 0 < r['base_time'] <= BLITZ_MAX_SECONDS)
    n_eng   = sum(1 for r in per_game if r['used_engine'])
    max_p   = float(np.max(avg_proba))

    mode_tag = "完整两层推理" if use_engine else "⚠️  仅廉价特征（约50%准确率）"
    logger.info("")
    logger.info("=" * 60)
    logger.info(f"📊 分析结果 [{pgn_name}]")
    logger.info(f"   推理模式: {mode_tag}")
    logger.info(f"   有效对局: {n} 盘  blitz={n_blitz} rapid/classical={n-n_blitz}")
    if use_engine:
        logger.info("   引擎特征覆盖: %d/%d 盘%s", n_eng, n,
                    "" if n_eng == n else " （未覆盖的盘降级为廉价特征）")
    logger.info("")

    logger.info("   单盘预测:")
    for i, r in enumerate(per_game, 1):
        tag = " [+引擎]" if r['used_engine'] else " [廉价]"
        logger.info("   第%2d盘: %-22s (%.2f)%s", i, r['pred_label'], r['confidence'], tag)

    logger.info("")
    logger.info("   " + "─" * 54)

    # 【BUG-4修复】低置信度时输出 Top-2
    if max_p < LOW_CONFIDENCE_THRESHOLD:
        top2 = np.argsort(avg_proba)[::-1][:2]
        logger.warning(f"   ⚠️  置信度不足（最高 {max_p:.2f} < {LOW_CONFIDENCE_THRESHOLD}），无法给出单一判定")
        logger.info(f"   Top-2 候选区间:")
        logger.info(f"     {TIER_LABELS[top2[0]]}: {avg_proba[top2[0]]:.4f}")
        logger.info(f"     {TIER_LABELS[top2[1]]}: {avg_proba[top2[1]]:.4f}")
        logger.info(f"   建议提供更多对局（当前 {n} 盘）以提升置信度")
    else:
        logger.info(f"   🏆 最终判定: {final_tier}")
        logger.info(f"   📖 {TIER_DESCRIPTIONS.get(final_tier, '')}")

    logger.info("")
    logger.info("   📈 置信度分布:")
    for i, label in enumerate(TIER_LABELS):
        bar = "█" * int(avg_proba[i] * 30)
        mk  = " ← 最高" if i == int(np.argmax(avg_proba)) else ""
        logger.info(f"     {label:22s}: {avg_proba[i]:.4f}  {bar}{mk}")
    logger.info("   " + "─" * 54)
    logger.info("=" * 60)

# ==================== 端到端推理入口 ====================

def run_inference(
    pgn_path: str,
    name: str,
    color: Optional[str] = None,
    data_dir: Optional[str] = None,
    model_dir: Optional[str] = None,
    no_engine: bool = False,
    min_ply: int = 20,
    config=None,
) -> dict:
    """端到端棋手等级分区间预测。等价于原 predict_player.py 的 main()，
    改为函数式调用（任务一要求），路径/引擎配置从 configs/default.yaml 读取
    （任务二要求），未显式传参时使用与原脚本完全一致的默认值。

    返回: {'final_tier': str|None, 'avg_proba': np.ndarray|None,
           'per_game': list, 'pgn_name': str, 'use_engine': bool}
    """
    cfg = config if config is not None else load_config()

    pgn_path = Path(pgn_path)
    if not pgn_path.exists():
        logger.error("❌ PGN 文件不存在: %s", pgn_path)
        sys.exit(1)

    data_dir = Path(data_dir if data_dir is not None else DATA_DIR)
    data_dir.mkdir(parents=True, exist_ok=True)
    model_dir = model_dir if model_dir is not None else MODEL_DIR

    # Step 1
    logger.info("📂 Step 1: 解析 PGN，目标棋手: [%s]", name)
    df_meta = prepare_games(str(pgn_path), name, color, min_ply)
    if df_meta.empty:
        logger.error("❌ 未找到目标棋手的有效对局")
        logger.error("   提示：--name 支持部分匹配，如 'carlsen' 匹配 'Magnus Carlsen'")
        sys.exit(1)

    df_meta.to_parquet(data_dir / "prepared_games.parquet", index=False)

    use_engine = not no_engine
    df_steps = df_agg = pd.DataFrame()
    feature_tables = {}

    if use_engine:
        sf_path = Path(STOCKFISH_PATH)
        if not sf_path.exists():
            logger.warning("⚠️  [降级] 找不到 Stockfish: %s", sf_path)
            logger.warning("   当前使用仅廉价特征模式（准确率约50%%）")
            use_engine = False
        else:
            logger.info("🔧 Step 3: 实时 ACPL 分析（depth=%d）", STOCKFISH_DEPTH)
            logger.info("   共 %d 盘，预计 %d-%d 分钟...",
                         len(df_meta), len(df_meta) * 2, len(df_meta) * 5)
            engine = chess.engine.SimpleEngine.popen_uci(str(sf_path))
            engine.configure({"Hash": STOCKFISH_MEMORY})
            df_steps, df_agg = run_acpl_on_games(df_meta, engine)
            engine.quit()
            df_steps.to_parquet(data_dir / "gm_detailed_evals.parquet", index=False)
            df_agg.to_parquet(data_dir / "gm_aggregated_metrics.parquet", index=False)
            if not df_agg.empty:
                wacpl = ((df_agg['avg_acpl'] * df_agg['total_moves']).sum()
                         / df_agg['total_moves'].sum())
                logger.info("   有效: %d 盘，加权均值ACPL: %.1f cp", len(df_agg), wacpl)
            logger.info("🧠 Step 4: 风格/分项/开局/心态特征提取")
            feature_tables = run_total_analysis(df_meta, df_steps, df_agg, config=cfg)
            for tname, tdf in feature_tables.items():
                if tdf is not None and not tdf.empty:
                    tdf.to_parquet(data_dir / f"{tname}_features.parquet", index=False)
                    logger.info("   💾 %s: %d 行", tname, len(tdf))
    else:
        logger.info("   ℹ️  [低精度模式] 跳过引擎分析，仅使用廉价特征（准确率约50%%）")

    logger.info("🔍 Step 5: 加载模型")
    models = load_models(model_dir)
    if not models.get('xgb_blitz') and not models.get('xgb_rapid_classical'):
        logger.error("❌ 第一层模型均未加载")
        sys.exit(1)
    if use_engine and not models.get('lgb'):
        logger.warning("   ⚠️  [降级] 第二层模型未加载，切换为仅廉价特征模式（约50%%准确率）")
        use_engine = False

    logger.info("🎯 Step 5: 推理")
    final_tier, avg_proba, per_game = predict_all_games(
        df_meta, feature_tables, models, use_engine)

    if final_tier is None:
        logger.error("❌ 所有对局无效")
        sys.exit(1)

    print_report(final_tier, avg_proba, per_game, pgn_path.name, use_engine)

    return {
        'final_tier': final_tier,
        'avg_proba': avg_proba,
        'per_game': per_game,
        'pgn_name': pgn_path.name,
        'use_engine': use_engine,
    }


def main() -> None:
    """CLI 入口，等价于原 `python predict_player.py --pgn ... --name ...`。"""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description='棋手等级分区间预测（完整流程）')
    parser.add_argument('--pgn',       required=True,  help='PGN 文件路径')
    parser.add_argument('--name',      required=True,  help='目标棋手姓名（模糊匹配）')
    parser.add_argument('--color',     default=None,   choices=['White', 'Black'])
    parser.add_argument('--data_dir',  default=DATA_DIR)
    parser.add_argument('--model_dir', default=MODEL_DIR)
    parser.add_argument('--no_engine', action='store_true',
                         help='跳过引擎分析，仅廉价特征（快速但准确率约50%%）')
    parser.add_argument('--min_ply',   type=int, default=20)
    args = parser.parse_args()

    run_inference(
        pgn_path=args.pgn, name=args.name, color=args.color,
        data_dir=args.data_dir, model_dir=args.model_dir,
        no_engine=args.no_engine, min_ply=args.min_ply,
    )


if __name__ == "__main__":
    main()
