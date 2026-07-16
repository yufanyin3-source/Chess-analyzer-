"""
core.color_utils
================================================================================
颜色视角转换 与 等级分分层（tier）工具。
从原 total.py (get_relative_score) 与 extra_style.py (assign_tier) 迁移，
逻辑 100% 保持不变；tier 分段现在可选地从 configs/default.yaml 的
`elo_tiers` 读取（未提供配置时退化为与原脚本完全一致的硬编码分段）。
"""

from __future__ import annotations

from typing import Optional

import pandas as pd

from chess_analyzer.core.config import load_config

# 与原 extra_style.py 中 assign_tier 完全一致的硬编码分段（默认兜底，
# 保证不提供 config.yaml 时行为 100% 不变）。
_DEFAULT_TIERS = [
    (0, 1400, None),
    (1400, 1600, "Tier0_1400_1599"),
    (1600, 1800, "Tier1_1600_1799"),
    (1800, 2000, "Tier2_1800_1999"),
    (2000, 2200, "Tier3_2000_2199"),
    (2200, 2400, "Tier4_2200_2399"),
    (2400, 2600, "Tier5_2400_2599"),
    (2600, 999999, "Tier6_2600plus"),
]


def get_relative_score(score_cp, target_color):
    """
    【核心修复】将白方视角的评估分转换为目标棋手视角。
    - 若目标执白，直接返回原分（正值=白方好=目标好）
    - 若目标执黑，取相反数（正值=黑方好=目标好）
    """
    if target_color == 'White':
        return score_cp
    else:  # 'Black'
        return -score_cp if score_cp is not None else None


def _tiers_from_config(config=None):
    cfg = config if config is not None else load_config()
    raw_tiers = cfg.get("elo_tiers") if isinstance(cfg, dict) else None
    if not raw_tiers:
        return _DEFAULT_TIERS
    return [(lo, hi, name) for lo, hi, name in raw_tiers]


def assign_tier(rating, config=None):
    """按等级分区间返回 tier 名称，None 表示低于最低分层（<1400）。

    `config` 为可选的 chess_analyzer 配置对象（见 core.config.load_config）；
    不传时使用与原脚本完全相同的硬编码分段。
    """
    if pd.isna(rating) or rating is None:
        return None
    r = float(rating)
    for lo, hi, name in _tiers_from_config(config):
        if lo <= r < hi:
            return name
    return None
