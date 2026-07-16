"""
stats.bootstrap
================================================================================
百分位统计：matched-N 双层 Bootstrap、tier 分层加权合并、方向翻转、
以及风格/能力雷达图和 ACPL 分布图共用的特征/方向常量定义。
从原 extra_style.py 迁移（逻辑与数值口径 100% 保持不变）。

所有百分位统一约定：数值越高 = 表现/水平越好（对"低优"指标已按 direction_map
自动翻转），风格类"倾向性"指标翻转后表示"高于基线均值的程度"（>50%=比基线更倾向）。
"""

import logging

logger = logging.getLogger(__name__)

import sys

import numpy as np
import pandas as pd

from chess_analyzer.core.color_utils import assign_tier
from chess_analyzer.core.config import load_config

_cfg = load_config()
_thresholds = _cfg.get("thresholds", {}) if hasattr(_cfg, "get") else {}
_bootstrap_cfg = _cfg.get("bootstrap", {}) if hasattr(_cfg, "get") else {}

# ==================== 风格雷达 5 个维度 ====================
STYLE_RADAR_FEATURES = [
    'effective_sacrifice_rate',      # 有效弃子率
    'pawn_storm_kingside_rate',      # 王翼兵风暴率
    'pawn_storm_queenside_rate',     # 后翼兵风暴率
    'pawn_storm_center_rate',        # 中心兵风暴率
    'heavy_invasion_rate',           # 重子侵入率
]
STYLE_LABEL_MAP = {
    'effective_sacrifice_rate': '有效弃子',
    'pawn_storm_kingside_rate': '王翼兵风暴',
    'pawn_storm_queenside_rate': '后翼兵风暴',
    'pawn_storm_center_rate': '中心兵风暴',
    'heavy_invasion_rate': '重子侵入',
}
# 风格特征是"倾向性"而非"好坏"，统一不翻转（direction=1，纯粹表示是否高于基线）
STYLE_DIRECTIONS = {feat: 1 for feat in STYLE_RADAR_FEATURES}

# ==================== 能力雷达（含战术警觉性） ====================
ABILITY_GROUPS = {
    '开局能力': {
        'features': ['opening_acpl', 'opening_deviation_rate', 'developed_light_pieces'],
        'directions': [-1, -1, 1],
        'labels': ['开局ACPL↓', '开局偏离↓', '轻子出动↑']
    },
    '中局能力': {
        'features': ['middlegame_acpl', 'mid_peak_accuracy', 'middlegame_invasion_rate', 'tactical_punish_rate'],
        'directions': [-1, 1, 1, 1],
        'labels': ['中局ACPL↓', '中局卓越↑', '中局侵入↑', '战术警觉↑']
    },
    '残局能力': {
        'features': ['endgame_acpl', 'valid_endgame_pool', 'endgame_win_rate', 'transition_success', 'entry_endgame_eval'],
        'directions': [-1, 1, 1, 1, -1],
        'labels': ['残局ACPL↓', '残局池(1/0)', '残局胜率↑', '过渡成功↑', '入场评估↓']
    }
}

# 【修复B】唯一的翻转依据来源，雷达图和CSV导出都从这里取，不会再互相矛盾
ABILITY_DIRECTIONS = {}
for _group in ABILITY_GROUPS.values():
    for _feat, _d in zip(_group['features'], _group['directions']):
        ABILITY_DIRECTIONS[_feat] = _d

# ACPL 分布图指标
ACPL_METRICS = ['opening_acpl', 'middlegame_acpl', 'endgame_acpl']
ACPL_LABELS = {'opening_acpl': '开局平均ACPL', 'middlegame_acpl': '中局平均ACPL', 'endgame_acpl': '残局平均ACPL'}

# 诊断阈值（来自 configs/default.yaml 的 thresholds.deviation_high / deviation_low，
# 未提供配置时使用与原脚本一致的 15 / -15）
DEVIATION_HIGH = _thresholds.get("deviation_high", 15)
DEVIATION_LOW = _thresholds.get("deviation_low", -15)

# 最小基线样本量
MIN_BASELINE_N = _thresholds.get("min_baseline_n", 30)

# Bootstrap 参数（来自 configs/default.yaml 的 bootstrap 段，未提供配置时使用原脚本默认值）
RADAR_N_OUTER = _bootstrap_cfg.get("radar_outer", 300)
RADAR_K_INNER = _bootstrap_cfg.get("radar_inner", 100)
RADAR_K_POINT = _bootstrap_cfg.get("radar_point", 2000)
ACPL_N_OUTER = _bootstrap_cfg.get("acpl_outer", 1000)
ACPL_K_INNER = _bootstrap_cfg.get("acpl_inner", 200)
ACPL_K_POINT = _bootstrap_cfg.get("acpl_point", 5000)


def load_and_merge(style_file, meta_file):
    df_style = pd.read_parquet(style_file)
    df_meta = pd.read_parquet(meta_file)
    if 'tier' in df_style.columns:
        df = df_style.copy()
    else:
        if 'target_elo' in df_meta.columns:
            df_meta['tier'] = df_meta['target_elo'].apply(assign_tier)
        elif 'max_rating' in df_meta.columns:
            df_meta['tier'] = df_meta['max_rating'].apply(assign_tier)
        else:
            raise ValueError("元数据中无等级分列")
        df = df_style.merge(df_meta[['game_id', 'tier']], on='game_id', how='left')
    df = df[df['tier'].notna()].copy()
    return df


def compute_style_rates(df):
    """
    为风格表计算 _rate 列。
    【修复E】total_moves缺失时防御性跳过，而不是直接报错。
    【修复G】原来用字符串replace自动生成rate列名（如 effective_sacrifices -> effective_sacrifices_rate），
    但STYLE_RADAR_FEATURES里写的是 effective_sacrifice_rate（单数），两者对不上，
    导致"有效弃子"这个维度在雷达图里悄悄消失、且不报错。改用显式映射表，一一对应，
    避免任何字段名不一致导致的静默丢失。
    """
    df = df.copy()
    # 原始列名 -> 目标rate列名（必须和 STYLE_RADAR_FEATURES 里的名字完全一致）
    count_to_rate = {
        'effective_sacrifices': 'effective_sacrifice_rate',
        'pawn_storm_kingside': 'pawn_storm_kingside_rate',
        'pawn_storm_queenside': 'pawn_storm_queenside_rate',
        'pawn_storm_center': 'pawn_storm_center_rate',
        'total_pawn_storm': 'total_pawn_storm_rate',
        'heavy_invasion_count': 'heavy_invasion_rate',
    }
    if 'total_moves' not in df.columns:
        logger.warning("⚠️ 警告：数据中缺少 total_moves 列，无法计算 _rate 系列指标，将跳过")
        return df
    for raw_col, rate_col in count_to_rate.items():
        if raw_col in df.columns and rate_col not in df.columns:
            df[rate_col] = df[raw_col] / df['total_moves'].clip(lower=1)

    # 安全检查：确认雷达图需要的每个特征都真的生成出来了，缺失就明确报警而不是静默消失
    missing = [f for f in STYLE_RADAR_FEATURES if f not in df.columns]
    if missing:
        logger.warning(f"⚠️ 警告：以下风格特征未能生成，将不会出现在雷达图中: {missing}")
    return df


# --------------------- 【核心修复A】matched-N bootstrap ---------------------
def matched_n_single_tier(player_vals, baseline_vals, n_outer, k_inner, k_point):
    """
    对单个tier内的一组数值，做matched-N bootstrap。
    返回：(percentile_point, ci_low, ci_high, player_mean, n_player, matched_point数组)
    matched_point数组供绘图时画"N局均值分布"曲线用。
    百分位约定：越高=均值越大（原始方向，未翻转，翻转交给上层按direction处理）。
    """
    player_vals = np.asarray(player_vals, dtype=float)
    baseline_vals = np.asarray(baseline_vals, dtype=float)
    n_player = len(player_vals)
    player_mean = player_vals.mean()

    matched_point = np.random.choice(baseline_vals, size=(k_point, n_player), replace=True).mean(axis=1)
    percentile_point = 100 * np.mean(matched_point > player_mean)

    boot = np.empty(n_outer)
    for b in range(n_outer):
        baseline_b = np.random.choice(baseline_vals, size=len(baseline_vals), replace=True)
        matched_b = np.random.choice(baseline_b, size=(k_inner, n_player), replace=True).mean(axis=1)
        player_mean_b = np.random.choice(player_vals, size=n_player, replace=True).mean()
        boot[b] = 100 * np.mean(matched_b > player_mean_b)
    ci_low, ci_high = np.percentile(boot, [2.5, 97.5])

    return percentile_point, ci_low, ci_high, player_mean, n_player, matched_point


def compute_tier_percentiles(player_df, baseline_df, feature_list, direction_map=None,
                              tier_col='tier', n_outer=RADAR_N_OUTER,
                              k_inner=RADAR_K_INNER, k_point=RADAR_K_POINT):
    """
    【已修复】按tier分层，对每个feature做matched-N bootstrap（而不是直接拿均值比单盘分布），
    再按朱锦尔各tier对局数加权合并成一个百分位点估计 + 95% CI。
    direction_map: {feature: 1 or -1}，-1表示"数值越低越好"，翻转百分位方向。
    返回：{feat: {'percentile', 'ci_low', 'ci_high', 'n_games'}}
    """
    direction_map = direction_map or {}
    results = {}
    tiers = player_df[tier_col].dropna().unique()

    for feat in feature_list:
        if feat not in player_df.columns or feat not in baseline_df.columns:
            continue

        # 【修复I】不再用 direction_map.get(feat, 1) 这种隐藏默认值。
        # 默认值=1 意味着"没声明方向就悄悄当成需要翻转"，如果以后新增特征、
        # 或者某处字段名打错导致对不上，这里会静默给出方向错误的百分位，
        # 且不会有任何报错提示——这是比"算错"更危险的"悄悄算错"。
        # 改为强制要求每个传入的特征都必须在 direction_map 里显式声明，
        # 否则直接跳过并打印警告，绝不用隐藏默认值蒙混过关。
        if feat not in direction_map:
            logger.warning("⚠️ 警告：特征 '%s' 未在 direction_map 中声明方向(+1/-1)，"
                           "为避免方向出错，已跳过该特征，不计入本次百分位计算。请检查拼写或补充声明。", feat)
            continue

        tier_data = []
        for tier in tiers:
            p_sub = player_df[player_df[tier_col] == tier][feat].dropna().values
            b_sub = baseline_df[baseline_df[tier_col] == tier][feat].dropna().values
            if len(p_sub) == 0 or len(b_sub) == 0:
                continue
            tier_data.append((p_sub, b_sub))
        if not tier_data:
            continue

        # ---- 点估计：各tier分别matched-N，再按朱锦尔盘数加权合并 ----
        w_sum, w_tot = 0.0, 0
        for p_sub, b_sub in tier_data:
            n_player = len(p_sub)
            player_mean = p_sub.mean()
            matched = np.random.choice(b_sub, size=(k_point, n_player), replace=True).mean(axis=1)
            pct = 100 * np.mean(matched > player_mean)
            w_sum += pct * n_player
            w_tot += n_player
        point_pct = w_sum / w_tot

        # ---- CI：双层bootstrap，每次同时重抽所有tier再加权合并 ----
        boot_pct = np.empty(n_outer)
        for i in range(n_outer):
            bw_sum, bw_tot = 0.0, 0
            for p_sub, b_sub in tier_data:
                n_player = len(p_sub)
                baseline_b = np.random.choice(b_sub, size=len(b_sub), replace=True)
                matched_b = np.random.choice(baseline_b, size=(k_inner, n_player), replace=True).mean(axis=1)
                player_mean_b = np.random.choice(p_sub, size=n_player, replace=True).mean()
                pct_b = 100 * np.mean(matched_b > player_mean_b)
                bw_sum += pct_b * n_player
                bw_tot += n_player
            boot_pct[i] = bw_sum / bw_tot if bw_tot > 0 else np.nan
        ci_low, ci_high = np.nanpercentile(boot_pct, [2.5, 97.5])

        # ---- 方向翻转（唯一依据：direction_map，此时feat已确认存在于其中）----
        # 原始公式 raw_pct = 100*mean(matched > player_mean)：
        #   对"低优"指标(direction=-1，如ACPL)，matched更大=基线更差=玩家更强，
        #   raw_pct本身就已经是"玩家优于X%基线"，不需要翻转。
        # 对"高优"指标(direction=+1，如弃子率、胜率)，matched更大=基线数值更高=基线更强，
        #   此时raw_pct实际是"基线比玩家强的比例"，要用 100-raw_pct 才是"玩家优于X%基线"。
        d = direction_map[feat]
        if d not in (1, -1):
            logger.warning(f"⚠️ 警告：特征 '{feat}' 的方向值为 {d}，不是合法的 +1/-1，已跳过该特征。")
            continue
        if d == 1:
            point_pct, ci_low, ci_high = 100 - point_pct, 100 - ci_high, 100 - ci_low

        results[feat] = {
            'percentile': max(0, min(100, point_pct)),
            'ci_low': max(0, min(100, ci_low)),
            'ci_high': max(0, min(100, ci_high)),
            'n_games': int(sum(len(p) for p, _ in tier_data)),
        }
    return results


def tier_stratified_category_pct(player_df, baseline_df, label_col='label', tier_col='tier'):
    """
    【修复D】开局标签占比对比，按tier分层后再按朱锦尔各tier对局数加权合并，
    避免"朱锦尔集中在Tier3，但基线是全体tier混合"造成的虚假偏差。
    返回：(all_labels, player_pct_dict, baseline_pct_dict)
    """
    tiers = player_df[tier_col].dropna().unique()
    tier_frames = []
    label_set = set()
    for tier in tiers:
        p_sub = player_df[player_df[tier_col] == tier]
        b_sub = baseline_df[baseline_df[tier_col] == tier]
        if len(p_sub) == 0 or len(b_sub) == 0:
            continue
        tier_frames.append((p_sub, b_sub))
        label_set |= set(p_sub[label_col].dropna().unique()) | set(b_sub[label_col].dropna().unique())

    all_labels = sorted(label_set)
    p_pct_agg = {lab: 0.0 for lab in all_labels}
    b_pct_agg = {lab: 0.0 for lab in all_labels}
    total_weight = 0

    for p_sub, b_sub in tier_frames:
        w = len(p_sub)
        total_weight += w
        p_counts = p_sub[label_col].value_counts()
        b_counts = b_sub[label_col].value_counts()
        p_total = p_counts.sum()
        b_total = b_counts.sum()
        for lab in all_labels:
            p_rate = 100 * p_counts.get(lab, 0) / p_total if p_total > 0 else 0
            b_rate = 100 * b_counts.get(lab, 0) / b_total if b_total > 0 else 0
            p_pct_agg[lab] += w * p_rate
            b_pct_agg[lab] += w * b_rate

    if total_weight > 0:
        for lab in all_labels:
            p_pct_agg[lab] /= total_weight
            b_pct_agg[lab] /= total_weight

    return all_labels, p_pct_agg, b_pct_agg
def validate_direction_maps():
    """
    【修复I配套】启动时自检：STYLE_RADAR_FEATURES 和 ABILITY_GROUPS 里用到的每个特征，
    是否都能在对应的 direction 字典里查到方向声明。提前暴露"字段名打错/漏声明"这类
    配置问题，而不是等到跑到一半才在某个特征上悄悄跳过。
    """
    problems = []
    for feat in STYLE_RADAR_FEATURES:
        if feat not in STYLE_DIRECTIONS:
            problems.append(f"STYLE_RADAR_FEATURES 中的 '{feat}' 未在 STYLE_DIRECTIONS 中声明方向")
    for group_name, group_info in ABILITY_GROUPS.items():
        for feat in group_info['features']:
            if feat not in ABILITY_DIRECTIONS:
                problems.append(f"ABILITY_GROUPS['{group_name}'] 中的 '{feat}' 未在 ABILITY_DIRECTIONS 中声明方向")
    if problems:
        logger.error("❌ 配置自检失败，以下特征缺少方向声明，请修正后再运行：")
        for p in problems:
            logger.info(f"   - {p}")
        sys.exit(1)
    logger.info("✅ 配置自检通过：所有特征均已声明方向(+1/-1)")
