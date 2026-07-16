"""
core.config
================================================================================
Centralised configuration loader for chess_analyzer.

Design goals (see refactor task #2):
  1. Zero hardcoded paths/thresholds in feature/pipeline code - everything
     lives in configs/default.yaml.
  2. Backward compatible: if the caller does not supply a config file, the
     library falls back to configs/default.yaml (shipped with the package),
     which reproduces the exact values that used to be hardcoded in
     total.py / extra_style.py / predict_player.py. The program never
     requires an external config to run.
  3. Environment variables override individual values, e.g.:
         export STOCKFISH_PATH=/opt/stockfish/stockfish
         export CHESS_ANALYZER_THRESHOLDS__OPENING_MOVES=10
     A small set of "well known" env vars (STOCKFISH_PATH, STOCKFISH_DEPTH,
     STOCKFISH_MEMORY, STOCKFISH_MULTIPV) are supported directly for
     convenience; any other value can be overridden with the generic
     CHESS_ANALYZER__<SECTION>__<KEY> convention (double underscore
     separated, case-insensitive).
"""

from __future__ import annotations

import os
import copy
from pathlib import Path
from typing import Any, Optional

import yaml

_DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[3] / "configs" / "default.yaml"

_ENV_PREFIX = "CHESS_ANALYZER__"

# Convenience aliases for the most commonly overridden values.
_SIMPLE_ENV_ALIASES = {
    "STOCKFISH_PATH": ("stockfish", "path"),
    "STOCKFISH_DEPTH": ("stockfish", "depth"),
    "STOCKFISH_MEMORY": ("stockfish", "memory"),
    "STOCKFISH_MULTIPV": ("stockfish", "multipv"),
}


class Config(dict):
    """A dict that also supports attribute-style access, recursively.

    Example:
        cfg = load_config()
        cfg.thresholds.opening_moves          # attribute access
        cfg["thresholds"]["opening_moves"]     # classic dict access
    """

    def __getattr__(self, item: str) -> Any:
        try:
            value = self[item]
        except KeyError as exc:
            raise AttributeError(item) from exc
        if isinstance(value, dict) and not isinstance(value, Config):
            value = Config(value)
            self[item] = value
        return value

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value

    def flat(self, section: str) -> dict:
        """Return a section as a plain flat UPPER_CASE dict.

        This exists purely to make migration painless: legacy code used
        module-level dicts such as ``CONFIG["OPENING_MOVES"]``. Calling
        ``cfg.flat("thresholds")`` returns
        ``{"OPENING_MOVES": 12, "DRAWN_EVAL_LIMIT": 150, ...}`` so old call
        sites keep working almost unchanged.
        """
        section_dict = self.get(section, {})
        return {k.upper(): v for k, v in section_dict.items()}


def _deep_update(base: dict, overrides: dict) -> dict:
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def _coerce(value: str) -> Any:
    """Best-effort string -> Python scalar conversion for env var overrides."""
    lowered = value.lower()
    if lowered in ("true", "false"):
        return lowered == "true"
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    return value


def _apply_env_overrides(cfg: dict) -> dict:
    # Generic CHESS_ANALYZER__SECTION__KEY overrides.
    for env_key, raw_value in os.environ.items():
        if env_key.startswith(_ENV_PREFIX):
            path = env_key[len(_ENV_PREFIX):].lower().split("__")
            node = cfg
            for part in path[:-1]:
                node = node.setdefault(part, {})
            node[path[-1]] = _coerce(raw_value)

    # Convenience aliases (STOCKFISH_PATH, etc.)
    for env_key, (section, key) in _SIMPLE_ENV_ALIASES.items():
        if env_key in os.environ:
            cfg.setdefault(section, {})[key] = _coerce(os.environ[env_key])

    return cfg


def load_config(path: Optional[str | Path] = None) -> Config:
    """Load configuration from ``path`` (or the packaged default), applying
    environment variable overrides on top.

    If ``path`` is None and no file exists at the default location either,
    an empty configuration is used and every consumer falls back to its own
    hardcoded defaults - the program must never hard-require a config file.
    """
    if path is not None:
        cfg_path = Path(path)
    else:
        cfg_path = _DEFAULT_CONFIG_PATH

    if cfg_path.exists():
        with open(cfg_path, encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
    else:
        raw = {}

    raw = copy.deepcopy(raw)
    raw = _apply_env_overrides(raw)
    return Config(raw)


# Module-level singleton so `from chess_analyzer.core.config import CONFIG`
# works too, but callers are encouraged to use load_config() explicitly so
# tests can inject their own configuration.
CONFIG = load_config()


def get_config(path: Optional[str | Path] = None) -> Config:
    """Return a freshly loaded Config (re-reads the file each call)."""
    return load_config(path)
