import json
from pathlib import Path

# ---- defaults  (if config is missing / partial) ----
_DEFAULTS = {
    "max_pen": 1e9,
    "feas_tol": 1e-6,
    "eps": 1e-12,
    "ctol": 1e-4,
    "stress_cases": [
        "load_extreme_high", "load_extreme_low",
        "fuel_price_up", "fuel_price_down", "geo_shift", "extreme_weather",
        "islanded_operation", "nk_contingencies",
        "reactive_scarcity",
    ],
}

# ---- load src/config.json ----
_cfg_path = Path(__file__).with_name("config.json")
_cfg = dict(_DEFAULTS)
if _cfg_path.exists():
    _cfg.update(json.loads(_cfg_path.read_text(encoding="utf-8")))


max_pen = _cfg["max_pen"]
feas_tol = _cfg["feas_tol"]
eps = _cfg["eps"]
ctol = _cfg["ctol"]
stress_cases = _cfg["stress_cases"]
