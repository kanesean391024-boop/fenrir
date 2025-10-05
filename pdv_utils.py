import os
import time
import json
import math
from typing import List, Dict
from config_loader import load_globals, load_pdvs, save_pdvs


def _clamp01(x: float) -> float:
    return 0.0 if x < 0.0 else 1.0 if x > 1.0 else x


def _ensure_dirs() -> None:
    os.makedirs("chatlogs", exist_ok=True)


def _pdv_values_map(pdvs_cfg: Dict[str, dict]) -> Dict[str, float]:
    return {name: float(cfg.get("value", 0.5)) for name, cfg in pdvs_cfg.items()}


def apply_and_persist_pdv_adjustments(adjs: List[dict]) -> Dict[str, float]:
    """Apply gamma-scaled PDV deltas and persist to pdvs.json + history/live files."""
    try:
        globals_cfg = load_globals()
    except Exception:
        globals_cfg = {}
    try:
        gamma = float(globals_cfg.get("pdv_gamma", 2.0))
    except Exception:
        gamma = 2.0

    try:
        pdvs_cfg = load_pdvs()
        pdvs_loaded = True
    except Exception:
        pdvs_cfg = {}
        pdvs_loaded = False
    values = _pdv_values_map(pdvs_cfg)

    changed = False
    for item in adjs or []:
        name = item.get("name")
        if not isinstance(name, str) or not name.strip():
            continue

        if "delta_pct" in item:
            try:
                delta = float(item["delta_pct"])
            except Exception:
                continue
        elif "delta" in item:
            try:
                delta = float(item["delta"])
            except Exception:
                continue
        else:
            continue
        if not math.isfinite(delta):
            continue

        x = float(values.get(name, 0.5))
        g = (4.0 * x * (1.0 - x)) ** gamma
        x2 = _clamp01(x + delta * g)

        if abs(x2 - x) > 1e-12:
            values[name] = x2
            if name not in pdvs_cfg:
                pdvs_cfg[name] = {"name": name, "description": "", "value": x2}
            else:
                pdvs_cfg[name]["value"] = x2
            print(f"[PDVM] {name}: {x:.4f} -> {x2:.4f}")
            changed = True

    if changed:
        if pdvs_loaded:
            save_pdvs(pdvs_cfg)
        _ensure_dirs()
        with open(os.path.join("chatlogs", "pdv_history.jsonl"), "a", encoding="utf-8") as f:
            f.write(json.dumps({"ts": time.time(), "pdvs": values}, ensure_ascii=False) + "\n")
        with open(os.path.join("chatlogs", "pdvs_live.json"), "w", encoding="utf-8") as f:
            json.dump(values, f)
        print("[PDVM] Discord-triggered PDVMs applied.")

    return values
