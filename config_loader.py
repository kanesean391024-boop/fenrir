# JSON configuration helpers for Fenrir.
from __future__ import annotations

import json
import os
from typing import Dict, List

def _atomic_write(path: str, data: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, sort_keys=True)
    os.replace(tmp, path)

def load_globals(path: str = 'confs/globals.json') -> dict:
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Missing globals config at {path}") from e
    if not isinstance(data, dict) or 'model' not in data:
        raise ValueError("globals.json missing required fields")
    return data

def save_globals(globals_cfg: dict, path: str = 'confs/globals.json') -> None:
    _atomic_write(path, globals_cfg)

def load_pdvs(path: str = 'confs/pdvs.json') -> Dict[str, dict]:
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Missing PDVs config at {path}") from e
    pdvs = data.get('pdvs')
    if not isinstance(pdvs, list):
        raise ValueError("pdvs.json: 'pdvs' must be a list")
    out = {}
    for idx, p in enumerate(pdvs):
        if not isinstance(p, dict) or 'name' not in p or 'value' not in p:
            raise ValueError(f"pdvs[{idx}] missing name or value")
        out[p['name']] = p
    return out
