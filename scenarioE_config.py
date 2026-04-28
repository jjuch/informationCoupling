import json, hashlib
from copy import deepcopy

def deep_update(base: dict, upd: dict) -> dict:
    """Recursively update nested dictionaries."""
    out = deepcopy(base)
    for k, v in upd.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_update(out[k], v)
        else:
            out[k] = deepcopy(v)
    return out

def load_cases_json(path):
    cfg = json.loads(path.read_text(encoding="utf-8"))
    return cfg

def select_case(cases_cfg: dict, kappa: float, G_shape: str) -> dict:
    """
    Return merged config dict for the requested (kappa, G_shape).
    If no specific case found, returns defaults.
    """
    defaults = cases_cfg["defaults"]
    chosen = None
    for c in cases_cfg.get("cases", []):
        if float(c.get("kappa")) == float(kappa) and c.get("G_shape") == G_shape:
            chosen = c
            break

    if chosen is None:
        return deepcopy(defaults)

    overrides = chosen.get("overrides", {})

    merged = deep_update(defaults, overrides)
    merged["_case_note"] = chosen.get("note", "")
    merged["_case_key"] = {"kappa": float(kappa), "G_shape": G_shape}
    return merged

def hash_config(cfg: dict) -> str:
    """
    Stable hash of config (excluding non-deterministic keys).
    """
    tmp = deepcopy(cfg)
    tmp.pop("_case_note", None)
    tmp.pop("_case_key", None)
    tmp.pop("plots", None)
    s = json.dumps(tmp, sort_keys=True, ensure_ascii=True).encode("utf-8")
    return hashlib.sha1(s).hexdigest()


def meta_matches(meta: dict, kappa: float, G_shape: str, cfg_hash: str) -> bool:
    if not meta:
        return False
    checks = [
        ("scenario", "E_orbit_tracking_FF_SMC"),
        ("kappa", float(kappa)),
        ("G_shape", G_shape),
        ("cfg_hash", cfg_hash),
    ]
    return all(meta.get(k) == v for k, v in checks)
