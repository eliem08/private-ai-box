"""Sizing math and configurator for on-prem LLM appliances."""

from __future__ import annotations

import math
from dataclasses import dataclass

__all__ = [
    "ModelSpec",
    "GPU",
    "MODELS",
    "GPUS",
    "QUANT_BYTES",
    "vram_needed",
    "decode_tps",
    "configure",
    "bom",
    "COMPLIANCE",
    "PRICES_AS_OF",
    "load_prices",
    "price_age_days",
]

QUANT_BYTES = {"fp16": 2.0, "int8": 1.0, "int4": 0.5}
KV_BYTES = 2.0  # fp16 KV cache
OVERHEAD = 1.10  # activations, CUDA context, fragmentation
BW_EFFICIENCY = 0.6


@dataclass(frozen=True)
class ModelSpec:
    key: str
    params_b: float
    active_b: float
    layers: int
    kv_heads: int
    head_dim: int


@dataclass(frozen=True)
class GPU:
    name: str
    vram_gb: int
    bandwidth_gbs: float
    price: int
    datacenter: bool


MODELS = {
    "8b": ModelSpec("8b", 8, 8, 32, 8, 128),
    "32b": ModelSpec("32b", 32, 32, 64, 8, 128),
    "70b": ModelSpec("70b", 70, 70, 80, 8, 128),
    "moe-120b": ModelSpec("moe-120b", 120, 12, 36, 8, 64),
}

GPUS = [
    GPU("RTX 4090 24GB", 24, 1008, 1900, False),
    GPU("RTX 5090 32GB", 32, 1792, 2600, False),
    GPU("RTX 6000 Ada 48GB", 48, 960, 7000, False),
    GPU("L40S 48GB", 48, 864, 8500, True),
    GPU("A100 80GB", 80, 2039, 15000, True),
    GPU("H100 80GB", 80, 3350, 28000, True),
]
PRICES_AS_OF = "2026-10-01"  # street prices for GPUS above; refresh, or pass a price file
STALE_AFTER_DAYS = 90


def load_prices(path: str) -> tuple[list[GPU], str]:
    """Merge a reseller price file into the built-in GPU list.

    {"as_of": "2026-11-15", "gpus": [{"name": "H100 80GB", "price": 26500},
     {"name": "B200 192GB", "vram_gb": 192, "bandwidth_gbs": 8000, "price": 38000, "datacenter": true}]}
    A known name with only a price updates it; an unknown name must carry every field.
    """
    import datetime
    import json

    with open(path, encoding="utf-8-sig") as fh:
        doc = json.load(fh)
    if not isinstance(doc, dict) or not isinstance(doc.get("gpus"), list):
        raise ValueError(f"{path}: expected {{'as_of': 'YYYY-MM-DD', 'gpus': [...]}}")
    as_of = str(doc.get("as_of", ""))
    datetime.date.fromisoformat(as_of)  # ValueError on a missing or malformed date
    by_name = {g.name: g for g in GPUS}
    for row in doc["gpus"]:
        name = row.get("name") if isinstance(row, dict) else None
        price = row.get("price") if isinstance(row, dict) else None
        if not name or not isinstance(price, (int, float)) or not math.isfinite(price) or price <= 0:
            raise ValueError(f"{path}: every gpu needs a name and a positive price: {row!r:.80}")
        old = by_name.get(name)
        if old and set(row) <= {"name", "price"}:
            by_name[name] = GPU(name, old.vram_gb, old.bandwidth_gbs, int(price), old.datacenter)
            continue
        try:
            vram, bw = int(row["vram_gb"]), float(row["bandwidth_gbs"])
        except (KeyError, TypeError, ValueError):
            raise ValueError(f"{path}: new gpu {name!r} needs vram_gb and bandwidth_gbs") from None
        by_name[name] = GPU(name, vram, bw, int(price), bool(row.get("datacenter", False)))
    return list(by_name.values()), as_of


def price_age_days(as_of: str, today=None) -> int:
    import datetime

    return ((today or datetime.date.today()) - datetime.date.fromisoformat(as_of)).days


COMPLIANCE = {
    "air_gapped": [
        "No WAN route from model server VLAN",
        "Offline model + update media with checksum verification",
        "Disable telemetry in OS, drivers and model server",
    ],
    "hipaa": [
        "Full-disk encryption (LUKS) with TPM unlock",
        "Per-user SSO + MFA",
        "Audit log of prompts/responses retained 6 years",
        "BAA with the integrator for on-site service",
    ],
    "soc2": [
        "Change management for model/config updates",
        "Quarterly access review",
        "Backups of RAG index and configs, tested restore",
    ],
}


def kv_cache_gb(m: ModelSpec, context: int, sequences: int) -> float:
    per_token = 2 * m.layers * m.kv_heads * m.head_dim * KV_BYTES
    return per_token * context * sequences / 1e9


def vram_needed(m: ModelSpec, quant: str, context: int, sequences: int) -> float:
    weights = m.params_b * QUANT_BYTES[quant]
    return (weights + kv_cache_gb(m, context, sequences)) * OVERHEAD


def decode_tps(m: ModelSpec, quant: str, gpu: GPU, count: int) -> float:
    """Aggregate single-stream decode speed; tensor parallel scales bandwidth at ~85% per extra GPU."""
    bytes_per_token = m.active_b * QUANT_BYTES[quant]
    scale = 1 + 0.85 * (count - 1)
    return gpu.bandwidth_gbs * BW_EFFICIENCY * scale / bytes_per_token


def configure(
    model: str,
    quant: str,
    users: int,
    concurrency: float,
    context: int,
    tps_per_user: float,
    datacenter_only: bool = False,
    gpus: list[GPU] | None = None,
) -> list[dict]:
    if users < 1 or context < 1:
        raise ValueError("need users >= 1 and context >= 1 token")  # context 0 drops the KV cache from sizing
    if not (math.isfinite(concurrency) and 0 < concurrency <= 1):
        raise ValueError("concurrency is the share of users generating at once: 0 < concurrency <= 1")
    if not (math.isfinite(tps_per_user) and tps_per_user > 0):
        raise ValueError("tps_per_user must be a finite number > 0")
    m = MODELS[model]
    seqs = max(1, math.ceil(users * concurrency))
    need = vram_needed(m, quant, context, seqs)
    options = []
    for g in gpus or GPUS:
        if datacenter_only and not g.datacenter:
            continue
        for count in (1, 2, 4, 8):
            if g.vram_gb * count < need:
                continue
            # batching: aggregate throughput grows with concurrent sequences but saturates
            batch_gain = min(seqs, 1 + 0.6 * (seqs - 1)) if seqs > 1 else 1
            per_user = decode_tps(m, quant, g, count) * batch_gain / seqs
            if per_user < tps_per_user:
                continue
            options.append(
                {
                    "gpu": g.name,
                    "count": count,
                    "vram_total": g.vram_gb * count,
                    "vram_needed": round(need, 1),
                    "tps_per_user": round(per_user, 1),
                    "gpu_cost": g.price * count,
                    "sequences": seqs,
                }
            )
            break
    return sorted(options, key=lambda o: o["gpu_cost"])


def bom(option: dict, compliance: list[str] | None = None, margin: float = 1.35) -> dict:
    if not math.isfinite(margin) or margin <= 0:
        raise ValueError(f"margin must be a finite number > 0, got {margin!r}")
    n = option["count"]
    lines = [
        (f"{n} x {option['gpu']}", option["gpu_cost"]),
        (
            "Chassis + PSU" + (" (4U, 8-slot)" if n > 4 else " (4U, 4-slot)" if n > 1 else " (tower)"),
            1500 if n == 1 else 3500 if n <= 4 else 9000,
        ),
        (f"CPU ({max(16, 8 * n)} cores) + {max(128, 64 * n)} GB RAM", 1200 + 250 * n),
        (f"{max(2, n)} TB NVMe (models + RAG index)", 150 * max(2, n)),
        ("Line-interactive UPS", 600 if n <= 2 else 1500),
        ("Imaging, burn-in, on-site install", 2500),
    ]
    total = sum(p for _, p in lines)
    return {
        "lines": lines,
        "total": total,
        "suggested_price": round(total * margin, -2),
        "compliance": {c: COMPLIANCE[c] for c in (compliance or [])},
    }
