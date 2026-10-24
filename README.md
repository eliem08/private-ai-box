# private-ai-box

[![CI](https://github.com/eliem08/private-ai-box/actions/workflows/ci.yml/badge.svg)](https://github.com/eliem08/private-ai-box/actions/workflows/ci.yml)

> **Dell for private AI** — AI in a box for firms that won't use the cloud.

Law firms, clinics, defence suppliers and banks want LLMs but won't send data
to a cloud API. They need a box in the closet that just works. The hard part
of selling that box is sizing it: which model, which GPUs, how many users.
`private-ai-box` is the configurator that turns requirements into a bill of
materials.

## What the MVP does

- **Model catalogue** (8B / 32B / 70B dense classes + MoE) with layer and
  KV-head geometry
- **VRAM math**: weights at fp16 / int8 / int4 + KV cache
  (`2 × layers × kv_heads × head_dim × bytes × context × concurrent users`) + overhead
- **Throughput math**: decode is memory-bandwidth bound →
  `tokens/s ≈ bandwidth × efficiency / active_bytes_per_token`, shared across users
- **GPU catalogue** (workstation and datacenter cards, list-price placeholders)
- **Configurator**: cheapest GPU × count that fits VRAM *and* hits the
  per-user tokens/s target (tensor-parallel up to 8 GPUs)
- **BOM**: GPUs, chassis, CPU/RAM/NVMe sized from GPU count, UPS, plus a
  **compliance pack** checklist (air-gapped, HIPAA, SOC 2-style controls)

## Quickstart

```bash
pip install -e .
private-ai-box size --users 40 --concurrency 0.25 --model 70b --quant int4 --context 16000 --tps 15
private-ai-box models
private-ai-box gpus
```

Prices are placeholder list prices; update `GPUS` with your distributor's quotes.

## Roadmap

- Pre-imaged OS with model server, RAG over SMB shares, SSO, audit logging
- Remote health monitoring without data egress (metrics only)
- Leasing / financing calculator, 3-year TCO vs. API spend

## Development

Zero runtime dependencies; Python 3.10+.

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
python -m unittest discover -s tests -t .           # unit + CLI smoke tests
ruff check . && ruff format --check .               # lint + format
```

```
private_ai_box/core.py       domain logic (pure, importable, fully tested)
private_ai_box/__main__.py   CLI (`private-ai-box ...` / `python -m private_ai_box ...`)
tests/                unit tests + generated CLI smoke tests
```

CI runs lint and tests on Ubuntu and Windows across Python 3.10, 3.12 and 3.13.
Design, data model, trade-offs and production path: **[ARCHITECTURE.md](ARCHITECTURE.md)**.

Part of the [agent-economy](https://github.com/eliem08/agent-economy) portfolio (idea #17).
