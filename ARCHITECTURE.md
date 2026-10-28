# Architecture — private-ai-box

## 1. Scope
Configure, price and deliver on-prem LLM appliances for firms that won't use cloud AI.
**MVP:** sizing math (weights + KV cache + overhead), bandwidth-bound decode throughput, cheapest
GPU × count search, BOM and compliance packs.
**Product:** configurator web app, pre-imaged appliance OS, remote metrics-only monitoring, financing and TCO.

## 2. Sizing model
- **VRAM** = `(params × bytes_per_weight + KV) × 1.10`, where
  `KV = 2 × layers × kv_heads × head_dim × 2 B × context × concurrent_sequences` (fp16 KV, GQA-aware).
- **Decode tokens/s** ≈ `bandwidth × 0.6 / (active_params × bytes_per_weight)`, which is memory-bound.
  Tensor parallel scales at 85% per extra GPU. MoE uses *active* params, which is why a 120B MoE out-runs a 70B dense model.
- **Per-user tokens/s** = aggregate × batch gain (`1 + 0.6·(n−1)`, saturating) / n.
- Search: for each GPU, the smallest count in {1, 2, 4, 8} that satisfies both VRAM and per-user
  tokens/s, then sort by GPU cost.

## 3. Design decisions
- **First-principles formulas over benchmark lookup tables.** They generalise to new models and
  cards instantly, and the efficiency constants get calibrated against measured runs (roadmap:
  ship a calibration harness with the appliance).
- **Both constraints have to hold.** Cheap cards that fit VRAM but miss latency are excluded,
  because they generate support tickets.
- **Placeholder list prices, clearly labelled.** Real quotes come from distributor feeds.

## 4. Appliance architecture (product)
```mermaid
flowchart LR
  Users -->|SSO| Gateway[API gateway + audit log]
  Gateway --> Serve[vLLM / llama.cpp server]
  Gateway --> RAG[RAG service] --> Index[(Vector index on NVMe)]
  RAG --> Shares[SMB / SharePoint connectors]
  Mon[Metrics agent] -->|metrics only, no data| Vendor[Remote monitoring]
```
Hardening: LUKS + TPM, no WAN route from the model VLAN (air-gapped profile), and signed offline
update bundles.

## 5. Risks
Hardware supply and price volatility, so quotes expire after 7 days. Model licence terms for
commercial on-prem use are checked per model in the catalogue.

## 6. Limitations (sell-ready audit, 2026-10-28)
- No feature gating by plan: every API key unlocks the same software. Reseller's
  branded quotes are not built and were removed from sales copy; margin control
  is now real via `--margin` (default 1.35) on the BOM suggested price.
- GPU prices are placeholder list prices as of 2026-10-01 (27 days old at audit
  time, under the 90-day staleness warning); real quotes must come from a
  distributor price file via `--prices`, as in examples/reseller-prices.json.
  Sizing is first-principles math (bandwidth-bound decode, 0.6 efficiency,
  85% tensor-parallel scaling), not measured benchmarks for the buyer's exact
  model and cards.
- Docker build and a live Render/Fly deployment were not verified in the audit
  environment (no Docker binary there); Dockerfile, render.yaml, and fly.toml are
  the same template shape as vibe-lube, but a deployment claim remains
  "not verified" until one is made.
- Stripe prices were not verified against a live Stripe account in this run;
  checkout and revocation are covered in tests with mocked Stripe calls, and the
  README example now uses integrator/reseller plan keys to match the plan names.
