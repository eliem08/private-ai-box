<!-- sell-kit:start -->
# Private AI Box

[![CI](https://github.com/eliem08/private-ai-box/actions/workflows/ci.yml/badge.svg)](https://github.com/eliem08/private-ai-box/actions/workflows/ci.yml)

> **Spec an on-prem AI server for firms that cannot use the cloud.**

**Who it's for:** Law, medical and finance firms, and the integrators who sell to them.

**The problem:** Regulated firms want private AI but can't size GPUs, VRAM and throughput, so they overspend or buy boxes that are too slow.

**What you get:** Size VRAM, KV cache and throughput for your users and model, pick the cheapest GPU configuration that fits, and output a bill of materials with compliance checklists.

## Quickstart

Customers receive a release archive with the wheel and the examples. Python 3.10+ and no other dependencies.

```bash
pip install private_ai_box-0.1.0-py3-none-any.whl
private-ai-box models
private-ai-box gpus
private-ai-box size --users 40 --concurrency 0.25 --model 32b --quant int4 --compliance hipaa
```

## Web app and API

Every command is also a JSON API endpoint and a browser form, served by the built-in web app.

```bash
private-ai-box-web --port 8000          # then open http://127.0.0.1:8000
private-ai-box-web keys create --label "first customer"   # prints an API key (stored hashed)
docker build -t private-ai-box . && docker run -p 8000:8000 -e API_KEYS=sk_... -v private-ai-box-data:/data private-ai-box
```

Every API call needs `Authorization: Bearer <key>` once any key exists; a server on a public address refuses to start without one. Calls are rate-limited per key (60/min, `--rate`) and metered: `private-ai-box-web usage` prints calls per key for billing.

Endpoints: `POST /api/size`, `POST /api/models`, `POST /api/gpus`, plus `GET /healthz` and `GET /api/spec` (machine-readable field list).
Files are sent as `{"name": ..., "content": ...}`, folders as `{"files": [...]}` or a base64 zip.

## Deploy and sell

[![Deploy to Render](https://render.com/images/deploy-to-render-button.svg)](https://render.com/deploy?repo=https://github.com/eliem08/private-ai-box)

`render.yaml` (one click, above) and `fly.toml` (`fly launch --copy-config`) both run the Dockerfile with a persistent `/data` volume and a `/healthz` check.
To sell self-serve, set four variables and point a Stripe webhook at `https://<your-host>/billing/webhook` (events `checkout.session.completed` and `customer.subscription.deleted`):

| Variable | Example |
|---|---|
| `STRIPE_SECRET_KEY` | `sk_live_...` |
| `STRIPE_WEBHOOK_SECRET` | `whsec_...` |
| `STRIPE_PRICES` | `team=price_123,business=price_456` |
| `PUBLIC_URL` | `https://private-ai-box.example.com` |

Buyers pick a plan on `/pricing`, pay on Stripe Checkout and see their API key once on the success page. Cancelling the subscription revokes the key. A selling server always requires a key.

## Pricing

Every plan starts with a 14-day free trial. Prices in USD, billed monthly unless noted.

| Plan | Price | Includes |
|---|---|---|
| **Integrator** | $99 per month | Unlimited configurations; BOM and price; Compliance packs |
| **Reseller** | $399 per month | Branded quotes; Custom hardware catalog; Margin controls |
| **Turnkey box** | Custom (quoted per build) | Built, imaged and installed on site; HIPAA and SOC 2 packs; Support contract |

To start a trial or buy a plan, [request access](https://github.com/eliem08/private-ai-box/issues/new?title=Private%20AI%20Box%20trial&labels=sales).

## Licence

Proprietary. Copyright (c) 2026 eliem08. All rights reserved. Use requires a paid plan or an active trial; see [LICENSE](LICENSE).
<!-- sell-kit:end -->

## What it does

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
