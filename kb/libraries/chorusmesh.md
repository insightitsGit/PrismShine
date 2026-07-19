# ChorusMesh

> Commercial ops layer on top of free `prismlib[fabric]`: Slack/PagerDuty alerting, escalation chains, custom health rules, Kafka/NATS durable transport for CHORUS frames.

| Field | Value |
|---|---|
| PyPI | `chorusmesh` |
| Version | 0.1.0 |
| License | Commercial (no OSS license file) |
| Python | >= 3.11 |
| Local path | `C:\code\ChorusMesh` |
| GitHub | https://github.com/insightitsGit/chorusmesh |
| Install | `pip install "chorusmesh[slack]"` |
| CLI | `chorusmesh-keygen` (issuer tooling) |

## Purpose

The paid add-on over PrismLib's free cluster (ClusterCache/SMTP/failover stay free in PrismLib). ChorusMesh adds enterprise alert channels (Slack, PagerDuty), multi-stage escalation chains, custom health rules, and durable message buses (Kafka/NATS) so no CHORUS frame is lost across service restarts.

## Architecture

| Module | Role |
|---|---|
| `chorusmesh.license` | Offline JWT (RS256) license + 30-day trial |
| `chorusmesh.alerts.slack` | `SlackAlerter` |
| `chorusmesh.alerts.pagerduty` | `PagerDutyAlerter` (+ `resolve`) |
| `chorusmesh.alerts.escalation` | `EscalationChain` / `EscalationRule` |
| `chorusmesh.alerts.custom_rules` | `CustomRuleSet` / `CustomRule` (restricted `eval` on health dict) |
| `chorusmesh.transport.kafka` | `KafkaTransport` (aiokafka) |
| `chorusmesh.transport.nats` | `NATSTransport` |
| `stripe_webhook/` | Separate FastAPI Stripe → license-key email automation (ops tooling, not the pip package core) |

**Important:** Raft consensus and multi-region/geo routing appear only as license feature *flags* (`raft`, `multi_region`, `geo_routing`) and marketing copy — no implementation in this tree. The implemented surface is alerts + Kafka/NATS + licensing.

## Public API

```python
from chorusmesh.license import load_license, require, LicenseInfo, LicenseError
# require("slack")  -> raises LicenseError if feature not licensed

from chorusmesh.alerts import SlackAlerter, PagerDutyAlerter, EscalationChain, EscalationRule
SlackAlerter(webhook_url, channel="#alerts", username="ChorusMesh")
    async .send(level, event_type, title, message, data, node_id)
PagerDutyAlerter(integration_key)
    async .send(...); async .resolve(node_id, event_type)
EscalationRule(level, channel, wait_minutes=0)
EscalationChain(rules=[...])
    async .fire(...); .acknowledge(node_id, event_type)

CustomRule(name, expression, level="warning", message="")
CustomRuleSet(rules).evaluate(health: dict) -> list[(event_type, level, message)]

from chorusmesh.transport import KafkaTransport, NATSTransport
KafkaTransport(bootstrap_servers, topic="chorusmesh-frames", group_id="chorusmesh")
    async .connect / .publish(frame_type, payload, source_node="", seq=0) / .subscribe(on_frame) / .disconnect
NATSTransport(servers="nats://localhost:4222", subject="chorusmesh.frames")  # same pattern
```

## Core logic

- **License**: JWT claims (`tier`, `nodes`, `features`) verified against bundled RSA public key; no phone-home. Missing key → 30-day Developer trial file `~/.chorusmesh_trial`. `require(feature)` gates paid constructors.
- **Escalation**: sequential async stages with `wait_minutes` sleeps; `acknowledge` cancels the pending task.
- **Custom rules**: `eval(expression, {"__builtins__": {}}, health)` — restricted eval over the health snapshot.
- **Transport**: JSON CHORUS-like frames `{frame_type, source_node, seq, ts, payload}` over Kafka or NATS; subscribers skip frames from their own `source_node`.

## Dependencies

- Core: **`prismlib>=0.4.0`** (the only Insight lib with a hard pip dep on a sibling), `pyjwt[crypto]>=2.8`, `cryptography>=42`, `httpx>=0.27`, `psutil>=5.9`
- Extras: `[slack]`, `[pagerduty]`, `[opsgenie]`, `[kafka]`, `[nats]`, `[history-pg]`, `[stripe-webhook]`, `[all]`
- Integrates with `prism.cluster.alerts.AlertManager` from PrismLib at runtime

## Config

Env: `CHORUSMESH_LICENSE_KEY`; `SLACK_WEBHOOK_URL`; `PAGERDUTY_KEY` / `PAGERDUTY_INTEGRATION_KEY`; Stripe env for the webhook tooling.

## Usage example

```python
import os
from chorusmesh.alerts import SlackAlerter
from prism.cluster.alerts import AlertManager

slack = SlackAlerter(webhook_url=os.getenv("SLACK_WEBHOOK_URL"), channel="#ops-alerts")
alerts = AlertManager(fabric=chorus_fabric, extra_channels=[slack])
await alerts.evaluate_health(health_snapshot)
```

## Tests / benchmarks

- **None.** No `tests/` directory and zero `test_*.py` files despite pytest config in pyproject. No benchmarks.

## Gotchas

- Marketing (Raft, multi-region) is ahead of the code — only alerts, transport, and licensing exist today.
- Fully commercial; tiers Developer / Team / Business / Enterprise with pricing in README.
- The lowest-test-coverage library in the ecosystem — treat with care in production designs.
