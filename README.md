# awedgerunner: run work on edge runners, under a lease

Read aloud: "Aither World Edgerunner".

Spare compute is everywhere: an idle gaming GPU, free CI minutes, a spot box, an edge
worker. Handing a contributor a slice of shared budget for it should not mean trusting
them not to overspend, remembering to switch it off, or finding out later whose bill it
landed on.

awedgerunner gives every piece of edge work a **lease**:

- **A hard cap.** A charge that would cross it is refused before anything runs. It is
  never clamped and never overdrawn.
- **An expiry.** Every call re-checks it, and a run is never allowed to outlive its lease.
- **A named payer.** There is no default payer and no fallback to someone else's
  account. A lane billed to one tenant refuses a lease paid by anyone else.
- **The pool owner's approval.** Each pool also has a per-lease cap and a total budget
  that open leases cannot overcommit.
- **Credentials by name, never by value.** Pooled API keys are allowed. A consumer chat
  subscription belongs to one person and is always refused.

```bash
pip install awedgerunner                 # Python 3.10+, no hard dependencies
awedgerunner --self-test                 # proves every refusal still fires
awedgerunner targets                     # which runner targets are usable here
awedgerunner pool create --id commons --owner me --funder commons-fund \
    --budget 20 --per-lease-cap 1
awedgerunner lease open --pool commons --contributor dev --approved-by me \
    --payer-kind pool --payer-id commons-fund --credential-kind pooled_api_key \
    --credential-ref POOL_API_KEY --cap 1 --ttl 1800 --targets awrun
awedgerunner run --lease <id> --target awrun --est-usd 0.4 --spec-json '{"task": "..."}'
awedgerunner run --lease <id> --target awrun --est-usd 0.7   # refused: crosses the cap
```

## Targets

awedgerunner owns the lease and the routing, not the runners. Each target hands the
work to something that already runs it, and refuses, with the reason, when that thing
is missing.

| target | hands work to | refuses when |
|---|---|---|
| `awrun` | the awrun queue | awrun is not installed |
| `ci-spot` | an awrun `ci` item that an ephemeral spot-runner lane scales to | the lane's config is missing, or the lease's payer is not the lane's tenant |
| `gpu-door` | a GPU lease door (preflighted), then an awrun GPU item | no door is configured, or the door does not say yes |
| `volunteer` | an awrun `agent` item pinned to a volunteer's node | the node's owner has not consented, or now is outside every consented window |
| `mesh-device` | an awrun item pinned to a device enrolled on your own mesh (phone, laptop, desk PC) | the device is off the roster, its declared capabilities (arch, RAM, GPU/NPU, power, model sizes, labels) do not match, its owner is using it, or its node decider does not say yes |
| `github-mirror` | (not built) | always: the public-runner boundary check is not written yet |
| `cloudflare` | (not built) | always: there is no Workers/Containers adapter yet |

Volunteer credit is recorded in a ledger in which every row is marked
`"simulated": true`. Nothing here settles real money.

State lives under `$AWEDGERUNNER_DIR` (default `~/.aither/awedgerunner`). It has no
dependencies; awrun is imported lazily (`pip install awedgerunner[awrun]`).

Exit codes: 0 ok, 1 refused, 2 could not run.
