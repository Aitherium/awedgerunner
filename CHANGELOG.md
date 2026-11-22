# Changelog

All notable changes to `awedgerunner` are recorded here.

## [0.1.0] - 2026-11-15

First public release, published to PyPI as `awedgerunner`. It is the lease and the
router for edge work; every runner target delegates and refuses honestly when the
thing it delegates to is absent, so the package needs nothing beyond the standard
library to install and run.

### Added

- **awedgerunner**: a lease for edge work. Each lease has a hard budget cap (a charge that would cross it is refused, never clamped), a named payer (there is no default payer and no fallback), approval by the pool owner, and an expiry that is checked on every call.
- **awedgerunner**: pools. Each pool has a total budget, a per-lease cap and a maximum lifetime, and open leases cannot overcommit it. Credentials are named by reference, never by value. A consumer chat subscription is refused, and a pooled API key must be paid by the pool.
- **awedgerunner**: runner targets that delegate instead of running anything themselves.
  - `awrun` queues the job, and the job may not outlive its lease.
  - `ci-spot` submits to an owner-billed spot lane, and refuses any other payer.
  - `gpu-door` preflights a GPU lease door and fails closed.
  - `volunteer` needs the owner's consent plus a weekly schedule window, and records credit that is always simulated.
  - `mesh-device` sends work to a device on your own mesh. The job must match the device's declared capabilities, the device is preempted while its owner uses it, and an optional on-device decider must say yes.
  - `github-mirror` and `cloudflare` refuse with the reason.
- **awedgerunner**: an `image` job class. A job is one ComfyUI graph and many prompts, loaded once per runner.
  - A lease on the customer's own key tries only that customer's targets, cheapest measured price first: Workers AI image models, Comfy Cloud, and spot or serverless GPUs. It never falls back to ours.
  - Any other lease goes through the local GPU lease door first, then volunteer GPUs, then mesh devices.
  - The ETA comes only from measured capacity. When nothing is measured it is unknown.
  - A `cpu-swarm` target runs Sana Sprint 0.6B on free hosted CPU runners. It was measured at 16.9 s per image and ranks after every GPU target. Every result carries the Apache-2.0 and Gemma attribution.
- **awedgerunner**: CLI `targets`, `pool`, `consent`, `lease`, `run`, `credit`, `doctor`, and `--self-test`, which checks that every refusal still fires.
