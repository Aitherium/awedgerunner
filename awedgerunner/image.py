"""The IMAGE job class: batch-shaped image generation across the edge pool.

Shape: one model load, many prompts per runner. This is the swarm pattern for
text, applied to images: one small model per runner, fan out. A job carries ONE
ComfyUI graph that is the same graph locally and on Comfy Cloud, plus a list of
prompts.

Routing rule (owner directive 2026-11-15). If the lease is the customer's own
key (BYOK), ONLY BYOK targets are tried, cheapest measured cost first, and
nothing falls back onto our infrastructure. Otherwise our own targets are tried
in order:

* BYOK: ``cf-workers-ai`` (Workers AI image models on the customer's Cloudflare
  account), ``comfy-cloud`` (the customer's Comfy Cloud key), ``byok-spot``
  (vast.ai Serverless or Lambda on-demand, on the customer's account).
* ours: ``gpu-door`` (the local GPU through the lease door, never around it), then
  ``volunteer`` (community GPUs), then ``mesh-device`` (the owner's devices that
  declare a GPU), then ``cpu-swarm`` (free hosted CPU runners), which is ranked
  after every GPU target.

``cpu-swarm`` runs ONLY models in ``CPU_PROVEN_MODELS``, each with its measured
benchmark and licence basis. The ternary 4B image model is CUDA-only and is not
in that set.

ETA comes from MEASURED throughput only. A target with no measured rate
contributes nothing, and an empty pool gives ``None`` (unknown), never a guess.
"""

from __future__ import annotations

import math
from typing import Any, Callable, Dict, List, Optional, Tuple

from .lease import Lease
from .targets import Ctx, Job, TargetRefused

IMAGE_OURS_ORDER = ("gpu-door", "volunteer", "mesh-device", "cpu-swarm")
IMAGE_BYOK_TARGETS = ("cf-workers-ai", "comfy-cloud", "byok-spot")

#: Cost per 320 px sticker (one 512x512 tile), from the providers' own price
#: pages, read 2026-11-15. None = not measured; it sorts after every measured
#: cost. Each value carries its source so it can be re-checked.
COST_PER_STICKER_USD: Dict[str, Dict[str, Any]] = {
    "cf-workers-ai": {"usd": 0.000287, "model": "@cf/black-forest-labs/flux-2-klein-4b",
                      "source": "developers.cloudflare.com/workers-ai/platform/pricing: "
                                "$0.000287 per output 512x512 tile, 4 fixed steps"},
    "comfy-cloud": {"usd": None, "source": "not measured yet"},
    "byok-spot": {"usd": None, "source": "not measured yet (depends on the rented GPU)"},
}

#: Image models a recorded benchmark has shown usable on a CPU runner. Each entry
#: carries its numbers, its licence basis and the attribution that must travel with
#: every output pack. The benchmark was measured 2026-11-15 on a free 4 vCPU /
#: 16 GB hosted runner (Aitherium/awrun bonsai-swarm mode=image-bench, run
#: 43759864209). Both sd-turbo CPU paths were rejected on quality.
SANA_SPRINT_NOTICE = (
    "Generated with Sana Sprint 0.6B (NVIDIA, Efficient-Large-Model/"
    "Sana_Sprint_0.6B_1024px_diffusers), licensed under the Apache License 2.0 "
    "(https://www.apache.org/licenses/LICENSE-2.0). Text encoder: Gemma 2 "
    "(google/gemma-2-2b-it), provided under and subject to the Gemma Terms of Use "
    "(https://ai.google.dev/gemma/terms) and the Gemma Prohibited Use Policy "
    "(https://ai.google.dev/gemma/prohibited_use_policy).")
CPU_PROVEN_MODELS: Dict[str, Dict[str, Any]] = {
    "sana-sprint-0.6b": {
        "repo": "Efficient-Large-Model/Sana_Sprint_0.6B_1024px_diffusers",
        "steps": 2,
        "s_per_image": {"320": 16.9, "512": 16.9},
        "load_s": 85,
        "peak_rss_mb": 8557,
        "runner": "GitHub-hosted ubuntu-latest, 4 vCPU / 16 GB",
        "benchmark_run": "https://github.com/Aitherium/awrun/actions/runs/43759864209",
        # Licence basis (ruled 2026-11-15). The weights are apache-2.0 (the model
        # card's cardData.license). The card's "research purposes only" is an
        # intended-use statement, not a licence term. The Gemma-2-2B-IT text
        # encoder is under the Gemma Terms of Use + Prohibited Use Policy, which
        # permit commercial use; the moderation screen covers the prohibited uses.
        "licence": "apache-2.0 (weights) + Gemma Terms of Use (text encoder)",
        "licence_links": ["https://huggingface.co/Efficient-Large-Model/"
                          "Sana_Sprint_0.6B_1024px_diffusers",
                          "https://www.apache.org/licenses/LICENSE-2.0",
                          "https://ai.google.dev/gemma/terms",
                          "https://ai.google.dev/gemma/prohibited_use_policy"],
        "notice": SANA_SPRINT_NOTICE,
    },
}

#: Measured CPU-swarm throughput per runner, for the ETA: 60 / 16.9 s.
CPU_SWARM_IMAGES_PER_MIN = round(60.0 / 16.9, 2)

#: The most prompts one runner takes per model load; past this, split the job.
BATCH_MAX = 64


def byok_order() -> List[str]:
    def key(name: str) -> Tuple[int, float, int]:
        usd = COST_PER_STICKER_USD.get(name, {}).get("usd")
        return (0 if usd is not None else 1, usd or 0.0, IMAGE_BYOK_TARGETS.index(name))
    return sorted(IMAGE_BYOK_TARGETS, key=key)


def is_byok(lease: Lease) -> bool:
    return lease.payer_kind == "byo" and lease.credential_kind == "byo_key"


def validate_image_job(job: Job) -> Optional[str]:
    if job.kind != "image":
        return "not an image job"
    graph = job.spec.get("graph")
    if not isinstance(graph, dict) or not graph:
        return "an image job carries its ComfyUI graph (spec.graph, the API-format dict)"
    prompts = job.spec.get("prompts")
    if not isinstance(prompts, list) or not prompts:
        return "an image job carries a non-empty spec.prompts list (one model load, many prompts)"
    if len(prompts) > BATCH_MAX:
        return f"{len(prompts)} prompts exceeds BATCH_MAX={BATCH_MAX}; split the job"
    return None


def split_batches(prompts: List[str], per_runner: int) -> List[List[str]]:
    """Fan out: one batch per runner, each one model load."""
    if per_runner <= 0:
        raise ValueError("per_runner must be > 0")
    return [prompts[i:i + per_runner] for i in range(0, len(prompts), per_runner)]


#: BYOK target -> provider keys in the platform's referral registry
#: (``referral.<provider>`` in config/aitherium-economics.yaml, read by the host's
#: lib/compute/referrals.py ``load_referrals``).
REFERRAL_PROVIDER = {"cf-workers-ai": ("cloudflare",), "comfy-cloud": ("comfy_cloud",),
                     "byok-spot": ("vastai", "lambda")}


def _signup(name: str, referrals: Optional[Dict[str, Dict[str, Any]]]) -> List[Dict[str, str]]:
    """Sign-up links for a BYOK provider the customer has no account on. Only a
    registry row marked ``configured`` with a URL is offered, with its disclosure
    line. An unconfigured row is left out: never a guessed link."""
    out: List[Dict[str, str]] = []
    for provider in REFERRAL_PROVIDER.get(name, ()):
        row = (referrals or {}).get(provider) or {}
        if row.get("configured") and str(row.get("url", "")).startswith("https://"):
            out.append({"provider": provider, "url": str(row["url"]),
                        "disclosure": str(row.get("disclosure", ""))})
    return out


def select_image_target(job: Job, lease: Lease, targets: Dict[str, Any], ctx: Ctx,
                        referrals: Optional[Dict[str, Dict[str, Any]]] = None
                        ) -> Tuple[Optional[Any], List[Dict[str, Any]]]:
    """Return ``(target, trail)``.

    A BYOK lease tries only BYOK targets, cheapest measured cost first, and never
    falls back onto ours. Any other lease tries ours in ``IMAGE_OURS_ORDER``.
    ``target`` is None when nothing may run the job, and the trail names every
    target tried and why it was skipped. A skipped BYOK target carries the
    provider's sign-up link from ``referrals`` (the host's referral registry,
    ``{provider: {configured, url, disclosure}}``) when one is configured."""
    why = validate_image_job(job)
    if why:
        return None, [{"target": "*", "skipped": why}]
    order = byok_order() if is_byok(lease) else list(IMAGE_OURS_ORDER)
    trail: List[Dict[str, Any]] = []
    for name in order:
        t = targets.get(name)
        if t is None:
            trail.append({"target": name, "skipped": "not configured"})
            continue
        if name not in lease.targets:
            trail.append({"target": name, "skipped": "not allowed by the lease"})
            continue
        reason = t.check(job, lease, ctx)
        if reason:
            row: Dict[str, Any] = {"target": name, "skipped": reason}
            links = _signup(name, referrals) if is_byok(lease) else []
            if links:
                row["signup"] = links
            trail.append(row)
            continue
        trail.append({"target": name, "chosen": "first target whose check passed"})
        return t, trail
    return None, trail


def _byok_check(lease: Lease, what: str) -> Optional[str]:
    if not is_byok(lease):
        return (f"{what} runs only on the customer's own key, paid by them "
                "(payer_kind=byo, credential_kind=byo_key); no platform key, no fallback")
    if not lease.credential_ref.startswith("workspace:"):
        return f"the {what} key must be a workspace Lockbox ref ('workspace:<id>/<NAME>')"
    return None


class _ByokImageTarget:
    """Shared shape. ``client`` is the HOST's submitter. It resolves the key from
    the customer's workspace Lockbox at call time; this class only ever holds the
    reference. Without a client the target is unavailable. A platform-side client
    is never substituted."""

    name = ""
    what = ""
    unwired = ""

    def __init__(self, client: Optional[Callable[..., Dict[str, Any]]] = None) -> None:
        self._client = client

    def available(self) -> Tuple[bool, str]:
        if self._client is None:
            return False, self.unwired
        return True, f"{self.what} (customer's own account)"

    def check(self, job: Job, lease: Lease, ctx: Ctx) -> Optional[str]:
        byok = _byok_check(lease, self.what)
        if byok:
            return byok
        why = validate_image_job(job)
        if why:
            return why
        ok, reason = self.available()
        return None if ok else reason

    def _call(self, job: Job, lease: Lease) -> Dict[str, Any]:
        raise NotImplementedError

    def submit(self, job: Job, lease: Lease, ctx: Ctx) -> Dict[str, Any]:
        if self._client is None:
            raise TargetRefused(self.unwired)
        out = dict(self._call(job, lease))
        out.setdefault("via", self.name)
        return out


class ComfyCloudTarget(_ByokImageTarget):
    """Comfy Cloud, BYOK only. The client receives the SAME ComfyUI graph the
    local GPU runs: ``client(graph, prompts, key_ref)``."""

    name = "comfy-cloud"
    what = "Comfy Cloud"
    unwired = ("no Comfy Cloud client wired; the host supplies one that resolves the "
               "customer's key from their workspace Lockbox")

    def _call(self, job: Job, lease: Lease) -> Dict[str, Any]:
        assert self._client is not None
        return self._client(job.spec["graph"], list(job.spec["prompts"]), lease.credential_ref)


class CloudflareWorkersAITarget(_ByokImageTarget):
    """Workers AI image models on the CUSTOMER'S Cloudflare account.

    The diffusion runs on Cloudflare's GPUs behind the Workers AI API. A Worker's
    or a Container's own CPU cannot run diffusion, so this is never a "run it in a
    Worker" target. ``client(model, prompts, size, key_ref)`` is the host's caller:
    either the account's REST API with the customer's token, or a Worker binding
    served on an aitherium.com subdomain (never a personal workers.dev URL)."""

    name = "cf-workers-ai"
    what = "Workers AI"
    unwired = ("no Workers AI client wired (the customer's account REST API, or a Worker "
               "binding on an aitherium.com subdomain)")
    #: Models whose price was read from the Workers AI pricing page on 2026-11-15.
    MODELS = ("@cf/black-forest-labs/flux-2-klein-4b", "@cf/black-forest-labs/flux-1-schnell",
              "@cf/black-forest-labs/flux-2-klein-9b")

    def __init__(self, client: Optional[Callable[..., Dict[str, Any]]] = None,
                 model: str = "@cf/black-forest-labs/flux-2-klein-4b") -> None:
        super().__init__(client)
        self.model = model

    def check(self, job: Job, lease: Lease, ctx: Ctx) -> Optional[str]:
        if self.model not in self.MODELS:
            return f"model {self.model!r} is not one whose price was read from the docs"
        return super().check(job, lease, ctx)

    def _call(self, job: Job, lease: Lease) -> Dict[str, Any]:
        assert self._client is not None
        return self._client(self.model, list(job.spec["prompts"]),
                            int(job.spec.get("size", 320)), lease.credential_ref)


class ByokSpotTarget(_ByokImageTarget):
    """Spot or serverless GPUs on the CUSTOMER'S account.

    * vast.ai Serverless: an endpoint billed per execution, according to vast's
      docs read 2026-11-15.
    * Lambda on-demand: VMs only. Its on-demand docs list no serverless product.

    The host's client delegates to the existing GPU tools (``vast_selfserve``,
    ``aither_gpu_*``) on the customer's key: ``client(graph, prompts, key_ref)``."""

    name = "byok-spot"
    what = "a BYOK spot GPU"
    unwired = ("no BYOK spot client wired (vast.ai Serverless or Lambda on the customer's "
               "account, through the existing GPU tools)")

    def _call(self, job: Job, lease: Lease) -> Dict[str, Any]:
        assert self._client is not None
        return self._client(job.spec["graph"], list(job.spec["prompts"]), lease.credential_ref)


class CpuSwarmTarget:
    """Free hosted CPU runners, one model load per runner, fanned out like the text
    swarm. Only ``CPU_PROVEN_MODELS`` run here (job ``spec.model``).

    The host supplies ``dispatch(model, batches, notice) -> {"handle": ...}``. It
    starts the swarm workflow, one job per batch, and must write ``notice`` into
    every output pack's metadata. Without a dispatcher the target is unavailable.
    Every result this target returns carries the licence ``attribution``."""

    name = "cpu-swarm"

    def __init__(self, dispatch: Optional[Callable[..., Dict[str, Any]]] = None,
                 per_runner: int = 8, max_runners: int = 32) -> None:
        self._dispatch = dispatch
        self.per_runner = per_runner
        self.max_runners = max_runners

    def available(self) -> Tuple[bool, str]:
        if self._dispatch is None:
            return False, "no CPU swarm dispatcher wired (the host starts the swarm workflow)"
        return True, (f"CPU swarm: {sorted(CPU_PROVEN_MODELS)}, "
                      f"{CPU_SWARM_IMAGES_PER_MIN} images/min per runner (measured)")

    def check(self, job: Job, lease: Lease, ctx: Ctx) -> Optional[str]:
        why = validate_image_job(job)
        if why:
            return why
        model = job.spec.get("model", "")
        if model not in CPU_PROVEN_MODELS:
            return (f"model {model!r} is not CPU-proven; cpu-swarm runs only "
                    f"{sorted(CPU_PROVEN_MODELS)}")
        batches = split_batches(list(job.spec["prompts"]), self.per_runner)
        if len(batches) > self.max_runners:
            return f"{len(batches)} runners needed, max {self.max_runners}"
        ok, reason = self.available()
        return None if ok else reason

    def submit(self, job: Job, lease: Lease, ctx: Ctx) -> Dict[str, Any]:
        if self._dispatch is None:
            raise TargetRefused(self.available()[1])
        meta = CPU_PROVEN_MODELS[job.spec["model"]]
        batches = split_batches(list(job.spec["prompts"]), self.per_runner)
        out = dict(self._dispatch(job.spec["model"], batches, meta["notice"]))
        out.update(via=self.name, runners=len(batches), attribution=meta["notice"],
                   licence=meta["licence"])
        return out

    def capacity(self, runners: int) -> Dict[str, Any]:
        """A measured capacity row for ``eta_seconds``."""
        ok, _ = self.available()
        slots = max(0, min(runners, self.max_runners))
        return {"available": ok and slots > 0, "images_per_min": CPU_SWARM_IMAGES_PER_MIN,
                "slots": slots}


def eta_seconds(queued_images: int, capacity: Dict[str, Dict[str, Any]]) -> Optional[float]:
    """Seconds to drain ``queued_images`` across the pool's MEASURED capacity.

    ``capacity`` maps target -> ``{"images_per_min": float, "slots": int,
    "available": bool}``. Unmeasured or unavailable targets contribute nothing.
    Returns None when nothing measured is available: the honest answer is
    "unknown", and a queue page must say so rather than invent a number."""
    if queued_images <= 0:
        return 0.0
    rate = 0.0
    for row in capacity.values():
        ipm = row.get("images_per_min")
        if not row.get("available") or not isinstance(ipm, (int, float)) or ipm <= 0:
            continue
        rate += float(ipm) * max(1, int(row.get("slots", 1)))
    if rate <= 0:
        return None
    return float(math.ceil(queued_images / rate * 60.0))
