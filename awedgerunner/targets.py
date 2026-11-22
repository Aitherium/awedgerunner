"""Runner targets: where leased work goes. Every adapter DELEGATES.

awedgerunner owns the lease (cap, expiry, payer, consent). It does not own
any runner. Each adapter hands the job to the piece that already runs that
kind of work, and refuses -- with a reason -- when that piece is absent or
the lease's rules forbid it:

=============  ==========================================================
target         delegates to
=============  ==========================================================
awrun          awrun's queue (``awrun.store.RunStore.submit``)
ci-spot        an awrun ``ci`` item; the ci-spot lane scales to its label
gpu-door       a GPU lease door (awrun's client), preflighted, fail closed
volunteer      an awrun ``agent`` item pinned to a consenting peer
mesh-device    an awrun item for a device on the owner's mesh roster
github-mirror  UNAVAILABLE: the public-runner boundary check is unwritten
cloudflare     UNAVAILABLE: no Workers/Containers runner adapter yet
=============  ==========================================================

Interface (duck-typed)::

    name: str
    available() -> (bool, reason)
    check(job, lease, ctx) -> None | reason      # BEFORE any charge
    submit(job, lease, ctx) -> dict              # {"handle": ..., ...}
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from .lease import Lease


@dataclass
class Job:
    kind: str = "agent"
    spec: Dict[str, Any] = field(default_factory=dict)
    est_usd: float = 0.0
    priority: int = 0
    peer: str = ""
    credit_units: float = 0.0
    gpu: Optional[Dict[str, Any]] = None


@dataclass
class Ctx:
    now: float
    store: Any  # awedgerunner.store.Store


class TargetRefused(RuntimeError):  # noqa: N818 - a refusal, not an error
    """The target said no. Nothing ran and nothing is charged."""


def _awrun_store(path: Optional[str] = None) -> Any:
    try:
        from awrun.store import RunStore  # lazy: awrun is a pairing, not a need
    except ImportError as e:
        raise TargetRefused("awrun is not installed (pip install awrun); "
                            "this target delegates to its queue") from e
    return RunStore(path) if path else RunStore()


def _awrun_spec(job: Job, lease: Lease, target: str) -> Dict[str, Any]:
    spec = dict(job.spec)
    spec["awedgerunner"] = {"lease": lease.id, "target": target,
                            "payer": f"{lease.payer_kind}:{lease.payer_id}",
                            "expires_at": lease.expires_at}
    return spec


def _limits(lease: Lease, ctx: Ctx) -> Dict[str, Any]:
    # A run may not outlive its lease: awrun enforces timeout_s and fails the
    # run if it cannot, which is the behaviour a lease needs.
    return {"timeout_s": max(1, lease.remaining_s(ctx.now))}


class AwrunTarget:
    name = "awrun"

    def __init__(self, store_factory: Optional[Callable[[], Any]] = None) -> None:
        self._factory = store_factory or _awrun_store

    def available(self) -> Tuple[bool, str]:
        try:
            self._factory()
        except TargetRefused as e:
            return False, str(e)
        return True, "awrun queue"

    def check(self, job: Job, lease: Lease, ctx: Ctx) -> Optional[str]:
        return None

    def _submit(self, kind: str, job: Job, lease: Lease, ctx: Ctx,
                gpu: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        store = self._factory()
        item = store.submit(kind, _awrun_spec(job, lease, self.name),
                            priority=job.priority, gpu=gpu, limits=_limits(lease, ctx))
        return {"handle": item.id, "via": "awrun", "kind": kind}

    def submit(self, job: Job, lease: Lease, ctx: Ctx) -> Dict[str, Any]:
        return self._submit(job.kind, job, lease, ctx, gpu=job.gpu)


def _read_scalars(path: Path) -> Dict[str, str]:
    """Top-level ``key: value`` scalars of a YAML file, stdlib only."""
    out: Dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        m = re.match(r"^([A-Za-z_][A-Za-z0-9_]*):\s*([^#\[{]+?)\s*(#.*)?$", line)
        if m:
            out[m.group(1)] = m.group(2).strip().strip("'\"")
    return out


class CiSpotTarget(AwrunTarget):
    """The ci-spot lane is OWNER-BILLED (one tenant, one daily cap). A lease
    paid by anyone else must not land there: that would be exactly the silent
    fallback to someone else's bill the BYO-pay rule forbids."""

    name = "ci-spot"

    def __init__(self, config_path: Optional[str] = None,
                 store_factory: Optional[Callable[[], Any]] = None) -> None:
        super().__init__(store_factory)
        self.config_path = config_path or os.environ.get("AWEDGERUNNER_CI_SPOT_CONFIG", "")

    def _lane(self) -> Dict[str, str]:
        if not self.config_path or not Path(self.config_path).is_file():
            raise TargetRefused("ci-spot lane config not found (set "
                                "AWEDGERUNNER_CI_SPOT_CONFIG to ci_spot_lane.yaml)")
        lane = _read_scalars(Path(self.config_path))
        for key in ("tenant_id", "label", "daily_cap_usd", "enabled"):
            if key not in lane:
                raise TargetRefused(f"ci-spot lane config has no {key!r}; refusing to guess")
        return lane

    def available(self) -> Tuple[bool, str]:
        try:
            lane = self._lane()
        except TargetRefused as e:
            return False, str(e)
        if lane["enabled"].lower() != "true":
            return False, "ci-spot lane is disabled"
        return True, (f"label {lane['label']}, billed to {lane['tenant_id']}, "
                      f"cap ${lane['daily_cap_usd']}/day")

    def check(self, job: Job, lease: Lease, ctx: Ctx) -> Optional[str]:
        ok, why = self.available()
        if not ok:
            return why
        lane = self._lane()
        if lease.payer_kind != "byo" or lease.payer_id != lane["tenant_id"]:
            return (f"ci-spot is billed to tenant {lane['tenant_id']!r}; this lease is paid by "
                    f"{lease.payer_kind}:{lease.payer_id} (BYO-pay: no fallback onto "
                    "another payer's lane)")
        if not job.spec.get("workflow"):
            return "a ci job needs spec.workflow"
        if job.est_usd > float(lane["daily_cap_usd"]):
            return "the job's estimate exceeds the lane's whole daily cap"
        return None

    def submit(self, job: Job, lease: Lease, ctx: Ctx) -> Dict[str, Any]:
        return self._submit("ci", job, lease, ctx)


class GpuDoorTarget(AwrunTarget):
    """GPU work through a lease door. awrun runs a job UNLEASED when the door is
    unreachable; an edge lease must not, so this target PREFLIGHTS the door
    (acquire + release) and refuses when it does not answer yes."""

    name = "gpu-door"

    def __init__(self, lease_client: Any = None,
                 store_factory: Optional[Callable[[], Any]] = None) -> None:
        super().__init__(store_factory)
        self._client = lease_client

    def _door(self) -> Any:
        if self._client is not None:
            return self._client
        try:
            from awrun import gpu_lease
        except ImportError as e:
            raise TargetRefused("awrun is not installed; the GPU door client is awrun's") from e
        return gpu_lease

    def available(self) -> Tuple[bool, str]:
        try:
            door = self._door()
        except TargetRefused as e:
            return False, str(e)
        bases = getattr(door, "bases", lambda: [])()
        if not bases:
            return False, "no GPU lease door configured (AITHER_GPU_LEASE_BASE); no default door"
        return True, "door " + ", ".join(bases)

    def check(self, job: Job, lease: Lease, ctx: Ctx) -> Optional[str]:
        ok, why = self.available()
        if not ok:
            return why
        if not job.gpu or "class" not in job.gpu or "vram_mb" not in job.gpu:
            return "a gpu-door job needs gpu={'class': ..., 'vram_mb': ...}"
        door = self._door()
        ttl = min(int(job.gpu.get("ttl_s", 600)), lease.remaining_s(ctx.now))
        try:
            grant = door.acquire(job.gpu["class"], int(job.gpu["vram_mb"]), ttl_s=max(1, ttl),
                                 job_ref=lease.id, consumer_id=f"awedgerunner:{lease.id}")
        except Exception as e:  # LeaseRefused / LeaseUnavailable / transport
            return f"GPU door did not grant ({type(e).__name__}: {e}); not running unleased"
        door.release(grant, "preflight")
        return None

    def submit(self, job: Job, lease: Lease, ctx: Ctx) -> Dict[str, Any]:
        gpu = dict(job.gpu or {})
        gpu["ttl_s"] = min(int(gpu.get("ttl_s", 600)), max(1, lease.remaining_s(ctx.now)))
        gpu_kinds = ("render", "artpack", "solve", "blender-frame")
        # An image batch is a render: one model load, many prompts.
        kind = job.kind if job.kind in gpu_kinds else ("render" if job.kind == "image" else "solve")
        return self._submit(kind, job, lease, ctx, gpu=gpu)


class VolunteerTarget(AwrunTarget):
    """A volunteer's node. Default-deny: the peer's owner must have consented
    to this purpose, the lease must be live, and now must fall in a consented
    schedule window. Credit earned is recorded SIMULATED."""

    name = "volunteer"
    purpose = "community_inference"

    def check(self, job: Job, lease: Lease, ctx: Ctx) -> Optional[str]:
        if not job.peer:
            return "a volunteer job names its peer"
        consent = ctx.store.consents().get(job.peer)
        if consent is None:
            return f"peer {job.peer!r} has no owner consent on record (default-deny)"
        return consent.allows(self.purpose, ctx.now)

    def submit(self, job: Job, lease: Lease, ctx: Ctx) -> Dict[str, Any]:
        job = Job(kind="agent", spec=dict(job.spec, peer=job.peer), est_usd=job.est_usd,
                  priority=job.priority, peer=job.peer, credit_units=job.credit_units)
        out = self._submit("agent", job, lease, ctx)
        ctx.store.append_credit({"at": round(ctx.now, 3), "lease": lease.id, "peer": job.peer,
                                 "handle": out["handle"], "credit_units": job.credit_units})
        out["credit"] = "simulated"
        return out


def _roster_from_file() -> List[Dict[str, Any]]:
    """The mesh roster as exported to ``$AWEDGERUNNER_MESH_ROSTER`` (a JSON list
    of the device records that enrollment through awnode/awdk already wrote).
    No roster = no devices: this module never enrolls anything itself."""
    raw = os.environ.get("AWEDGERUNNER_MESH_ROSTER", "").strip()
    if not raw or not Path(raw).is_file():
        raise TargetRefused("no mesh roster (set AWEDGERUNNER_MESH_ROSTER to the exported "
                            "roster of enrolled devices); awedgerunner does not enroll")
    import json

    data = json.loads(Path(raw).read_text(encoding="utf-8"))
    return data if isinstance(data, list) else list(data.get("devices", []))


#: Capability keys a mesh device declares in its roster record.
DEVICE_KEYS = ("peer", "arch", "ram_mb", "accel", "on_battery", "charging", "models",
               "labels", "owner_active")


def device_fit(device: Dict[str, Any], needs: Dict[str, Any]) -> Optional[str]:
    """None when ``device`` may take a job that ``needs`` this, else why not.
    Pure: every input is the device's DECLARED capability record."""
    if not device.get("enrolled", False):
        return "device is not enrolled on the mesh"
    if device.get("owner_active", True):
        # Default True on purpose: a device that does not say its owner is away
        # is treated as in use. The owner always wins.
        return "the owner is using the device (preempted)"
    if device.get("on_battery", True) and not device.get("charging", False):
        return "device is on battery and not charging"
    labels = set(device.get("labels") or [])
    missing = sorted(set(needs.get("labels") or []) - labels)
    if missing:
        return f"device lacks labels {missing}"
    if needs.get("arch") and needs["arch"] != device.get("arch"):
        return f"job needs arch {needs['arch']}, device is {device.get('arch')}"
    if int(needs.get("min_ram_mb", 0)) > int(device.get("ram_mb", 0)):
        return f"job needs {needs['min_ram_mb']} MB RAM, device declares {device.get('ram_mb')}"
    if needs.get("accel") and needs["accel"] != device.get("accel"):
        return f"job needs accel {needs['accel']}, device has {device.get('accel')}"
    if needs.get("model") and needs["model"] not in (device.get("models") or []):
        return f"device cannot run model {needs['model']!r}"
    return None


class MeshDeviceTarget(AwrunTarget):
    """A device enrolled on the owner's mesh (a phone, a laptop, a desk PC) as an
    awrun runner. It takes only jobs its DECLARED capabilities match (arch, RAM,
    GPU/NPU, power, which Bonsai sizes it can run, labels), and is preempted the
    moment its owner uses it.

    Reuses what exists: the device record comes from the mesh roster that
    awnode/awdk enrollment already writes, and the "may this run now?" question
    can be put to the device's own awnode node decider (``POST /decide/local``)
    through an injected ``decider``. A decider that is configured but does not
    answer a confident yes is a refusal: when in doubt, the owner's device is
    the owner's."""

    name = "mesh-device"

    def __init__(self, roster: Optional[Callable[[], List[Dict[str, Any]]]] = None,
                 decider: Optional[Callable[[Dict[str, Any], Job], Optional[bool]]] = None,
                 store_factory: Optional[Callable[[], Any]] = None) -> None:
        super().__init__(store_factory)
        self._roster = roster or _roster_from_file
        self._decider = decider

    def _device(self, peer: str) -> Dict[str, Any]:
        for d in self._roster():
            if d.get("peer") == peer:
                return d
        raise TargetRefused(f"peer {peer!r} is not on the mesh roster")

    def available(self) -> Tuple[bool, str]:
        try:
            n = len(self._roster())
        except (TargetRefused, OSError, ValueError) as e:
            return False, str(e)
        return True, f"{n} device(s) on the mesh roster"

    def check(self, job: Job, lease: Lease, ctx: Ctx) -> Optional[str]:
        if not job.peer:
            return "a mesh-device job names its peer"
        try:
            device = self._device(job.peer)
        except (TargetRefused, OSError, ValueError) as e:
            return str(e)
        why = device_fit(device, dict(job.spec.get("needs") or {}))
        if why:
            return why
        if self._decider is not None:
            try:
                ok = self._decider(device, job)
            except Exception as e:  # unreachable decider: refuse, never assume yes
                return f"node decider did not answer ({type(e).__name__}); not running"
            if ok is not True:
                return "node decider did not say yes"
        return None

    def submit(self, job: Job, lease: Lease, ctx: Ctx) -> Dict[str, Any]:
        device = self._device(job.peer)
        spec = dict(job.spec, target_peer=job.peer,
                    runner_labels=sorted(device.get("labels") or []))
        job = Job(kind=job.kind, spec=spec, est_usd=job.est_usd, priority=job.priority,
                  peer=job.peer)
        out = self._submit(job.kind, job, lease, ctx)
        # awrun claims by kind/run id today, not by label: the device's own
        # dispatcher takes exactly this item with `--run-id <handle>`.
        out["claim"] = f"python -m awrun.dispatcher --once --run-id {out['handle']}"
        return out


class UnavailableTarget:
    def __init__(self, name: str, reason: str) -> None:
        self.name = name
        self._reason = reason

    def available(self) -> Tuple[bool, str]:
        return False, self._reason

    def check(self, job: Job, lease: Lease, ctx: Ctx) -> Optional[str]:
        return self._reason

    def submit(self, job: Job, lease: Lease, ctx: Ctx) -> Dict[str, Any]:
        raise TargetRefused(self._reason)


def default_targets() -> Dict[str, Any]:
    ts: List[Any] = [
        AwrunTarget(), CiSpotTarget(), GpuDoorTarget(), VolunteerTarget(), MeshDeviceTarget(),
        UnavailableTarget("github-mirror",
                          "the public-runner boundary check is not written yet; nothing runs "
                          "on a stranger-shared runner until it is"),
        UnavailableTarget("cloudflare", "no Cloudflare Workers/Containers runner adapter yet"),
    ]
    return {t.name: t for t in ts}


def dispatch(store: Any, lease_id: str, target: Any, job: Job, now: float) -> Dict[str, Any]:
    """Run ``job`` under a lease on ``target``. Order is the whole contract:
    active -> target allowed -> target check -> charge -> submit (refund if the
    submit fails). Any refusal before the charge leaves the lease unchanged."""
    lease = store.lease(lease_id)
    lease.require_active(now)
    if target.name not in lease.targets:
        raise TargetRefused(f"lease {lease.id} does not allow target {target.name!r} "
                            f"(allowed: {lease.targets})")
    ctx = Ctx(now=now, store=store)
    reason = target.check(job, lease, ctx)
    if reason:
        raise TargetRefused(f"{target.name}: {reason}")
    lease.charge(job.est_usd, now, why=f"{target.name}:{job.kind}")
    try:
        out = target.submit(job, lease, ctx)
    except Exception as e:
        lease.refund(job.est_usd, now, why=f"submit failed: {type(e).__name__}")
        store.put_lease(lease)
        raise
    lease._event("run", now, target=target.name, handle=out.get("handle"))
    store.put_lease(lease)
    return dict(out, lease=lease.id, charged_usd=job.est_usd,
                remaining_usd=lease.remaining_usd())
