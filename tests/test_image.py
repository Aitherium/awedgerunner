"""awedgerunner image class: BYOK routes only to BYOK, ours never to the cloud."""

from __future__ import annotations

import pytest
from awedgerunner.image import (
    BATCH_MAX,
    CPU_PROVEN_MODELS,
    CPU_SWARM_IMAGES_PER_MIN,
    ByokSpotTarget,
    CloudflareWorkersAITarget,
    ComfyCloudTarget,
    CpuSwarmTarget,
    byok_order,
    eta_seconds,
    select_image_target,
    split_batches,
    validate_image_job,
)
from awedgerunner.lease import Pool, open_lease
from awedgerunner.store import Store
from awedgerunner.targets import Ctx, GpuDoorTarget, Job, TargetRefused

NOW = 1_763_200_000.0
GRAPH = {"3": {"class_type": "KSampler", "inputs": {}}}


class Door:
    def __init__(self, grant=True):
        self.grant = grant

    def bases(self):
        return ["https://door.invalid"]

    def acquire(self, *a, **k):
        if not self.grant:
            raise RuntimeError("busy")
        return "tok"

    def release(self, *a):
        return True


class Runs:
    def __init__(self):
        self.items = []

    def submit(self, kind, spec, **kw):
        it = type("I", (), {"id": f"r-{len(self.items)}", "kind": kind})()
        self.items.append(it)
        return it


ALL = ["gpu-door", "volunteer", "mesh-device", "cf-workers-ai", "comfy-cloud", "byok-spot"]


@pytest.fixture
def store(tmp_path):
    s = Store(tmp_path)
    s.put_pool(Pool(id="p", owner="o", funder="fund", budget_usd=10, per_lease_cap_usd=5))
    return s


def _byok(store, ref="workspace:acme/CF_TOKEN"):
    return open_lease(store.pool("p"), [], contributor="acme", approved_by="o",
                      payer_kind="byo", payer_id="acme", credential_kind="byo_key",
                      credential_ref=ref, cap_usd=1, ttl_s=600, targets=ALL, now=NOW)


def _ours(store):
    return open_lease(store.pool("p"), [], contributor="dev", approved_by="o",
                      payer_kind="pool", payer_id="fund", credential_kind="none",
                      credential_ref="", cap_usd=1, ttl_s=600, targets=ALL, now=NOW)


def _job(n=3):
    return Job(kind="image", spec={"graph": GRAPH, "prompts": [f"p{i}" for i in range(n)]},
               gpu={"class": "interactive_media", "vram_mb": 6000})


def _targets(runs, door_grant=True, cf=True, comfy=True):
    return {
        "gpu-door": GpuDoorTarget(Door(door_grant), lambda: runs),
        "cf-workers-ai": CloudflareWorkersAITarget(
            (lambda m, p, s, k: {"handle": "cf-1"}) if cf else None),
        "comfy-cloud": ComfyCloudTarget((lambda g, p, k: {"handle": "cc-1"}) if comfy else None),
        "byok-spot": ByokSpotTarget(None),
    }


def test_byok_lease_goes_to_cheapest_byok_and_never_to_ours(store):
    ctx = Ctx(now=NOW, store=store)
    t, trail = select_image_target(_job(), _byok(store), _targets(Runs()), ctx)
    assert t.name == "cf-workers-ai"
    assert byok_order()[0] == "cf-workers-ai"
    t, trail = select_image_target(_job(), _byok(store), _targets(Runs(), cf=False, comfy=False),
                                   ctx)
    assert t is None
    assert all(row["target"] not in ("gpu-door", "volunteer", "mesh-device") for row in trail)


def test_our_lease_uses_the_gpu_door_and_never_byok(store):
    ctx = Ctx(now=NOW, store=store)
    t, _ = select_image_target(_job(), _ours(store), _targets(Runs()), ctx)
    assert t.name == "gpu-door"
    t, trail = select_image_target(_job(), _ours(store), _targets(Runs(), door_grant=False), ctx)
    assert t is None
    assert {row["target"] for row in trail} == {"gpu-door", "volunteer", "mesh-device",
                                                 "cpu-swarm"}


def test_gpu_door_image_batch_is_one_render_item(store):
    runs = Runs()
    lease = _ours(store)
    door = GpuDoorTarget(Door(), lambda: runs)
    assert door.check(_job(8), lease, Ctx(now=NOW, store=store)) is None
    door.submit(_job(8), lease, Ctx(now=NOW, store=store))
    assert [i.kind for i in runs.items] == ["render"]


@pytest.mark.parametrize("cls", [ComfyCloudTarget, CloudflareWorkersAITarget, ByokSpotTarget])
def test_byok_targets_refuse_platform_keys_and_value_refs(store, cls):
    ctx = Ctx(now=NOW, store=store)
    t = cls(lambda *a: {"handle": "x"})
    assert "customer's own key" in t.check(_job(), _ours(store), ctx)
    assert "workspace Lockbox ref" in t.check(_job(), _byok(store, ref="CF_TOKEN"), ctx)
    assert t.check(_job(), _byok(store), ctx) is None


def test_unwired_byok_target_is_unavailable_and_submit_refuses(store):
    t = ComfyCloudTarget(None)
    assert t.available()[0] is False
    with pytest.raises(TargetRefused):
        t.submit(_job(), _byok(store), Ctx(now=NOW, store=store))


def test_comfy_cloud_gets_the_same_graph(store):
    seen = {}

    def client(graph, prompts, key_ref):
        seen.update(graph=graph, prompts=prompts, key_ref=key_ref)
        return {"handle": "cc-9"}

    out = ComfyCloudTarget(client).submit(_job(2), _byok(store), Ctx(now=NOW, store=store))
    assert seen["graph"] is GRAPH and seen["key_ref"].startswith("workspace:")
    assert out == {"handle": "cc-9", "via": "comfy-cloud"}


def test_job_shape_and_batching():
    assert "graph" in validate_image_job(Job(kind="image", spec={"prompts": ["a"]}))
    assert "prompts" in validate_image_job(Job(kind="image", spec={"graph": GRAPH}))
    assert "split" in validate_image_job(_job(BATCH_MAX + 1))
    assert split_batches(list("abcde"), 2) == [["a", "b"], ["c", "d"], ["e"]]


def test_eta_only_from_measured_capacity():
    assert eta_seconds(10, {}) is None
    assert eta_seconds(10, {"gpu-door": {"available": True}}) is None
    assert eta_seconds(10, {"gpu-door": {"available": False, "images_per_min": 60}}) is None
    assert eta_seconds(120, {"gpu-door": {"available": True, "images_per_min": 30, "slots": 2},
                             "cf": {"available": True, "images_per_min": 60}}) == 60.0
    assert eta_seconds(0, {}) == 0.0


def test_skipped_byok_target_offers_only_configured_referral_links(store):
    ctx = Ctx(now=NOW, store=store)
    refs = {"cloudflare": {"configured": True, "url": "https://cf.example/ref",
                           "disclosure": "We earn a referral credit."},
            "comfy_cloud": {"configured": False, "url": ""},
            "vastai": {"configured": True, "url": "https://cloud.vast.ai/?ref_id=1",
                       "disclosure": "d"},
            "lambda": {"configured": True, "url": "http://not-https.example"}}
    t, trail = select_image_target(_job(), _byok(store), _targets(Runs(), cf=False, comfy=False),
                                   ctx, referrals=refs)
    assert t is None
    rows = {r["target"]: r for r in trail}
    assert rows["cf-workers-ai"]["signup"][0]["url"] == "https://cf.example/ref"
    assert "signup" not in rows["comfy-cloud"]
    assert [x["provider"] for x in rows["byok-spot"]["signup"]] == ["vastai"]
    _, ours = select_image_target(_job(), _ours(store), _targets(Runs(), door_grant=False),
                                  ctx, referrals=refs)
    assert all("signup" not in r for r in ours)


ALL_CPU = ALL + ["cpu-swarm"]


def _sana_job(n=20):
    return Job(kind="image", spec={"graph": GRAPH, "model": "sana-sprint-0.6b",
                                   "prompts": [f"p{i}" for i in range(n)]},
               gpu={"class": "interactive_media", "vram_mb": 6000})


def test_cpu_swarm_ranks_after_gpu_and_carries_the_notice(store):
    ctx = Ctx(now=NOW, store=store)
    lease = open_lease(store.pool("p"), [], contributor="dev", approved_by="o",
                       payer_kind="pool", payer_id="fund", credential_kind="none",
                       credential_ref="", cap_usd=1, ttl_s=600, targets=ALL_CPU, now=NOW)
    seen = {}

    def dispatch(model, batches, notice):
        seen.update(model=model, batches=batches, notice=notice)
        return {"handle": "swarm-1"}

    targets = dict(_targets(Runs()), **{"cpu-swarm": CpuSwarmTarget(dispatch, per_runner=8)})
    t, _ = select_image_target(_sana_job(), lease, targets, ctx)
    assert t.name == "gpu-door"
    targets["gpu-door"] = GpuDoorTarget(Door(False), Runs)
    t, trail = select_image_target(_sana_job(), lease, targets, ctx)
    assert t.name == "cpu-swarm" and [r["target"] for r in trail][0] == "gpu-door"
    out = t.submit(_sana_job(), lease, ctx)
    assert out["runners"] == 3 and len(seen["batches"]) == 3
    assert "Apache License 2.0" in out["attribution"] and "Gemma Terms of Use" in out["attribution"]
    assert seen["notice"] == out["attribution"]


def test_cpu_swarm_refuses_unproven_models_and_byok_never_reaches_it(store):
    ctx = Ctx(now=NOW, store=store)
    t = CpuSwarmTarget(lambda *a: {"handle": "x"})
    assert "not CPU-proven" in t.check(_job(), _ours(store), ctx)
    lease = _byok(store)
    lease.targets.append("cpu-swarm")
    targets = {"cpu-swarm": t}
    chosen, trail = select_image_target(_sana_job(), lease, targets, ctx)
    assert chosen is None and all(r["target"] != "cpu-swarm" for r in trail)


def test_cpu_swarm_capacity_feeds_the_eta():
    meta = CPU_PROVEN_MODELS["sana-sprint-0.6b"]
    assert meta["licence"].startswith("apache-2.0") and meta["s_per_image"]["320"] == 16.9
    row = CpuSwarmTarget(lambda *a: {}).capacity(10)
    assert row["images_per_min"] == CPU_SWARM_IMAGES_PER_MIN == 3.55
    assert eta_seconds(355, {"cpu-swarm": row}) == 600.0
    assert CpuSwarmTarget(None).capacity(10)["available"] is False
    assert CpuSwarmTarget(lambda *a: {}).capacity(0)["available"] is False
