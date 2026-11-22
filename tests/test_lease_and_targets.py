"""awedgerunner: the lease refuses what it must, and targets delegate."""

from __future__ import annotations

import json
from typing import Any, List

import pytest
from awedgerunner import cli
from awedgerunner.lease import (
    BudgetExceeded,
    Consent,
    LeaseError,
    LeaseExpired,
    Pool,
    open_lease,
)
from awedgerunner.store import Store
from awedgerunner.targets import (
    AwrunTarget,
    CiSpotTarget,
    GpuDoorTarget,
    Job,
    MeshDeviceTarget,
    TargetRefused,
    VolunteerTarget,
    default_targets,
    dispatch,
)

NOW = 1_763_200_000.0  # Saturday 2025-11-15 09:46 UTC


class FakeRuns:
    def __init__(self) -> None:
        self.items: List[Any] = []

    def submit(self, kind, spec, **kw):
        item = type("Item", (), {"id": f"r-{len(self.items)}", "kind": kind, "spec": spec,
                                 "kw": kw})()
        self.items.append(item)
        return item


class FakeDoor:
    def __init__(self, grant: bool, bases=("https://door.invalid",)) -> None:
        self.grant, self._bases, self.released = grant, list(bases), []

    def bases(self):
        return self._bases

    def acquire(self, cls, vram_mb, **kw):
        if not self.grant:
            raise RuntimeError("busy")
        return "tok"

    def release(self, grant, outcome):
        self.released.append(outcome)


@pytest.fixture
def store(tmp_path):
    s = Store(tmp_path)
    s.put_pool(Pool(id="commons", owner="owner", funder="fund", budget_usd=5,
                    per_lease_cap_usd=2, max_ttl_s=3600))
    return s


def _lease(store, **over):
    kw = dict(contributor="dev", approved_by="owner", payer_kind="pool", payer_id="fund",
              credential_kind="pooled_api_key", credential_ref="POOL_KEY", cap_usd=1.0,
              ttl_s=1800, targets=["awrun", "ci-spot", "gpu-door", "volunteer",
                                   "github-mirror", "cloudflare"], now=NOW)
    kw.update(over)
    return store.put_lease(open_lease(store.pool("commons"), store.leases(), **kw))


@pytest.mark.parametrize("over,match", [
    ({"credential_kind": "consumer_subscription"}, "never shared"),
    ({"credential_ref": "sk-ant-api03-xyz"}, "secret VALUE"),
    ({"credential_ref": ""}, "needs a credential_ref"),
    ({"payer_kind": "platform"}, "no platform fallback"),
    ({"payer_id": ""}, "who pays"),
    ({"payer_id": "someone-else"}, "funder"),
    ({"payer_kind": "byo", "payer_id": "me"}, "pooled API key is paid by the pool"),
    ({"approved_by": "dev"}, "approved by the pool owner"),
    ({"cap_usd": 0}, "no unlimited"),
    ({"cap_usd": 3}, "per-lease cap"),
    ({"ttl_s": 7200}, "ttl_s"),
    ({"targets": []}, "targets"),
])
def test_open_refuses(store, over, match):
    with pytest.raises(LeaseError, match=match):
        _lease(store, **over)


def test_pool_overcommit_refused(store):
    _lease(store, cap_usd=2)
    _lease(store, cap_usd=2)
    with pytest.raises(BudgetExceeded):
        _lease(store, cap_usd=2)


def test_cap_is_hard_and_refusal_leaves_lease_unchanged(store):
    lease = _lease(store)
    runs = FakeRuns()
    t = AwrunTarget(store_factory=lambda: runs)
    out = dispatch(store, lease.id, t, Job(est_usd=0.75), NOW)
    assert out["remaining_usd"] == 0.25
    with pytest.raises(BudgetExceeded):
        dispatch(store, lease.id, t, Job(est_usd=0.5), NOW)
    assert store.lease(lease.id).spent_usd == 0.75
    assert len(runs.items) == 1


def test_expiry_and_close(store):
    lease = _lease(store)
    t = AwrunTarget(store_factory=FakeRuns)
    with pytest.raises(LeaseExpired):
        dispatch(store, lease.id, t, Job(), NOW + 1800)
    lease.close(NOW)
    store.put_lease(lease)
    with pytest.raises(LeaseExpired):
        dispatch(store, lease.id, t, Job(), NOW)


def test_run_cannot_outlive_lease(store):
    lease = _lease(store)
    runs = FakeRuns()
    dispatch(store, lease.id, AwrunTarget(store_factory=lambda: runs), Job(), NOW + 1000)
    assert runs.items[0].kw["limits"]["timeout_s"] == 800
    assert runs.items[0].spec["awedgerunner"]["lease"] == lease.id


def test_target_not_on_lease_refused(store):
    lease = _lease(store, targets=["awrun"])
    with pytest.raises(TargetRefused, match="does not allow"):
        dispatch(store, lease.id, VolunteerTarget(store_factory=FakeRuns), Job(peer="p"), NOW)


def test_failed_submit_refunds(store):
    lease = _lease(store)

    class Boom:
        def submit(self, *a, **k):
            raise OSError("disk")

    with pytest.raises(OSError):
        dispatch(store, lease.id, AwrunTarget(store_factory=Boom), Job(est_usd=0.5), NOW)
    assert store.lease(lease.id).spent_usd == 0


def test_ci_spot_refuses_other_payer(store, tmp_path):
    cfg = tmp_path / "lane.yaml"
    cfg.write_text("enabled: true\nlabel: aws\ntenant_id: owner-t\ndaily_cap_usd: 10\n",
                   encoding="utf-8")
    runs = FakeRuns()
    t = CiSpotTarget(config_path=str(cfg), store_factory=lambda: runs)
    pooled = _lease(store)
    with pytest.raises(TargetRefused, match="BYO-pay"):
        dispatch(store, pooled.id, t, Job(spec={"workflow": "ci.yml"}), NOW)
    own = _lease(store, payer_kind="byo", payer_id="owner-t", credential_kind="none",
                 credential_ref="")
    out = dispatch(store, own.id, t, Job(spec={"workflow": "ci.yml"}, est_usd=0.1), NOW)
    assert out["kind"] == "ci" and runs.items[0].kind == "ci"


def test_ci_spot_without_config_is_unavailable(store):
    ok, why = CiSpotTarget(config_path="/nonexistent.yaml").available()
    assert not ok and "not found" in why


def test_gpu_door_fails_closed(store):
    lease = _lease(store)
    runs = FakeRuns()
    gpu = {"class": "chat", "vram_mb": 8000}
    with pytest.raises(TargetRefused, match="not running unleased"):
        dispatch(store, lease.id, GpuDoorTarget(FakeDoor(False), lambda: runs), Job(gpu=gpu),
                 NOW)
    with pytest.raises(TargetRefused, match="no GPU lease door"):
        dispatch(store, lease.id, GpuDoorTarget(FakeDoor(True, bases=()), lambda: runs),
                 Job(gpu=gpu), NOW)
    door = FakeDoor(True)
    dispatch(store, lease.id, GpuDoorTarget(door, lambda: runs), Job(gpu=gpu), NOW)
    assert door.released == ["preflight"]
    assert runs.items[0].kw["gpu"]["ttl_s"] <= 1800


def test_volunteer_consent_window_and_simulated_credit(store):
    lease = _lease(store)
    runs = FakeRuns()
    t = VolunteerTarget(store_factory=lambda: runs)
    with pytest.raises(TargetRefused, match="default-deny"):
        dispatch(store, lease.id, t, Job(peer="pc"), NOW)
    store.put_consent(Consent(peer="pc", owner="gamer", purpose="community_inference",
                              windows=[[0, 18, 23]], expires_at=NOW + 3600))
    with pytest.raises(TargetRefused, match="window"):
        dispatch(store, lease.id, t, Job(peer="pc"), NOW)
    store.put_consent(Consent(peer="pc", owner="gamer", purpose="community_inference",
                              windows=[[5, 9, 10]], expires_at=NOW + 3600))
    out = dispatch(store, lease.id, t, Job(peer="pc", credit_units=2), NOW)
    assert out["credit"] == "simulated"
    assert store.credit()[0]["simulated"] is True
    store.revoke_consent("pc")
    with pytest.raises(TargetRefused):
        dispatch(store, lease.id, t, Job(peer="pc"), NOW)


@pytest.mark.parametrize("name", ["github-mirror", "cloudflare"])
def test_unbuilt_targets_refuse(store, name):
    lease = _lease(store)
    t = default_targets()[name]
    assert t.available()[0] is False
    with pytest.raises(TargetRefused):
        dispatch(store, lease.id, t, Job(), NOW)


def test_self_test_passes():
    assert cli.self_test() == 0


def test_cli_roundtrip(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("AWEDGERUNNER_DIR", str(tmp_path))
    assert cli.main(["pool", "create", "--id", "p", "--owner", "o", "--funder", "f",
                     "--budget", "5", "--per-lease-cap", "1"]) == 0
    capsys.readouterr()
    assert cli.main(["lease", "open", "--pool", "p", "--contributor", "c", "--approved-by",
                     "o", "--payer-kind", "pool", "--payer-id", "f", "--credential-kind",
                     "pooled_api_key", "--credential-ref", "K", "--cap", "1", "--ttl",
                     "600"]) == 0
    lease_id = json.loads(capsys.readouterr().out)["id"]
    assert cli.main(["lease", "open", "--pool", "p", "--contributor", "c", "--approved-by",
                     "o", "--payer-kind", "pool", "--payer-id", "f", "--credential-kind",
                     "consumer_subscription", "--cap", "1", "--ttl", "600"]) == 1
    assert cli.main(["lease", "close", lease_id]) == 0
    assert cli.main(["targets"]) == 0


PIXEL = {"peer": "pixel-10", "enrolled": True, "arch": "arm64", "ram_mb": 12288,
         "accel": "npu", "on_battery": True, "charging": True,
         "models": ["bonsai-2b", "bonsai-4b"], "labels": ["android", "bonsai"],
         "owner_active": False}


def test_mesh_device_takes_matching_job_and_pins_the_peer(store):
    lease = _lease(store, targets=["mesh-device"])
    runs = FakeRuns()
    t = MeshDeviceTarget(roster=lambda: [PIXEL], store_factory=lambda: runs)
    needs = {"labels": ["bonsai"], "model": "bonsai-2b", "arch": "arm64", "min_ram_mb": 4096}
    out = dispatch(store, lease.id, t, Job(peer="pixel-10", spec={"needs": needs}), NOW)
    assert runs.items[0].spec["target_peer"] == "pixel-10"
    assert out["claim"].endswith(out["handle"])


@pytest.mark.parametrize("change,needs,match", [
    ({"owner_active": True}, {}, "preempted"),
    ({"owner_active": None}, {}, "preempted"),
    ({"charging": False}, {}, "battery"),
    ({"enrolled": False}, {}, "not enrolled"),
    ({}, {"labels": ["gpu"]}, "lacks labels"),
    ({}, {"model": "bonsai-27b"}, "cannot run model"),
    ({}, {"min_ram_mb": 65536}, "RAM"),
    ({}, {"arch": "x86_64"}, "arch"),
    ({}, {"accel": "gpu"}, "accel"),
])
def test_mesh_device_refuses_mismatch_and_owner_use(store, change, needs, match):
    lease = _lease(store, targets=["mesh-device"])
    device = dict(PIXEL, **change)
    if change.get("owner_active", False) is None:
        device.pop("owner_active")
    t = MeshDeviceTarget(roster=lambda: [device], store_factory=FakeRuns)
    with pytest.raises(TargetRefused, match=match):
        dispatch(store, lease.id, t, Job(peer="pixel-10", spec={"needs": needs}), NOW)


def test_mesh_device_decider_must_say_yes(store):
    lease = _lease(store, targets=["mesh-device"])

    def down(device, job):
        raise OSError("unreachable")

    for decider, match in ((lambda d, j: False, "did not say yes"),
                           (lambda d, j: None, "did not say yes"),
                           (down, "did not answer")):
        t = MeshDeviceTarget(roster=lambda: [PIXEL], decider=decider, store_factory=FakeRuns)
        with pytest.raises(TargetRefused, match=match):
            dispatch(store, lease.id, t, Job(peer="pixel-10"), NOW)


def test_mesh_device_off_roster_and_no_roster(store, monkeypatch):
    lease = _lease(store, targets=["mesh-device"])
    t = MeshDeviceTarget(roster=lambda: [PIXEL], store_factory=FakeRuns)
    with pytest.raises(TargetRefused, match="not on the mesh roster"):
        dispatch(store, lease.id, t, Job(peer="stranger"), NOW)
    monkeypatch.delenv("AWEDGERUNNER_MESH_ROSTER", raising=False)
    assert MeshDeviceTarget().available()[0] is False
