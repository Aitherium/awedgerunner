"""awedgerunner CLI.

    awedgerunner targets
    awedgerunner pool create --id P --owner O --funder F --budget 50 --per-lease-cap 5
    awedgerunner consent grant --peer N --owner O --window 0,18,23 --days 30
    awedgerunner lease open --pool P --contributor C --approved-by O --payer-kind pool
                            --payer-id F --credential-kind pooled_api_key
                            --credential-ref POOL_KEY_NAME --cap 1 --ttl 1800 --targets awrun
    awedgerunner lease show|close ID      awedgerunner lease list
    awedgerunner run --lease ID --target awrun --kind agent --spec-json '{}' --est-usd 0.2
    awedgerunner credit                   (SIMULATED rows only)
    awedgerunner --self-test

Exit codes: 0 ok, 1 refused / violation, 2 could not run.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, List, Optional

from . import __version__
from .lease import Consent, LeaseError, Pool, open_lease
from .store import Store
from .targets import (
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


def _print(obj: Any) -> None:
    print(json.dumps(obj, indent=2, sort_keys=True, default=str))


def _cmd_targets(args: argparse.Namespace) -> int:
    from .image import ByokSpotTarget, CloudflareWorkersAITarget, ComfyCloudTarget

    rows = []
    every = dict(default_targets())
    for t in (CloudflareWorkersAITarget(), ComfyCloudTarget(), ByokSpotTarget()):
        every[t.name] = t  # image BYOK targets: unavailable until the host wires a client
    for name, t in every.items():
        ok, why = t.available()
        rows.append({"target": name, "available": ok, "detail": why})
    _print(rows)
    return 0


def _cmd_pool(args: argparse.Namespace) -> int:
    pool = Pool(id=args.id, owner=args.owner, funder=args.funder, budget_usd=args.budget,
                per_lease_cap_usd=args.per_lease_cap, max_ttl_s=args.max_ttl)
    _print(Store().put_pool(pool).to_dict())
    return 0


def _cmd_consent(args: argparse.Namespace) -> int:
    store = Store()
    if args.action == "revoke":
        _print({"peer": args.peer, "revoked": store.revoke_consent(args.peer)})
        return 0
    now = time.time()
    windows = [[int(x) for x in w.split(",")] for w in args.window]
    c = Consent(peer=args.peer, owner=args.owner, purpose=args.purpose, windows=windows,
                expires_at=now + args.days * 86400, granted_at=now)
    _print(store.put_consent(c).to_dict())
    return 0


def _cmd_lease(args: argparse.Namespace) -> int:
    store = Store()
    now = time.time()
    if args.action == "open":
        lease = open_lease(store.pool(args.pool), store.leases(), contributor=args.contributor,
                           approved_by=args.approved_by, payer_kind=args.payer_kind,
                           payer_id=args.payer_id, credential_kind=args.credential_kind,
                           credential_ref=args.credential_ref or "", cap_usd=args.cap,
                           ttl_s=args.ttl, targets=args.targets.split(","), now=now)
        _print(store.put_lease(lease).to_dict())
    elif args.action == "list":
        _print([{"id": le.id, "pool": le.pool, "contributor": le.contributor,
                 "spent_usd": le.spent_usd, "cap_usd": le.cap_usd,
                 "state": le.active_reason(now) or "active"} for le in store.leases()])
    elif args.action == "show":
        _print(store.lease(args.id).to_dict())
    elif args.action == "close":
        lease = store.lease(args.id)
        lease.close(now, why="closed by cli")
        _print(store.put_lease(lease).to_dict())
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    targets = default_targets()
    if args.target not in targets:
        print(f"unknown target {args.target!r}; one of {sorted(targets)}", file=sys.stderr)
        return 2
    job = Job(kind=args.kind, spec=json.loads(args.spec_json), est_usd=args.est_usd,
              priority=args.priority, peer=args.peer or "",
              gpu=json.loads(args.gpu_json) if args.gpu_json else None)
    _print(dispatch(Store(), args.lease, targets[args.target], job, time.time()))
    return 0


def _cmd_credit(args: argparse.Namespace) -> int:
    _print({"simulated": True, "note": "credit is SIMULATED; nothing here settles",
            "rows": Store().credit()})
    return 0


# -- self-test ------------------------------------------------------------------


class _FakeRunStore:
    def __init__(self) -> None:
        self.items: List[Any] = []

    def submit(self, kind: str, spec: dict, **kw: Any) -> Any:
        item = type("Item", (), {"id": f"r-fake{len(self.items):04d}", "kind": kind,
                                 "spec": spec, "kw": kw})()
        self.items.append(item)
        return item


class _FakeDoor:
    def __init__(self, grant: bool) -> None:
        self.grant = grant
        self.released: List[str] = []

    def bases(self) -> List[str]:
        return ["https://door.invalid"]

    def acquire(self, cls: str, vram_mb: int, **kw: Any) -> str:
        if not self.grant:
            raise RuntimeError("door refused")
        return "tok"

    def release(self, grant: str, outcome: str) -> bool:
        self.released.append(outcome)
        return True


def self_test() -> int:
    """Prove every refusal arm can still fire. 0 = all proven, 1 = an arm failed."""
    failures: List[str] = []

    def expect(label: str, fn: Any, exc: Any) -> None:
        try:
            fn()
        except exc:
            return
        except Exception as e:  # wrong refusal type is a failure too
            failures.append(f"{label}: raised {type(e).__name__}: {e}")
            return
        failures.append(f"{label}: was NOT refused")

    with tempfile.TemporaryDirectory() as tmp:
        store = Store(Path(tmp))
        now = 1_763_200_000.0  # a fixed instant: a Saturday, 09:46 UTC
        pool = store.put_pool(Pool(id="commons", owner="owner", funder="commons-fund",
                                   budget_usd=3.0, per_lease_cap_usd=1.0, max_ttl_s=3600))
        common = dict(contributor="dev", approved_by="owner", payer_kind="pool",
                      payer_id="commons-fund", credential_kind="pooled_api_key",
                      credential_ref="POOL_KEY", targets=["awrun", "ci-spot", "gpu-door",
                                                          "volunteer", "github-mirror"])
        expect("consumer subscription", lambda: open_lease(
            pool, [], **dict(common, credential_kind="consumer_subscription"), cap_usd=1,
            ttl_s=60, now=now), LeaseError)
        expect("secret value as ref", lambda: open_lease(
            pool, [], **dict(common, credential_ref="sk-ant-abc123"), cap_usd=1, ttl_s=60,
            now=now), LeaseError)
        expect("no payer", lambda: open_lease(
            pool, [], **dict(common, payer_id=""), cap_usd=1, ttl_s=60, now=now), LeaseError)
        expect("cap over per-lease cap", lambda: open_lease(
            pool, [], **common, cap_usd=2, ttl_s=60, now=now), LeaseError)
        expect("unapproved", lambda: open_lease(
            pool, [], **dict(common, approved_by="dev"), cap_usd=1, ttl_s=60, now=now),
            LeaseError)
        leases = [open_lease(pool, [], **common, cap_usd=1, ttl_s=1800, now=now)
                  for _ in range(3)]
        expect("pool overcommit", lambda: open_lease(
            pool, leases, **common, cap_usd=1, ttl_s=60, now=now), LeaseError)
        lease = store.put_lease(leases[0])

        runs = _FakeRunStore()
        awrun = AwrunTarget(store_factory=lambda: runs)
        out = dispatch(store, lease.id, awrun, Job(spec={"task": "x"}, est_usd=0.6), now)
        if out["remaining_usd"] != 0.4 or out["handle"] != "r-fake0000":
            failures.append(f"awrun dispatch: unexpected {out}")
        if runs.items[0].kw.get("limits", {}).get("timeout_s", 0) > 1800:
            failures.append("awrun dispatch: run allowed to outlive its lease")
        expect("cap crossing", lambda: dispatch(
            store, lease.id, awrun, Job(est_usd=0.5), now), LeaseError)
        if store.lease(lease.id).spent_usd != 0.6:
            failures.append("refused charge changed the lease")
        expect("expired", lambda: dispatch(
            store, lease.id, awrun, Job(est_usd=0.1), now + 1801), LeaseError)

        cfg = Path(tmp) / "lane.yaml"
        cfg.write_text("enabled: true\nlabel: aws\ntenant_id: owner-tenant\n"
                       "daily_cap_usd: 10\n", encoding="utf-8")
        spot = CiSpotTarget(config_path=str(cfg), store_factory=lambda: runs)
        expect("ci-spot BYO-pay", lambda: dispatch(
            store, lease.id, spot, Job(spec={"workflow": "ci.yml"}, est_usd=0.1), now),
            TargetRefused)
        expect("github-mirror boundary", lambda: dispatch(
            store, lease.id, default_targets()["github-mirror"], Job(), now), TargetRefused)
        expect("gpu door refuses", lambda: dispatch(
            store, lease.id, GpuDoorTarget(lease_client=_FakeDoor(False),
                                           store_factory=lambda: runs),
            Job(gpu={"class": "chat", "vram_mb": 8000}), now), TargetRefused)

        vol = VolunteerTarget(store_factory=lambda: runs)
        expect("volunteer without consent", lambda: dispatch(
            store, lease.id, vol, Job(peer="gamer-pc"), now), TargetRefused)
        store.put_consent(Consent(peer="gamer-pc", owner="gamer",
                                  purpose="community_inference", windows=[[0, 18, 23]],
                                  expires_at=now + 86400))
        expect("volunteer outside window", lambda: dispatch(
            store, lease.id, vol, Job(peer="gamer-pc"), now), TargetRefused)
        store.put_consent(Consent(peer="gamer-pc", owner="gamer",
                                  purpose="community_inference", windows=[[5, 0, 24]],
                                  expires_at=now + 86400))
        dispatch(store, lease.id, vol, Job(peer="gamer-pc", credit_units=1.0), now)
        busy = {"peer": "pixel-10", "enrolled": True, "arch": "arm64", "ram_mb": 12288,
                "accel": "npu", "on_battery": False, "models": ["bonsai-2b"],
                "labels": ["bonsai"], "owner_active": True}
        mesh_lease = store.put_lease(open_lease(
            pool, [lease], **dict(common, targets=["mesh-device"]), cap_usd=1, ttl_s=600,
            now=now))
        expect("mesh device in owner use", lambda: dispatch(
            store, mesh_lease.id, MeshDeviceTarget(roster=lambda: [busy],
                                                   store_factory=lambda: runs),
            Job(peer="pixel-10"), now), TargetRefused)
        expect("mesh device wrong model", lambda: dispatch(
            store, mesh_lease.id, MeshDeviceTarget(
                roster=lambda: [dict(busy, owner_active=False)], store_factory=lambda: runs),
            Job(peer="pixel-10", spec={"needs": {"model": "bonsai-27b"}}), now), TargetRefused)
        rows = store.credit()
        if len(rows) != 1 or rows[0].get("simulated") is not True:
            failures.append(f"volunteer credit not recorded as simulated: {rows}")

    if failures:
        print("awedgerunner self-test: FAILED")
        for f in failures:
            print("  x " + f)
        return 1
    print("awedgerunner self-test: OK (every refusal arm fired; credit stayed simulated)")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="awedgerunner", description=__doc__.splitlines()[0])
    p.add_argument("--version", action="version", version=f"awedgerunner {__version__}")
    p.add_argument("--self-test", action="store_true", help="prove the refusals still fire")
    sub = p.add_subparsers(dest="cmd")

    sub.add_parser("targets", help="list runner targets and whether each is available")

    pc = sub.add_parser("pool", help="create a community budget pool")
    pc.add_argument("action", choices=["create"])
    pc.add_argument("--id", required=True)
    pc.add_argument("--owner", required=True)
    pc.add_argument("--funder", required=True)
    pc.add_argument("--budget", type=float, required=True)
    pc.add_argument("--per-lease-cap", type=float, required=True)
    pc.add_argument("--max-ttl", type=int, default=24 * 3600)

    cc = sub.add_parser("consent", help="a volunteer node owner's opt-in")
    cc.add_argument("action", choices=["grant", "revoke"])
    cc.add_argument("--peer", required=True)
    cc.add_argument("--owner", default="")
    cc.add_argument("--purpose", default="community_inference")
    cc.add_argument("--window", action="append", default=[],
                    help="weekday,start_hour,end_hour in UTC (0=Mon); repeatable")
    cc.add_argument("--days", type=int, default=30)

    lc = sub.add_parser("lease", help="open, list, show or close a lease")
    lc.add_argument("action", choices=["open", "list", "show", "close"])
    lc.add_argument("id", nargs="?")
    lc.add_argument("--pool")
    lc.add_argument("--contributor")
    lc.add_argument("--approved-by")
    lc.add_argument("--payer-kind")
    lc.add_argument("--payer-id")
    lc.add_argument("--credential-kind", default="none")
    lc.add_argument("--credential-ref")
    lc.add_argument("--cap", type=float)
    lc.add_argument("--ttl", type=int)
    lc.add_argument("--targets", default="awrun")

    rc = sub.add_parser("run", help="run one job under a lease")
    rc.add_argument("--lease", required=True)
    rc.add_argument("--target", required=True)
    rc.add_argument("--kind", default="agent")
    rc.add_argument("--spec-json", default="{}")
    rc.add_argument("--gpu-json", default="")
    rc.add_argument("--est-usd", type=float, required=True)
    rc.add_argument("--priority", type=int, default=0)
    rc.add_argument("--peer", default="")

    sub.add_parser("credit", help="the SIMULATED volunteer credit ledger")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    # GENERATED doctor intercept (gen_aw_doctor.py) -- do not edit
    _dv = locals().get("argv")
    if (_dv if _dv is not None else __import__("sys").argv[1:])[:1] == ["doctor"]:
        from ._doctor import report
        return report()
    # GENERATED repo-state intercept (gen_aw_doctor.py) -- do not edit
    try:
        from awgit import state as _aw_state
    except Exception:
        _aw_state = None
    if _aw_state is not None:
        _sv = locals().get("argv")
        if _aw_state.cli_banner(_sv if _sv is not None else __import__("sys").argv[1:]):
            return 0
    args = build_parser().parse_args(argv)
    if args.self_test:
        return self_test()
    handlers = {"targets": _cmd_targets, "pool": _cmd_pool, "consent": _cmd_consent,
                "lease": _cmd_lease, "run": _cmd_run, "credit": _cmd_credit}
    if args.cmd not in handlers:
        build_parser().print_help()
        return 2
    if args.cmd == "lease" and args.action in ("show", "close") and not args.id:
        print("lease show/close needs a lease id", file=sys.stderr)
        return 2
    try:
        return handlers[args.cmd](args)
    except (LeaseError, TargetRefused) as e:
        print(f"refused: {e}", file=sys.stderr)
        return 1
    except (OSError, ValueError, TypeError) as e:
        print(f"could not run: {type(e).__name__}: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
