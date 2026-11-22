"""A small JSON store for pools, consents, leases and the simulated credit ledger.

Layout under ``$AWEDGERUNNER_DIR`` (default ``~/.aither/awedgerunner``)::

    pools.json      {pool_id: Pool}
    consents.json   {peer: Consent}
    leases/<id>.json
    credit.jsonl    one SIMULATED credit row per line

Writes are atomic (temp file + ``os.replace``); a half-written lease never
reads as a valid one.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from .lease import Consent, Lease, LeaseError, Pool


def default_dir() -> Path:
    raw = os.environ.get("AWEDGERUNNER_DIR", "").strip()
    return Path(raw) if raw else Path.home() / ".aither" / "awedgerunner"


def _atomic_write(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(tmp, path)


def _read(path: Path, default: Any) -> Any:
    if not path.is_file():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        raise LeaseError(f"{path} is unreadable ({e}); refusing to guess its contents") from e


class Store:
    def __init__(self, path: Optional[Path] = None) -> None:
        self.path = Path(path) if path else default_dir()

    # -- pools ---------------------------------------------------------------

    def pools(self) -> Dict[str, Pool]:
        raw = _read(self.path / "pools.json", {})
        return {k: Pool.from_dict(v) for k, v in raw.items()}

    def put_pool(self, pool: Pool) -> Pool:
        pool.validate()
        pools = {k: v.to_dict() for k, v in self.pools().items()}
        pools[pool.id] = pool.to_dict()
        _atomic_write(self.path / "pools.json", pools)
        return pool

    def pool(self, pool_id: str) -> Pool:
        pool = self.pools().get(pool_id)
        if pool is None:
            raise LeaseError(f"no such pool: {pool_id!r}")
        return pool

    # -- consents ------------------------------------------------------------

    def consents(self) -> Dict[str, Consent]:
        raw = _read(self.path / "consents.json", {})
        return {k: Consent.from_dict(v) for k, v in raw.items()}

    def put_consent(self, consent: Consent) -> Consent:
        consent.validate()
        all_ = {k: v.to_dict() for k, v in self.consents().items()}
        all_[consent.peer] = consent.to_dict()
        _atomic_write(self.path / "consents.json", all_)
        return consent

    def revoke_consent(self, peer: str) -> bool:
        all_ = {k: v.to_dict() for k, v in self.consents().items()}
        found = all_.pop(peer, None) is not None
        _atomic_write(self.path / "consents.json", all_)
        return found

    # -- leases --------------------------------------------------------------

    def _lease_path(self, lease_id: str) -> Path:
        if not lease_id.startswith("el-") or not lease_id[3:].isalnum():
            raise LeaseError(f"not a lease id: {lease_id!r}")
        return self.path / "leases" / f"{lease_id}.json"

    def put_lease(self, lease: Lease) -> Lease:
        _atomic_write(self._lease_path(lease.id), lease.to_dict())
        return lease

    def lease(self, lease_id: str) -> Lease:
        raw = _read(self._lease_path(lease_id), None)
        if raw is None:
            raise LeaseError(f"no such lease: {lease_id!r}")
        return Lease.from_dict(raw)

    def leases(self) -> List[Lease]:
        d = self.path / "leases"
        if not d.is_dir():
            return []
        return [Lease.from_dict(_read(p, {})) for p in sorted(d.glob("el-*.json"))]

    # -- simulated credit ----------------------------------------------------

    def append_credit(self, row: Dict[str, Any]) -> None:
        row = dict(row, simulated=True)
        self.path.mkdir(parents=True, exist_ok=True)
        with open(self.path / "credit.jsonl", "a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, sort_keys=True) + "\n")

    def credit(self) -> List[Dict[str, Any]]:
        p = self.path / "credit.jsonl"
        if not p.is_file():
            return []
        return [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines()
                if line.strip()]
