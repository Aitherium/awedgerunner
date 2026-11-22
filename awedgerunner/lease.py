"""The lease: who may spend how much of whose budget, until when.

Every rule here FAILS CLOSED. There is no default payer, no default cap, no
"unlimited", and no fallback: a lease that cannot say who pays is refused,
and a charge that would cross the cap is refused rather than clamped.

Owner rules carried as code:

* BYO-pay -- a lease names its payer. Nothing in this package substitutes
  another payer when the named one cannot cover a charge.
* Shared budget is POOLED API KEYS with a per-lease cap. A consumer chat
  subscription is one person's and is never a pooled credential, so a lease
  that names one is refused.
* Credentials are named by REFERENCE (a secret name), never by value.
"""

from __future__ import annotations

import re
import secrets
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


class LeaseError(RuntimeError):
    """A lease operation was refused. The message says exactly why."""


class BudgetExceeded(LeaseError):  # noqa: N818 - a refusal, not an error
    """The charge would cross the lease cap (or the pool budget)."""


class LeaseExpired(LeaseError):  # noqa: N818 - a refusal, not an error
    """The lease is past its expiry or was closed."""


#: Credential kinds a lease may carry. ``consumer_subscription`` is named so
#: the refusal can say what it saw; it is never accepted.
CREDENTIAL_KINDS = ("pooled_api_key", "byo_key", "none")
REFUSED_CREDENTIAL_KINDS = ("consumer_subscription",)

#: Payer kinds. ``byo`` = the lessee's own account/tenant; ``pool`` = a
#: community pool's funder. There is no ``platform`` payer on purpose.
PAYER_KINDS = ("byo", "pool")

#: Hard ceiling on any lease lifetime, whatever a pool says: 7 days.
MAX_TTL_S = 7 * 24 * 3600

# Shapes that look like a secret VALUE pasted where a secret NAME belongs.
_SECRET_SHAPES = re.compile(
    r"(sk-|sk_live_|pk_live_|ghp_|ghs_|gho_|github_pat_|xox[bp]-|AKIA[0-9A-Z]{8,}|-----BEGIN)")


def _check_ref(ref: str) -> str:
    ref = (ref or "").strip()
    if _SECRET_SHAPES.search(ref):
        raise LeaseError("credential ref looks like a secret VALUE; pass the secret's NAME "
                         "(e.g. 'POOL_ANTHROPIC_KEY'), never the key itself")
    if ref and not re.fullmatch(r"[A-Za-z0-9_.:/-]{1,128}", ref):
        raise LeaseError(f"credential ref {ref!r} is not a secret name")
    return ref


@dataclass
class Pool:
    """A community budget. ``budget_usd`` is the most every open lease may
    reserve in total; ``per_lease_cap_usd`` is the most any one lease may."""

    id: str
    owner: str
    funder: str
    budget_usd: float
    per_lease_cap_usd: float
    max_ttl_s: int = 24 * 3600

    def validate(self) -> "Pool":
        if not re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,63}", self.id or ""):
            raise LeaseError(f"pool id {self.id!r} must be lowercase [a-z0-9_-]")
        if not self.owner or not self.funder:
            raise LeaseError("a pool needs an owner (who approves leases) and a funder "
                             "(who pays); neither has a default")
        if not (self.budget_usd > 0 and self.per_lease_cap_usd > 0):
            raise LeaseError("pool budget and per-lease cap must both be > 0")
        if self.per_lease_cap_usd > self.budget_usd:
            raise LeaseError("per-lease cap cannot exceed the pool budget")
        if not (0 < int(self.max_ttl_s) <= MAX_TTL_S):
            raise LeaseError(f"pool max_ttl_s must be in (0, {MAX_TTL_S}]")
        return self

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Pool":
        return cls(**{k: d[k] for k in ("id", "owner", "funder", "budget_usd",
                                        "per_lease_cap_usd", "max_ttl_s") if k in d})


@dataclass
class Consent:
    """A volunteer node owner's opt-in: this peer may serve this purpose, in
    these weekly UTC windows, until ``expires_at``. Default-deny: no consent
    record, or a time outside every window, is a refusal."""

    peer: str
    owner: str
    purpose: str
    windows: List[List[int]]  # [[weekday 0=Mon..6, start_hour, end_hour), ...]
    expires_at: float
    granted_at: float = 0.0

    def validate(self) -> "Consent":
        if not self.peer or not self.owner:
            raise LeaseError("consent needs the peer and the owner who granted it")
        if not self.windows:
            raise LeaseError("consent needs at least one schedule window; there is no "
                             "'always' default")
        for w in self.windows:
            if (len(w) != 3 or not 0 <= w[0] <= 6 or not 0 <= w[1] < w[2] <= 24):
                raise LeaseError(f"bad window {w!r}: want [weekday 0-6, start_h, end_h]")
        return self

    def allows(self, purpose: str, now: float) -> Optional[str]:
        """None when allowed, else the reason it is not."""
        if purpose != self.purpose:
            return f"consent covers purpose {self.purpose!r}, not {purpose!r}"
        if now >= self.expires_at:
            return "consent has expired"
        t = time.gmtime(now)
        for day, start, end in self.windows:
            if t.tm_wday == day and start <= t.tm_hour < end:
                return None
        return "outside every consented schedule window"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Consent":
        return cls(**d)


@dataclass
class Lease:
    id: str
    pool: str
    contributor: str
    approved_by: str
    payer_kind: str
    payer_id: str
    credential_kind: str
    credential_ref: str
    cap_usd: float
    created_at: float
    expires_at: float
    targets: List[str]
    spent_usd: float = 0.0
    closed: bool = False
    events: List[Dict[str, Any]] = field(default_factory=list)

    # -- state ---------------------------------------------------------------

    def remaining_usd(self) -> float:
        return round(self.cap_usd - self.spent_usd, 6)

    def remaining_s(self, now: float) -> int:
        return max(0, int(self.expires_at - now))

    def active_reason(self, now: float) -> Optional[str]:
        if self.closed:
            return "lease is closed"
        if now >= self.expires_at:
            return "lease has expired"
        return None

    def require_active(self, now: float) -> None:
        reason = self.active_reason(now)
        if reason:
            raise LeaseExpired(f"{self.id}: {reason}")

    def _event(self, kind: str, now: float, **kw: Any) -> None:
        self.events.append({"at": round(now, 3), "event": kind, **kw})

    # -- money ---------------------------------------------------------------

    def charge(self, usd: float, now: float, *, why: str) -> None:
        """Reserve ``usd`` against the cap. Refuses -- never clamps -- when the
        charge would cross it. A refusal leaves the lease unchanged."""
        self.require_active(now)
        if not isinstance(usd, (int, float)) or usd < 0:
            raise LeaseError(f"charge must be a non-negative number, got {usd!r}")
        if self.spent_usd + usd > self.cap_usd + 1e-9:
            raise BudgetExceeded(
                f"{self.id}: charge ${usd:.4f} would cross the cap "
                f"(spent ${self.spent_usd:.4f} of ${self.cap_usd:.4f}); refused, nothing ran")
        self.spent_usd = round(self.spent_usd + usd, 6)
        self._event("charge", now, usd=usd, why=why)

    def refund(self, usd: float, now: float, *, why: str) -> None:
        """Give back a reservation whose work never started."""
        usd = min(max(0.0, usd), self.spent_usd)
        self.spent_usd = round(self.spent_usd - usd, 6)
        self._event("refund", now, usd=usd, why=why)

    def close(self, now: float, *, why: str = "closed") -> None:
        if not self.closed:
            self.closed = True
            self._event("close", now, why=why)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Lease":
        return cls(**d)


def open_lease(pool: Pool, open_leases: List[Lease], *, contributor: str,
               approved_by: str, payer_kind: str, payer_id: str,
               credential_kind: str, credential_ref: str, cap_usd: float,
               ttl_s: int, targets: List[str], now: Optional[float] = None) -> Lease:
    """Mint a lease, or raise LeaseError naming the first rule it breaks."""
    now = time.time() if now is None else now
    pool.validate()
    if not contributor:
        raise LeaseError("a lease needs a contributor")
    if approved_by != pool.owner:
        raise LeaseError(f"lease must be approved by the pool owner {pool.owner!r}, "
                         f"got {approved_by!r}")
    if credential_kind in REFUSED_CREDENTIAL_KINDS:
        raise LeaseError("a consumer chat subscription is one person's and is never "
                         "shared; pool an API key instead")
    if credential_kind not in CREDENTIAL_KINDS:
        raise LeaseError(f"credential_kind must be one of {CREDENTIAL_KINDS}")
    ref = _check_ref(credential_ref)
    if credential_kind != "none" and not ref:
        raise LeaseError(f"credential_kind {credential_kind!r} needs a credential_ref")
    if payer_kind not in PAYER_KINDS:
        raise LeaseError(f"payer_kind must be one of {PAYER_KINDS}; there is no default "
                         "payer and no platform fallback")
    if not payer_id:
        raise LeaseError("payer_id is required: a lease says who pays")
    if payer_kind == "pool" and payer_id != pool.funder:
        raise LeaseError(f"a pool-paid lease must name the pool's funder {pool.funder!r}")
    if credential_kind == "pooled_api_key" and payer_kind != "pool":
        raise LeaseError("a pooled API key is paid by the pool; payer_kind must be 'pool'")
    if not (isinstance(cap_usd, (int, float)) and cap_usd > 0):
        raise LeaseError("cap_usd must be > 0; there is no unlimited lease")
    if cap_usd > pool.per_lease_cap_usd:
        raise LeaseError(f"cap ${cap_usd} exceeds the pool's per-lease cap "
                         f"${pool.per_lease_cap_usd}")
    if not (0 < int(ttl_s) <= min(pool.max_ttl_s, MAX_TTL_S)):
        raise LeaseError(f"ttl_s must be in (0, {min(pool.max_ttl_s, MAX_TTL_S)}]")
    if not targets:
        raise LeaseError("a lease names the targets it may run on")
    reserved = sum(le.cap_usd for le in open_leases
                   if le.pool == pool.id and le.active_reason(now) is None)
    if reserved + cap_usd > pool.budget_usd + 1e-9:
        raise BudgetExceeded(f"pool {pool.id}: ${reserved:.2f} already reserved of "
                             f"${pool.budget_usd:.2f}; a ${cap_usd:.2f} lease would overcommit it")
    lease = Lease(id="el-" + secrets.token_hex(6), pool=pool.id, contributor=contributor,
                  approved_by=approved_by, payer_kind=payer_kind, payer_id=payer_id,
                  credential_kind=credential_kind, credential_ref=ref,
                  cap_usd=float(cap_usd), created_at=now, expires_at=now + int(ttl_s),
                  targets=list(targets))
    lease._event("open", now, cap_usd=float(cap_usd), ttl_s=int(ttl_s))
    return lease
