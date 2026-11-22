"""awedgerunner -- Aither World Edgerunner.

Run agent and CI work on edge runners nobody has to rack, under a LEASE: a hard
budget cap, a named payer, an owner's approval and an expiry. awedgerunner owns
the lease and the routing; every runner target delegates to the piece that
already runs that kind of work. Standard library only.
"""

# Kept equal to pyproject's `version`.
__version__ = "0.1.0"

from .lease import (
    BudgetExceeded,
    Consent,
    Lease,
    LeaseError,
    LeaseExpired,
    Pool,
    open_lease,
)
from .store import Store
from .targets import Job, TargetRefused, default_targets, dispatch

__all__ = [
    "__version__",
    "BudgetExceeded",
    "Consent",
    "Job",
    "Lease",
    "LeaseError",
    "LeaseExpired",
    "Pool",
    "Store",
    "TargetRefused",
    "default_targets",
    "dispatch",
    "open_lease",
]
