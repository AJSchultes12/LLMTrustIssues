from __future__ import annotations

import json
import math
import os
import tempfile
from pathlib import Path

from llmtrust.models import Arm


class BudgetExceeded(Exception):
    pass


def estimate_tokens(text: str) -> int:
    return math.ceil(len(text) / 4)


class Budget:
    """Per-pool spend tracker persisted as JSON ({"spent": {pool: usd}})."""

    def __init__(self, caps: dict[str, float], state_path: Path):
        self.caps = dict(caps)
        self.state_path = Path(state_path)
        self._spent: dict[str, float] = {}
        if self.state_path.exists():
            data = json.loads(self.state_path.read_text(encoding="utf-8"))
            self._spent = {k: float(v) for k, v in data.get("spent", {}).items()}

    def spent(self, pool: str) -> float:
        return self._spent.get(pool, 0.0)

    def estimate_usd(self, arm: Arm, in_tokens: int, out_tokens: int) -> float:
        if arm.pool == "free":
            return 0.0
        if arm.usd_per_mtok_in is None or arm.usd_per_mtok_out is None:
            raise ValueError(
                f"arm {arm.name!r} (pool {arm.pool!r}) has no price: the price must be "
                f"filled in config/models.yaml from the provider's pricing page")
        return (in_tokens * arm.usd_per_mtok_in + out_tokens * arm.usd_per_mtok_out) / 1_000_000

    def check(self, pool: str, usd: float) -> None:
        """Raise BudgetExceeded if usd would pass the pool's cap. Pool "free" has no cap;
        any other pool without one fails closed with ValueError."""
        if pool == "free":
            return
        cap = self.caps.get(pool)
        if cap is None:
            raise ValueError(f"pool {pool!r} has no cap: set caps.{pool} in "
                             f"config/models.yaml before spending from it")
        if self.spent(pool) + usd > cap:
            raise BudgetExceeded(
                f"pool {pool!r}: spent ${self.spent(pool):.4f} + ${usd:.4f} "
                f"would exceed cap ${cap:.2f}")

    def record(self, pool: str, usd: float) -> None:
        self._spent[pool] = self.spent(pool) + usd
        self._save()

    def _save(self) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=self.state_path.parent, suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump({"spent": self._spent}, f, indent=2)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, self.state_path)
        except BaseException:
            if os.path.exists(tmp):
                os.unlink(tmp)
            raise
