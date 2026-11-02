"""Application-enforced spending caps. A call is reserved at its worst-case cost before it is made and settled afterwards."""
import json
import threading
from datetime import UTC, datetime
from pathlib import Path


class BudgetExceeded(Exception):
    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


def _today() -> str:
    return datetime.now(UTC).date().isoformat()


class Budget:
    """Spend in USD, persisted to a JSON file so the cumulative cap survives restarts. Days are UTC."""

    def __init__(self, caps: dict, prices: dict, path: Path | None = None, today=_today):
        self.caps, self.prices, self.path, self.today = caps, prices, path, today
        self.lock, self.pending = threading.Lock(), 0.0
        self.state = {"total": 0.0, "days": {}}
        if path and path.exists():
            self.state = json.loads(path.read_text("utf-8"))

    def cost(self, tokens_in: float, tokens_out: float) -> float:
        return (tokens_in * self.prices["input"] + tokens_out * self.prices["output"]) / 1e6

    def spent_today(self) -> float:
        return self.state["days"].get(self.today(), 0.0)

    def reserve(self, tokens_in: int, max_out: int) -> float:
        """Hold the worst-case cost of one call; raise BudgetExceeded if either cap would be passed."""
        worst = self.cost(tokens_in, max_out)
        with self.lock:
            if self.state["total"] + self.pending + worst > self.caps["total"]:
                raise BudgetExceeded("budget_total")
            if self.spent_today() + self.pending + worst > self.caps["daily"]:
                raise BudgetExceeded("budget_daily")
            self.pending += worst
        return worst

    def settle(self, held: float, tokens_in: int, tokens_out: int) -> float:
        actual = self.cost(tokens_in, tokens_out)
        with self.lock:
            self.pending = max(0.0, self.pending - held)
            day = self.today()
            self.state["total"] += actual
            self.state["days"][day] = self.state["days"].get(day, 0.0) + actual
            if self.path:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                tmp = self.path.with_suffix(".tmp")
                tmp.write_text(json.dumps(self.state), "utf-8")
                tmp.replace(self.path)
        return actual
