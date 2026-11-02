"""Shared pacing for the one provider key. The per-call input cap is not a rate limit: several visitors can arrive in the same minute, so
calls are admitted one at a time and against the provider's tokens-per-minute window, and a visitor who cannot be admitted is told when to retry."""
import time
from collections import deque


class Governor:
    """Admits a call if none is in flight and the estimated tokens in the last minute plus this call stay within `share` of the limit.

    Not thread-safe by design: the service runs on one event loop and nothing awaits between `admit` and the call it guards."""

    def __init__(self, tokens_per_minute: int, share: float = 0.9, concurrent: int = 1, now=time.monotonic):
        self.limit, self.concurrent, self.now = int(tokens_per_minute * share), concurrent, now
        self.window: deque = deque()  # (start time, tokens) of calls in the last minute
        self.inflight = 0

    def _expire(self) -> None:
        while self.window and self.now() - self.window[0][0] >= 60:
            self.window.popleft()

    def admit(self, tokens: int) -> tuple[object | None, float]:
        """A ticket for the call, or (None, seconds to wait). A call larger than the whole window is never admitted."""
        self._expire()
        if self.inflight >= self.concurrent:
            return None, 3.0
        used, ticket = sum(t for _, t in self.window), [self.now(), tokens]
        if used + tokens > self.limit:
            need, freed = used + tokens - self.limit, 0
            for start, t in self.window:
                freed += t
                if freed >= need:
                    return None, max(1.0, round(60 - (self.now() - start), 1))
            return None, 60.0
        self.window.append(ticket)
        self.inflight += 1
        return ticket, 0.0

    def release(self, ticket, actual: int | None = None) -> None:
        """End a call; replace the reserved tokens with what it actually used if known."""
        self.inflight = max(0, self.inflight - 1)
        if actual is not None:
            ticket[1] = actual
