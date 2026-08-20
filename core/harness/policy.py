"""Retry, backoff, and deadline policy for the harness.

Retry logic lives here rather than inside the LLM/STT clients on purpose: the
clients stay thin transport wrappers, and the orchestrator owns *when* it is
worth spending more of the budget on another attempt. A client that retries
internally cannot know it has 12 ms of budget left.
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass, field

import httpx

# Transient by nature: worth another attempt.
RETRYABLE_STATUS = frozenset({408, 409, 425, 429, 500, 502, 503, 504})
RETRYABLE_EXCEPTIONS = (
    httpx.TimeoutException,
    httpx.ConnectError,
    httpx.ReadError,
    httpx.RemoteProtocolError,
)


class DeadlineExceeded(RuntimeError):
    """Raised when the remaining budget cannot cover the next attempt."""


@dataclass
class Deadline:
    """Monotonic wall-clock budget for one run."""

    budget_ms: float
    _start: float = field(default_factory=time.perf_counter)

    def elapsed_ms(self) -> float:
        return (time.perf_counter() - self._start) * 1000.0

    def remaining_ms(self) -> float:
        return self.budget_ms - self.elapsed_ms()

    def expired(self) -> bool:
        return self.remaining_ms() <= 0.0

    def check(self, stage: str) -> None:
        if self.expired():
            raise DeadlineExceeded(f"budget exhausted before {stage}")


@dataclass
class RetryPolicy:
    max_attempts: int = 3
    base_delay: float = 0.25
    max_delay: float = 2.0
    jitter: float = 0.1

    def should_retry(self, exc: BaseException) -> bool:
        if isinstance(exc, RETRYABLE_EXCEPTIONS):
            return True
        if isinstance(exc, httpx.HTTPStatusError):
            return exc.response.status_code in RETRYABLE_STATUS
        return False

    def delay_for(self, attempt: int, *, rng: random.Random | None = None) -> float:
        """Exponential backoff with full-width jitter, capped at max_delay.

        Jitter matters when several clients retry against the same upstream:
        without it they resynchronize and hammer it in lockstep.
        """
        rand = rng or random
        raw = min(self.base_delay * (2 ** max(attempt - 1, 0)), self.max_delay)
        return max(0.0, raw + rand.uniform(-self.jitter, self.jitter) * raw)


def call_with_retry(
    fn,
    *,
    policy: RetryPolicy,
    deadline: Deadline | None = None,
    sleep=time.sleep,
    rng: random.Random | None = None,
) -> tuple[object, int]:
    """Run ``fn`` under the retry policy. Returns (result, attempts_used).

    Stops early, without sleeping, when the remaining deadline could not
    accommodate the backoff, so a doomed retry never eats budget the fallback
    path still needs.
    """
    last_exc: BaseException | None = None
    for attempt in range(1, policy.max_attempts + 1):
        try:
            return fn(), attempt
        except BaseException as exc:
            last_exc = exc
            if attempt >= policy.max_attempts or not policy.should_retry(exc):
                raise
            delay = policy.delay_for(attempt, rng=rng)
            if deadline is not None and deadline.remaining_ms() <= delay * 1000.0:
                raise DeadlineExceeded(
                    f"no budget for retry {attempt + 1}/{policy.max_attempts}"
                ) from exc
            sleep(delay)
    raise last_exc  # pragma: no cover - loop always returns or raises
