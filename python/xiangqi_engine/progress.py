"""Human-readable elapsed / ETA lines for the training loop."""

from __future__ import annotations

import time
from datetime import datetime, timedelta


def format_hms(seconds: float | None) -> str:
    if seconds is None:
        return "?"
    if seconds < 0:
        seconds = 0
    total = int(round(seconds))
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h{m:02d}m{s:02d}s"
    if m:
        return f"{m}m{s:02d}s"
    return f"{s}s"


def eta_seconds(done: int, total: int, elapsed: float) -> float | None:
    if done <= 0 or total <= 0:
        return None
    if done >= total:
        return 0.0
    return elapsed * (total - done) / done


def finish_clock(eta_sec: float | None) -> str:
    if eta_sec is None:
        return "?"
    return (datetime.now() + timedelta(seconds=eta_sec)).strftime("%m-%d %H:%M")


def progress_line(prefix: str, done: int, total: int, started_at: float, extra: str = "") -> str:
    elapsed = time.time() - started_at
    eta = eta_seconds(done, total, elapsed)
    pct = 100.0 * done / total if total else 100.0
    parts = [
        f"{prefix}: {done}/{total} ({pct:.0f}%)",
        f"elapsed {format_hms(elapsed)}",
        f"eta {format_hms(eta)}",
    ]
    if eta is not None and done < total:
        parts.append(f"finish {finish_clock(eta)}")
    if extra:
        parts.append(extra)
    return " ".join(parts)


class Progress:
    """Print every N items, or at least every ``interval`` seconds."""

    def __init__(self, prefix: str, total: int, interval: float = 30.0):
        self.prefix = prefix
        self.total = max(int(total), 0)
        self.interval = float(interval)
        self.started_at = time.time()
        self._last_print = 0.0

    def update(self, done: int, extra: str = "") -> None:
        now = time.time()
        step = max(1, self.total // 16) if self.total else 1
        due = (
            done <= 1
            or done >= self.total
            or (step and done % step == 0)
            or now - self._last_print >= self.interval
        )
        if not due:
            return
        self._last_print = now
        print(progress_line(self.prefix, done, self.total, self.started_at, extra), flush=True)
