from collections import deque
from time import monotonic, sleep

class RateLimiter:
    def __init__(self, rate: int, period: float):
        self.rate = rate
        self.period = period
        self._timestamps: deque[float] = deque()

    def wait(self) -> None:
        while True:
            now = monotonic()
            while self._timestamps and now - self._timestamps[0] >= self.period:
                self._timestamps.popleft()

            if len(self._timestamps) < self.rate:
                break

            sleep_for = self.period - (now - self._timestamps[0])
            if sleep_for > 0:
                sleep(sleep_for)

        self._timestamps.append(monotonic())