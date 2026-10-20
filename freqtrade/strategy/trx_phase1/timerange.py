"""UTC date windows used by TRX research requests."""

import re
from dataclasses import dataclass
from datetime import UTC, datetime


@dataclass(frozen=True)
class TimeRange:
    startdt: datetime | None = None
    stopdt: datetime | None = None

    @classmethod
    def parse_timerange(cls, text):
        if not text:
            return cls()
        parts = text.split("-")
        if len(parts) != 2 or not any(parts):
            raise ValueError("Invalid timerange")
        start, stop = (parse_boundary(part) for part in parts)
        if start and stop and start > stop:
            raise ValueError("Timerange start is after end")
        return cls(start, stop)


def parse_boundary(text):
    if not text:
        return None
    for pattern, date_format in (
        (r"\d{8}", "%Y%m%d"),
        (r"\d{8}T\d{4}", "%Y%m%dT%H%M"),
        (r"\d{8}T\d{6}", "%Y%m%dT%H%M%S"),
    ):
        if re.fullmatch(pattern, text):
            return datetime.strptime(text, date_format).replace(tzinfo=UTC)
    if re.fullmatch(r"\d{10}|\d{13}", text):
        seconds = int(text) // 1000 if len(text) == 13 else int(text)
        return datetime.fromtimestamp(seconds, UTC)
    raise ValueError("Invalid timerange boundary")
