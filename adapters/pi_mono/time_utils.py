from __future__ import annotations

from datetime import datetime, timezone

DEFAULT_TIMESTAMP = datetime(1970, 1, 1, tzinfo=timezone.utc)


def parse_pi_datetime(value: object, fallback: datetime | None = DEFAULT_TIMESTAMP) -> datetime | None:
    """Parse Pi ISO timestamps and Unix-ms numeric timestamps."""
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(value / 1000, tz=timezone.utc)
    if isinstance(value, str) and value:
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
            return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
        except ValueError:
            pass
    return fallback
