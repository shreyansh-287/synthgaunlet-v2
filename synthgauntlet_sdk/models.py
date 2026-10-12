"""Small response records returned by the SDK. They never contain raw rows."""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Scorecard:
    run_id: str
    status: str
    final_status: str
    summary: str
    metrics: dict[str, Any] = field(default_factory=dict)


@dataclass
class Verification:
    status: str
    reason: str
    run_id: str
