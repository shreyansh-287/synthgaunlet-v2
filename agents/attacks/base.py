"""Shared contract for a privacy attack plugin."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict

import pandas as pd

from models import DatasetProfile


@dataclass
class PluginAttackResult:
    name: str
    score: float
    passed: bool
    detail: str = ""
    metrics: Dict[str, Any] = field(default_factory=dict)


class BaseAttack(ABC):
    name: str = "attack"

    @abstractmethod
    def run(
        self,
        real_train: pd.DataFrame,
        real_holdout: pd.DataFrame,
        synth: pd.DataFrame,
        profile: DatasetProfile,
    ) -> PluginAttackResult:
        """Score leakage. Higher scores are worse for the synthetic table."""
