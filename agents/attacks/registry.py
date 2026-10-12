"""Run the extended attack plugins and collect their scores."""

from __future__ import annotations

from typing import Dict, List

import pandas as pd

from agents.attacks.attribute_inference import AttributeInferenceAttack
from agents.attacks.base import BaseAttack, PluginAttackResult
from agents.attacks.linkage import LinkageAttack
from agents.attacks.membership import MembershipAttack
from agents.attacks.near_duplicate import NearDuplicateAttack
from agents.attacks.reid import ReidentificationAttack
from models import DatasetProfile


def all_attacks() -> List[BaseAttack]:
    return [
        ReidentificationAttack(),
        MembershipAttack(),
        AttributeInferenceAttack(),
        LinkageAttack(),
        NearDuplicateAttack(),
    ]


def extended_attacks() -> List[BaseAttack]:
    """Attacks added on top of the existing re-identification and membership passes."""
    return [AttributeInferenceAttack(), LinkageAttack(), NearDuplicateAttack()]


def run_attacks(
    attacks: List[BaseAttack],
    real_train: pd.DataFrame,
    real_holdout: pd.DataFrame,
    synth: pd.DataFrame,
    profile: DatasetProfile,
) -> Dict[str, PluginAttackResult]:
    results = {}
    for attack in attacks:
        results[attack.name] = attack.run(real_train, real_holdout, synth, profile)
    return results
