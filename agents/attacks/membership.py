"""Membership-inference adapter. The distance math stays in attack_tools."""

from __future__ import annotations

from agents import attack_tools
from agents.attacks.base import BaseAttack, PluginAttackResult


class MembershipAttack(BaseAttack):
    name = "membership"

    def run(self, real_train, real_holdout, synth, profile) -> PluginAttackResult:
        columns = [column.name for column in profile.columns]
        result = attack_tools.run_membership_inference(real_train, real_holdout, synth, columns)
        score = float(result.get("membership_accuracy", 0.5))
        return PluginAttackResult(self.name, score, True, "Membership inference.", result)
