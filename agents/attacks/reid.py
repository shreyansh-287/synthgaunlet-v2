"""Re-identification adapter. The distance math stays in attack_tools."""

from __future__ import annotations

from agents import attack_tools
from agents.attacks.base import BaseAttack, PluginAttackResult


class ReidentificationAttack(BaseAttack):
    name = "reidentification"

    def run(self, real_train, real_holdout, synth, profile) -> PluginAttackResult:
        columns = [column.name for column in profile.columns if column.is_quasi_identifier]
        result = attack_tools.run_reidentification(
            real_train, synth, columns, threshold=0.1, real_holdout=real_holdout
        )
        score = float(result.get("re_id_rate", 0.0))
        return PluginAttackResult(self.name, score, True, "DCR re-identification.", result)
