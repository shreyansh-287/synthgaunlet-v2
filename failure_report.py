"""Diagnostic PDF issued when a run cannot release a synthetic CSV.

The report records the engine, every release-gate measurement, and the checks
that withheld the file. It is not a certificate and does not authorize download.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from certificate import (
    _GATE_ROWS,
    _INK,
    _MUTED,
    _NAVY,
    _PANEL,
    _RULE,
    _draw_frame,
    _draw_gate_table,
    _draw_header,
    _face,
    use_ui_fonts,
    _draw_identity_grid,
    _draw_paragraph,
    _draw_section_title,
    _format_count,
    _format_percent,
    _format_timestamp,
    _outcome,
    _panel,
)
import llm
from config import OUTPUT_DIR, get_settings

_EXPLANATIONS = {
    "fidelity_score": (
        "Distribution fidelity compares each synthetic column with the real training column. "
        "A low score means the synthetic values do not follow the real distribution closely enough."
    ),
    "utility_ratio": (
        "Model utility trains a predictor on the synthetic table and tests it on held-out real rows, "
        "then compares that accuracy with a model trained on real rows. A low score means the "
        "synthetic table did not preserve the relationship between the features and the label."
    ),
    "query_equivalence": (
        "Query equivalence runs the same grouped average on the real and synthetic tables. "
        "A low score means those aggregate answers disagree."
    ),
    "re_id_rate": (
        "Re-identification measures synthetic rows that land closer to a rare training person than "
        "a held-out real person typically does. A high score means the table can point back at individuals."
    ),
    "membership_accuracy": (
        "Membership inference tries to tell which real rows were used for training. Accuracy near 50% "
        "is expected. A higher score means the synthetic table leaks membership."
    ),
    "attribute_inference_advantage": (
        "Attribute inference measures how much the synthetic table improves a guess of a sensitive value. "
        "A high advantage means that secret is easier to infer."
    ),
    "linkage_rate": (
        "Auxiliary linkage counts unique public quasi-identifier combinations that match exactly one "
        "synthetic row. A high rate means an outside table could be joined back to this release."
    ),
    "near_duplicate_rate": (
        "The near-duplicate check counts synthetic rows that sit very close to a full training record. "
        "A high rate means too many rows are near copies."
    ),
}


def issue(run_id: str, response: dict, context: Optional[dict] = None) -> dict:
    """Persist the failure analysis and render its PDF. No synthetic rows are stored."""
    report = _build(run_id, response, context or {})
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / f"{run_id}.failure.json").write_text(
        json.dumps(report, indent=2, default=str), encoding="utf-8"
    )
    (OUTPUT_DIR / f"{run_id}.failure.pdf").write_bytes(render_pdf(report))
    return report


def load_report(run_id: str) -> Optional[dict]:
    path = OUTPUT_DIR / f"{run_id}.failure.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def failure_pdf(run_id: str) -> Optional[Path]:
    stored = load_report(run_id)
    path = OUTPUT_DIR / f"{run_id}.failure.pdf"
    if stored is not None:
        path.write_bytes(render_pdf(stored))
        return path
    return path if path.exists() else None


def render_pdf(report: dict) -> bytes:
    import io

    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas

    buffer = io.BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=letter)
    use_ui_fonts(pdf)
    pdf.setTitle(f"SynthGauntlet failure report {report.get('run_id', '')}")
    pdf.setAuthor("SynthGauntlet")
    pdf.setSubject("Synthetic data release withheld")
    writer = _Writer(pdf, report)
    writer.cover()
    writer.analysis()
    writer.history()
    writer.close()
    pdf.save()
    return buffer.getvalue()


def _build(run_id: str, response: dict, context: dict) -> dict:
    rounds = response.get("rounds") or []
    final = rounds[-1] if rounds else {}
    utility = final.get("utility") or {}
    attack = final.get("attack") or {}
    thresholds = response.get("thresholds") or {}
    scorecard = {
        "fidelity_score": utility.get("fidelity_score"),
        "utility_ratio": utility.get("utility_ratio"),
        "query_equivalence": utility.get("query_equivalence"),
        "re_id_rate": attack.get("re_id_rate"),
        "membership_accuracy": attack.get("membership_accuracy"),
        "attribute_inference_advantage": attack.get("attribute_inference_advantage"),
        "linkage_rate": attack.get("linkage_rate"),
        "near_duplicate_rate": attack.get("near_duplicate_rate"),
    }
    findings = []
    for label, key, threshold_key, direction, family in _GATE_ROWS:
        value = scorecard.get(key)
        limit = thresholds.get(threshold_key)
        outcome = _outcome(value, limit, direction)
        findings.append({
            "label": label,
            "key": key,
            "family": family,
            "value": value,
            "limit": limit,
            "direction": direction,
            "outcome": outcome,
            "explanation": _EXPLANATIONS[key],
        })
    mode = str(context.get("synthesis_mode") or "hybrid")
    guarantee = str(response.get("formal_dp_guarantee") or "No formal DP guarantee")
    return {
        "run_id": run_id,
        "issued_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "dataset_key": response.get("dataset_key") or context.get("dataset_key") or "",
        "domain": response.get("domain") or context.get("domain") or "",
        "policy_name": response.get("policy_name") or "Balanced",
        "policy_explanation": response.get("policy_explanation") or "",
        "synthesis_mode": mode,
        "formal_dp_guarantee": guarantee,
        "dp_epsilon": context.get("dp_epsilon"),
        "epsilon_spent": context.get("epsilon_spent"),
        "dp_queries": list(context.get("dp_queries") or []),
        "auto_heal_enabled": bool(response.get("auto_heal_enabled")),
        "final_status": response.get("final_status") or "NOT_CERTIFIED",
        "status": response.get("status") or "FAILED_TO_CONVERGE",
        "difficulty": response.get("difficulty") or "",
        "n_rows": context.get("n_rows"),
        "thresholds": thresholds,
        "scorecard": scorecard,
        "findings": findings,
        "rounds": [_round_row(item) for item in rounds],
        "rescue_attempts": [_rescue_row(item) for item in (response.get("rescue_attempts") or [])],
        "diagnostics": _diagnostics(utility),
        "tstr_detail": utility.get("tstr_detail") or {},
        "engine_summary": _engine_summary(mode, guarantee, context, response),
        "ai_narrative": (
            ""
            if context.get("skip_ai_narrative")
            else _ai_narrative(findings, mode, guarantee)
        ),
    }


def _round_row(item: dict) -> dict:
    utility = item.get("utility") or {}
    attack = item.get("attack") or {}
    plan = item.get("plan") or {}
    return {
        "round_index": item.get("round_index"),
        "plan_name": plan.get("plan_name") or "",
        "passed": bool(item.get("passed")),
        "fidelity_score": utility.get("fidelity_score"),
        "utility_ratio": utility.get("utility_ratio"),
        "query_equivalence": utility.get("query_equivalence"),
        "re_id_rate": attack.get("re_id_rate"),
        "membership_accuracy": attack.get("membership_accuracy"),
        "attribute_inference_advantage": attack.get("attribute_inference_advantage"),
        "linkage_rate": attack.get("linkage_rate"),
        "near_duplicate_rate": attack.get("near_duplicate_rate"),
        "feedback": item.get("orchestrator_feedback") or "",
    }


def _rescue_row(item: dict) -> dict:
    utility = item.get("utility") or {}
    attack = item.get("attack") or {}
    return {
        "attempt": item.get("attempt"),
        "name": item.get("name") or "",
        "passed": bool(item.get("passed")),
        "rows_kept": item.get("rows_kept"),
        "detail": item.get("detail") or "",
        "fidelity_score": utility.get("fidelity_score"),
        "utility_ratio": utility.get("utility_ratio"),
        "query_equivalence": utility.get("query_equivalence"),
        "re_id_rate": attack.get("re_id_rate"),
        "membership_accuracy": attack.get("membership_accuracy"),
        "attribute_inference_advantage": attack.get("attribute_inference_advantage"),
        "linkage_rate": attack.get("linkage_rate"),
        "near_duplicate_rate": attack.get("near_duplicate_rate"),
    }


def _diagnostics(utility: dict) -> dict:
    report = utility.get("diagnostic_report") or {}
    fairness = utility.get("fairness") or {}
    relationships = []
    for item in (report.get("broken_relationships") or [])[:4]:
        relationships.append(
            f"{item.get('col_a')} and {item.get('col_b')} moved from "
            f"{_signed(item.get('real_corr'))} to {_signed(item.get('synth_corr'))}."
        )
    groups = []
    for item in (fairness.get("groups") or [])[:4]:
        if item.get("status") == "WARN":
            groups.append(f"{item.get('attribute')} / {item.get('group')}")
    return {
        "weakest_columns": list(report.get("weakest_columns") or [])[:6],
        "broken_relationships": relationships,
        "fairness_status": fairness.get("status") or "",
        "fairness_summary": fairness.get("summary") or "",
        "fairness_warnings": groups,
    }


def _engine_summary(mode: str, guarantee: str, context: dict, response: dict) -> str:
    rescues = response.get("rescue_attempts") or []
    if mode == "dp":
        spent = context.get("epsilon_spent")
        total = context.get("dp_epsilon")
        queries = list(context.get("dp_queries") or [])
        budget = ""
        if spent is not None and total is not None:
            budget = f" The Laplace release spent epsilon {float(spent):.4f} of {float(total):.4f}"
            if queries:
                shown = ", ".join(queries[:8])
                extra = len(queries) - 8
                budget += f" across {len(queries)} marginal queries ({shown}"
                budget += f", and {extra} more" if extra > 0 else ""
                budget += ")."
            else:
                budget += "."
        if rescues:
            path = (
                "This run started on the Laplace engine. Auto-Heal then left that pure differential-privacy "
                "path and tested repaired tables. The final measurements below are from the last evaluated table."
            )
            formal = " The repaired table does not carry a formal differential-privacy guarantee."
        else:
            path = (
                "This run used the Laplace engine. The noisy marginals were fit once, and each round redrew "
                "rows from that same release. Later rounds did not spend additional epsilon."
            )
            formal = (
                f" The scorecard records: {guarantee}."
                if guarantee.startswith("Formal DP guarantee")
                else " The final table does not carry a formal differential-privacy guarantee."
            )
        return path + budget + formal
    rescue = (
        " Auto-Heal then searched privacy-safe redraws, generalization, and second-generation tables."
        if rescues else ""
    )
    return (
        "This run used the hybrid marginal engine. Each ordinary round compared up to three candidate "
        f"tables and kept the strongest one.{rescue} This path has no formal differential-privacy guarantee. "
        f"The scorecard records: {guarantee}."
    )


def _ai_narrative(findings: list[dict], mode: str, guarantee: str) -> str:
    failed = [
        f"{item['label']} measured {_format_percent(item['value'])}"
        for item in findings if item["outcome"] == "FAIL"
    ]
    if not failed:
        return ""
    engine = "Laplace" if mode == "dp" else "hybrid marginals"
    return llm.advise(
        "reporter",
        "Explain why the synthetic CSV was withheld.",
        (
            f"The {engine} run was not certified. Failed checks: {'; '.join(failed)}. "
            f"Privacy statement: {guarantee}. "
            "Explain the withholding in two sentences. Do not approve a release."
        ),
        model=get_settings().reporter_model,
    )


def _signed(value) -> str:
    try:
        return f"{float(value):.2f}"
    except (TypeError, ValueError):
        return "n/a"


class _Writer:
    def __init__(self, pdf, report: dict):
        self.pdf = pdf
        self.report = report
        self.page = 1
        self.width, self.height = pdf._pagesize
        self.left = 54
        self.right = self.width - 54
        self.y = self.height

    def cover(self) -> None:
        self._open("Synthetic data release withheld")
        self.pdf.setFillColor(_hex(_NAVY))
        self.pdf.setFont("Times-Bold", 24)
        self.pdf.drawCentredString(self.width / 2, self.y, "Release withheld")
        self.y -= 18
        self.pdf.setFillColor(_hex(_MUTED))
        self.pdf.setFont("Helvetica", 9.5)
        self.pdf.drawCentredString(self.width / 2, self.y, "Diagnostic report from the SynthGauntlet release gate")
        self.y -= 28
        self._status_pill("NOT CERTIFIED")
        failed = [item["label"].lower() for item in self.report["findings"] if item["outcome"] == "FAIL"]
        cause = ", ".join(failed) if failed else "the release gate"
        statement = (
            f"SynthGauntlet did not release a synthetic CSV for this run. The final table failed {cause}. "
            "Every number below was measured on the original training and held-out split. This document "
            "explains that decision. It is not a certificate and cannot be used to verify a data file."
        )
        self.y = _draw_paragraph(
            self.pdf, statement, self.left + 8, self.y, self.right - self.left - 16,
            "Times-Roman", 11, 14.5, _INK, centered=True,
        )
        self.y -= 12
        self.y = _draw_section_title(self.pdf, "Run identity", self.left, self.y)
        mode = "Laplace differential privacy" if self.report["synthesis_mode"] == "dp" else "Hybrid marginals"
        identity = [
            ("Report ID", str(self.report["run_id"])),
            ("Issued (UTC)", _format_timestamp(self.report["issued_at"])),
            ("Dataset", str(self.report["dataset_key"] or self.report["domain"] or "Uploaded table")),
            ("Release policy", str(self.report["policy_name"])),
            ("Synthesis engine", mode),
            ("Privacy statement", _privacy_label(self.report["formal_dp_guarantee"])),
            ("Rows requested", _format_count(self.report["n_rows"])),
            ("Tables evaluated", _format_count(len(self.report["rounds"]))),
        ]
        self.y = _draw_identity_grid(self.pdf, identity, self.left, self.y, self.right - self.left)
        self.y -= 14
        self.y = _draw_section_title(self.pdf, "Final release-gate results", self.left, self.y)
        self.y = _draw_gate_table(self.pdf, self.report, self.left, self.y, self.right - self.left)
        self.y -= 18
        self._callout(
            "The synthetic CSV, certificate, and BigQuery export were not made available. "
            "Only a table that passes every gate can be downloaded."
        )
        self.y -= 52

    def analysis(self) -> None:
        self._next("Why this synthetic table was not released")
        self.y = _draw_section_title(self.pdf, "Checks that withheld the release", self.left, self.y)
        failed = [item for item in self.report["findings"] if item["outcome"] == "FAIL"]
        if not failed:
            self.y = _draw_paragraph(
                self.pdf,
                "The stored final round has no individual gate failure. The run still ended uncertified, "
                "so the CSV was withheld.",
                self.left, self.y, self.right - self.left, "Helvetica", 9, 12, _INK,
            )
        for item in failed:
            self._ensure(62)
            word = "at least" if item["direction"] == "min" else "at most"
            headline = (
                f"{item['label']}: measured {_format_percent(item['value'])}, "
                f"required {word} {_format_percent(item['limit'])}."
            )
            self.y = self._finding(item["family"], headline, item["explanation"])
        self._ensure(90)
        self.y -= 6
        narrative = (self.report.get("ai_narrative") or "").strip()
        if narrative:
            self._ensure(70)
            self.y -= 4
            self.y = _draw_section_title(self.pdf, "Gemini reading of this failure", self.left, self.y)
            self.y = _draw_paragraph(
                self.pdf, narrative, self.left, self.y, self.right - self.left, "Times-Italic", 10, 13, _INK,
            )
            self.y -= 8
        self.y = _draw_section_title(self.pdf, "Synthesis engine", self.left, self.y)
        self.y = _draw_paragraph(
            self.pdf, self.report["engine_summary"], self.left, self.y,
            self.right - self.left, "Helvetica", 9, 12, _INK,
        )
        detail = self.report.get("tstr_detail") or {}
        if detail.get("task"):
            self.y -= 8
            self._ensure(48)
            features = ", ".join(str(item) for item in (detail.get("top_features") or [])[:5]) or "none recorded"
            text = (
                f"The prediction task was {detail.get('task')}. Real-data accuracy was "
                f"{_format_percent(detail.get('acc_real'))} and synthetic-trained accuracy was "
                f"{_format_percent(detail.get('acc_synth'))}. Leading features: {features}."
            )
            self.y = _draw_paragraph(self.pdf, text, self.left, self.y, self.right - self.left, "Helvetica", 9, 12, _INK)
        self._diagnostics()

    def history(self) -> None:
        self._next("Round-by-round measurements")
        self.y = _draw_section_title(self.pdf, "Every evaluated table", self.left, self.y)
        self.y = _draw_paragraph(
            self.pdf,
            "Ordinary rounds come first. Auto-Heal rounds, when present, continue the numbering. "
            "A PASS in this table is the full release gate, not a single metric.",
            self.left, self.y, self.right - self.left, "Helvetica", 8.5, 11, _MUTED,
        )
        self.y -= 8
        self._history_table(self.report["rounds"])
        attempts = self.report["rescue_attempts"]
        self.y -= 12
        self._ensure(70)
        self.y = _draw_section_title(self.pdf, "Auto-Heal", self.left, self.y)
        if not self.report["auto_heal_enabled"]:
            body = "Auto-Heal was turned off, so no rescue ladder ran after the ordinary rounds."
        elif not attempts:
            body = "Auto-Heal was enabled, but the ordinary rounds finished before a rescue was required."
        else:
            body = (
                f"Auto-Heal evaluated {len(attempts)} repair stage(s) after the ordinary rounds missed the gate. "
                "Each stage was scored again on the original real split."
            )
        self.y = _draw_paragraph(self.pdf, body, self.left, self.y, self.right - self.left, "Helvetica", 9, 12, _INK)
        for item in attempts:
            self._ensure(46)
            result = "passed the gate" if item["passed"] else "still failed the gate"
            headline = f"Attempt {item['attempt']} ({item['name']}): {result}."
            misses = _privacy_misses(item, self.report["thresholds"])
            utility = (
                f"Fidelity {_format_percent(item['fidelity_score'])}, prediction "
                f"{_format_percent(item['utility_ratio'])}, query {_format_percent(item['query_equivalence'])}."
            )
            if misses:
                utility += " Privacy misses: " + ", ".join(misses) + "."
            self.y = self._finding("Rescue", headline, utility)

    def close(self) -> None:
        self._footer()

    def _diagnostics(self) -> None:
        info = self.report["diagnostics"]
        lines = []
        if info["weakest_columns"]:
            lines.append("Weakest columns: " + ", ".join(info["weakest_columns"]) + ".")
        lines.extend(info["broken_relationships"])
        if info["fairness_summary"]:
            lines.append(f"Fairness {info['fairness_status']}: {info['fairness_summary']}")
        if info["fairness_warnings"]:
            lines.append("Fairness warnings: " + ", ".join(info["fairness_warnings"]) + ".")
        if not lines:
            return
        self.y -= 8
        self._ensure(50)
        self.y = _draw_section_title(self.pdf, "Additional analysis", self.left, self.y)
        self.y = _draw_paragraph(
            self.pdf,
            "Fairness warnings are advisory and do not by themselves withhold the CSV. "
            "Broken relationships show column pairs whose correlation changed sharply.",
            self.left, self.y, self.right - self.left, "Helvetica", 8.5, 11, _MUTED,
        )
        self.y -= 4
        for line in lines:
            self._ensure(24)
            self.y = _draw_paragraph(self.pdf, line, self.left, self.y, self.right - self.left, "Helvetica", 9, 12, _INK)
            self.y -= 2

    def _history_table(self, rows: list[dict]) -> None:
        from reportlab.lib.colors import white

        headers = ("ROUND", "FIDELITY", "PREDICT", "QUERY", "RE-ID", "MEMBER", "RESULT")
        offsets = (0.0, 0.14, 0.30, 0.46, 0.62, 0.76, 0.90)
        width = self.right - self.left
        header_height, row_height = 18, 15

        def draw_header() -> float:
            self._ensure(header_height + row_height)
            self.pdf.setFillColor(_hex(_NAVY))
            self.pdf.roundRect(self.left, self.y - header_height + 4, width, header_height, 5, fill=1, stroke=0)
            self.pdf.setFillColor(white)
            self.pdf.setFont("Helvetica-Bold", 7)
            for label, offset in zip(headers, offsets):
                self.pdf.drawString(self.left + 8 + offset * width, self.y - 8, label)
            return self.y - header_height + 4

        cursor = draw_header()
        for index, row in enumerate(rows):
            if cursor - row_height < 68:
                self._footer()
                self.pdf.showPage()
                self.page += 1
                self._open("Round-by-round measurements")
                cursor = draw_header()
            if index % 2 == 0:
                self.pdf.setFillColor(_hex(_PANEL))
                self.pdf.rect(self.left, cursor - row_height, width, row_height, fill=1, stroke=0)
            values = (
                str(row.get("round_index") or ""),
                _format_percent(row.get("fidelity_score")),
                _format_percent(row.get("utility_ratio")),
                _format_percent(row.get("query_equivalence")),
                _format_percent(row.get("re_id_rate")),
                _format_percent(row.get("membership_accuracy")),
                "PASS" if row.get("passed") else "FAIL",
            )
            baseline = cursor - 10.5
            for value, offset in zip(values, offsets):
                self.pdf.setFillColor(_hex("#065f46" if value == "PASS" else "#9f1239" if value == "FAIL" else _INK))
                self.pdf.setFont("Helvetica-Bold" if value in {"PASS", "FAIL"} else "Helvetica", 7.5)
                self.pdf.drawString(self.left + 8 + offset * width, baseline, value)
            cursor -= row_height
        self.pdf.setStrokeColor(_hex(_RULE))
        self.pdf.setLineWidth(0.7)
        self.pdf.line(self.left, cursor, self.right, cursor)
        self.y = cursor - 8

    def _finding(self, family: str, headline: str, body: str) -> float:
        from reportlab.lib.colors import HexColor

        color = "#be123c" if family != "Rescue" else "#b45309"
        self.pdf.setFillColor(HexColor(color))
        self.pdf.circle(self.left + 4, self.y + 2, 2.4, fill=1, stroke=0)
        y = _draw_paragraph(self.pdf, headline, self.left + 14, self.y, self.right - self.left - 14, "Helvetica-Bold", 9, 12, _NAVY)
        y = _draw_paragraph(self.pdf, body, self.left + 14, y - 1, self.right - self.left - 14, "Helvetica", 8.6, 11.2, _INK)
        return y - 8

    def _callout(self, text: str) -> None:
        from reportlab.lib.colors import HexColor

        self._ensure(48)
        self.pdf.setFillColor(HexColor("#fff1f2"))
        self.pdf.setStrokeColor(HexColor("#fb7185"))
        self.pdf.setLineWidth(1)
        self.pdf.roundRect(self.left, self.y - 36, self.right - self.left, 44, 8, fill=1, stroke=1)
        _draw_paragraph(self.pdf, text, self.left + 12, self.y - 4, self.right - self.left - 24, "Helvetica", 8.5, 11, "#9f1239")

    def _status_pill(self, label: str) -> None:
        from reportlab.lib.colors import HexColor
        from reportlab.pdfbase.pdfmetrics import stringWidth

        text_width = stringWidth(label, _face("Helvetica-Bold"), 9.5)
        pill = text_width + 28
        x = self.width / 2 - pill / 2
        self.pdf.setFillColor(HexColor("#ffe4e6"))
        self.pdf.setStrokeColor(HexColor("#e11d48"))
        self.pdf.setLineWidth(1)
        self.pdf.roundRect(x, self.y - 6, pill, 20, 10, fill=1, stroke=1)
        self.pdf.setFillColor(HexColor("#9f1239"))
        self.pdf.setFont("Helvetica-Bold", 9.5)
        self.pdf.drawCentredString(self.width / 2, self.y, label)
        self.y -= 28

    def _open(self, subtitle: str) -> None:
        _draw_frame(self.pdf, self.width, self.height)
        _draw_header(self.pdf, self.report, self.width, self.height, subtitle, "REPORT ID")
        self.y = self.height - 148

    def _next(self, subtitle: str) -> None:
        self._footer()
        self.pdf.showPage()
        self.page += 1
        self._open(subtitle)

    def _ensure(self, needed: float) -> None:
        if self.y - needed < 68:
            self._next("Why this synthetic table was not released")

    def _footer(self) -> None:
        from reportlab.lib.colors import HexColor

        self.pdf.setStrokeColor(HexColor(_RULE))
        self.pdf.setLineWidth(0.6)
        self.pdf.line(54, 52, self.width - 54, 52)
        self.pdf.setFillColor(HexColor(_MUTED))
        self.pdf.setFont("Helvetica", 7.5)
        self.pdf.drawString(54, 40, "SynthGauntlet failure report. The synthetic CSV was not released.")
        self.pdf.drawRightString(self.width - 54, 40, f"Page {self.page}")


def _privacy_label(guarantee: str) -> str:
    if str(guarantee).startswith("Formal DP guarantee"):
        return str(guarantee).replace("Formal DP guarantee: ", "Formal, ")
    return "No formal DP guarantee"


def _privacy_misses(row: dict, thresholds: dict) -> list[str]:
    checks = (
        ("re_id_rate", "reid_max", "re-identification"),
        ("membership_accuracy", "membership_max", "membership"),
        ("attribute_inference_advantage", "attribute_inference_max", "attribute inference"),
        ("linkage_rate", "linkage_max", "linkage"),
        ("near_duplicate_rate", "near_duplicate_max", "near-duplicate"),
    )
    missed = []
    for key, limit_key, label in checks:
        value, limit = row.get(key), thresholds.get(limit_key)
        if value is None or limit is None:
            continue
        if float(value) > float(limit):
            missed.append(f"{label} {_format_percent(value)}")
    return missed


def _hex(value: str):
    from reportlab.lib.colors import HexColor

    return HexColor(value)
