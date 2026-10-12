"""Signed release certificate. The signature records technical evidence; it is not a legal approval."""

from __future__ import annotations

import hashlib
import hmac
import io
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd

from config import OUTPUT_DIR, get_settings

_PDF_RECORD = b"% SYNTHGAUNTLET-CERTIFICATE "

EVIDENCE_STATEMENT = (
    "This certificate supports technical evidence for privacy principles such as data minimization "
    "and re-identification testing. It is not a legal HIPAA or GDPR certification."
)


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def frame_csv_sha256(frame: pd.DataFrame) -> str:
    return sha256_bytes(frame.to_csv(index=False).encode("utf-8"))


def issue(run_id: str, response: dict, synthetic: pd.DataFrame, training: pd.DataFrame) -> dict:
    synthetic_bytes = synthetic.to_csv(index=False).encode("utf-8")
    payload = {
        "run_id": run_id,
        "issued_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "dataset_rows": int(len(synthetic)),
        "training_rows": int(len(training)),
        "synthetic_sha256": sha256_bytes(synthetic_bytes),
        "training_sha256": frame_csv_sha256(training),
        "engine_mode": response.get("formal_dp_guarantee", "No formal DP guarantee"),
        "final_status": response.get("final_status", ""),
        "policy_name": response.get("policy_name", "Balanced"),
        "thresholds": response.get("thresholds", {}),
        "scorecard": _compact_scorecard(response),
        "process": _process_summary(response),
        "evidence_statement": EVIDENCE_STATEMENT,
        "verify_url": f"{get_settings().public_base_url.rstrip('/')}/api/v1/verify/{run_id}",
    }
    body = canonical_bytes(payload)
    certificate = {**payload, "signature": _sign(body)}
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / f"{run_id}.certified.csv").write_bytes(synthetic_bytes)
    (OUTPUT_DIR / f"{run_id}.certificate.json").write_text(
        json.dumps(certificate, indent=2), encoding="utf-8"
    )
    (OUTPUT_DIR / f"{run_id}.certificate.pdf").write_bytes(render_pdf(certificate))
    return certificate


def load_certificate(run_id: str) -> Optional[dict]:
    path = OUTPUT_DIR / f"{run_id}.certificate.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def certificate_pdf(run_id: str) -> Optional[Path]:
    path = OUTPUT_DIR / f"{run_id}.certificate.pdf"
    stored = load_certificate(run_id)
    if stored is not None:
        path.write_bytes(render_pdf(stored))
        return path
    return path if path.exists() else None


def load_certificate_document(raw: bytes) -> dict:
    """Read a certificate from certificate.json or from a downloaded certificate PDF."""
    stripped = raw.lstrip()
    if stripped.startswith(b"{") or stripped.startswith(b"["):
        parsed = json.loads(raw)
        if not isinstance(parsed, dict):
            raise ValueError("The certificate file must be a JSON object.")
        return parsed
    if raw.startswith(b"%PDF"):
        start = raw.rfind(_PDF_RECORD)
        if start < 0:
            raise ValueError("This PDF has no embedded certificate. Download the certificate again, then upload that PDF.")
        line = raw[start + len(_PDF_RECORD):].splitlines()[0]
        parsed = json.loads(line)
        if not isinstance(parsed, dict):
            raise ValueError("The certificate embedded in this PDF is not valid.")
        return parsed
    raise ValueError("Upload the certificate PDF from Download certificate, or a certificate.json file.")


def verify(csv_bytes: bytes, certificate: dict) -> dict:
    signature = str(certificate.get("signature", ""))
    payload = {key: value for key, value in certificate.items() if key != "signature"}
    signature_ok = hmac.compare_digest(signature, _sign(canonical_bytes(payload)))
    fingerprint = sha256_bytes(csv_bytes)
    fingerprint_ok = hmac.compare_digest(fingerprint, str(certificate.get("synthetic_sha256", "")))
    status = "VALID" if signature_ok and fingerprint_ok else "TAMPERED"
    reason = "The signature and synthetic-file fingerprint match."
    if not signature_ok:
        reason = "The certificate signature does not match this signing key."
    elif not fingerprint_ok:
        reason = "Digital fingerprint mismatch."
    return {
        "status": status,
        "reason": reason,
        "run_id": certificate.get("run_id", ""),
        "evidence_statement": EVIDENCE_STATEMENT,
    }


def canonical_bytes(payload: dict) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")


def _sign(payload: bytes) -> str:
    key = get_settings().cert_signing_key.encode("utf-8")
    return hmac.new(key, payload, hashlib.sha256).hexdigest()


def _compact_scorecard(response: dict) -> dict:
    rounds = response.get("rounds") or []
    final = rounds[-1] if rounds else {}
    utility = final.get("utility") or {}
    attack = final.get("attack") or {}
    return {
        "fidelity_score": utility.get("fidelity_score"),
        "utility_ratio": utility.get("utility_ratio"),
        "query_equivalence": utility.get("query_equivalence"),
        "re_id_rate": attack.get("re_id_rate"),
        "membership_accuracy": attack.get("membership_accuracy"),
        "attribute_inference_advantage": attack.get("attribute_inference_advantage"),
        "linkage_rate": attack.get("linkage_rate"),
        "near_duplicate_rate": attack.get("near_duplicate_rate"),
    }


def _process_summary(response: dict) -> dict:
    rounds = response.get("rounds") or []
    thresholds = response.get("thresholds") or {}
    return {
        "rounds_evaluated": len(rounds),
        "max_rounds": thresholds.get("max_rounds"),
        "candidates_evaluated": sum(len(item.get("evaluated_candidates") or []) for item in rounds),
        "rescue_attempts": len(response.get("rescue_attempts") or []),
        "auto_heal_enabled": bool(response.get("auto_heal_enabled", True)),
        "llm_enabled": bool(response.get("llm_enabled", False)),
    }


_NAVY = "#0f172a"
_INK = "#1e293b"
_MUTED = "#64748b"
_RULE = "#e2e8f0"
_PANEL = "#f8fafc"
_INDIGO = "#6366f1"
_PURPLE = "#9333ea"
_DEEP_INDIGO = "#4338ca"

# (label, scorecard key, threshold key, direction, family)
_GATE_ROWS = (
    ("Distribution fidelity", "fidelity_score", "fidelity_min", "min", "Utility"),
    ("Model utility (TSTR)", "utility_ratio", "tstr_min", "min", "Utility"),
    ("Query equivalence", "query_equivalence", "query_equivalence_min", "min", "Utility"),
    ("Re-identification risk", "re_id_rate", "reid_max", "max", "Privacy"),
    ("Membership inference", "membership_accuracy", "membership_max", "max", "Privacy"),
    ("Attribute inference advantage", "attribute_inference_advantage", "attribute_inference_max", "max", "Privacy"),
    ("Auxiliary linkage", "linkage_rate", "linkage_max", "max", "Privacy"),
    ("Near-duplicate rows", "near_duplicate_rate", "near_duplicate_max", "max", "Privacy"),
)


def render_pdf(certificate: dict) -> bytes:
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas

    buffer = io.BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=letter)
    pdf.setTitle(f"SynthGauntlet Release Certificate {certificate.get('run_id', '')}")
    pdf.setAuthor("SynthGauntlet")
    pdf.setSubject("Synthetic data release certificate")
    _draw_certificate_page(pdf, certificate)
    pdf.showPage()
    _draw_method_page(pdf, certificate)
    pdf.showPage()
    pdf.save()
    document = buffer.getvalue()
    record = json.dumps(certificate, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    eof = document.rfind(b"%%EOF")
    if eof < 0:
        return document + b"\n" + _PDF_RECORD + record + b"\n"
    return document[:eof] + _PDF_RECORD + record + b"\n" + document[eof:]


def _draw_certificate_page(pdf, certificate: dict) -> None:
    from reportlab.lib.colors import HexColor

    width, height = pdf._pagesize
    left, right = 54, width - 54
    _draw_frame(pdf, width, height)
    _draw_header(pdf, certificate, width, height, "Synthetic Data Release Certificate")

    y = height - 152
    pdf.setFillColor(HexColor(_NAVY))
    pdf.setFont("Times-Bold", 25)
    pdf.drawCentredString(width / 2, y, "Certificate of Synthetic Data Release")
    y -= 17
    pdf.setFillColor(HexColor(_MUTED))
    pdf.setFont("Helvetica", 9.5)
    pdf.drawCentredString(width / 2, y, "Issued by the SynthGauntlet deterministic release gate")

    y -= 27
    status = str(certificate.get("final_status") or "CERTIFIED")
    _draw_status_pill(pdf, width / 2, y, _status_label(status))

    y -= 27
    policy = certificate.get("policy_name") or "Balanced"
    rescue_clause = (
        " after an Auto-Heal rescue attempt"
        if status == "CERTIFIED_AFTER_RESCUE" else ""
    )
    statement = (
        "This certifies that the synthetic dataset identified by the SHA-256 fingerprint in this "
        "document was produced by SynthGauntlet and passed every utility and adversarial privacy "
        f"check required by the {policy} release policy{rescue_clause}. Each result below was "
        "measured against held-out real data and approved by deterministic code at the time of issue."
    )
    y = _draw_paragraph(pdf, statement, left + 18, y, right - left - 36, "Times-Roman", 11.5, 15.5, _INK, centered=True)

    y -= 12
    y = _draw_section_title(pdf, "Release identity", left, y)
    process = certificate.get("process") or {}
    identity = [
        ("Certificate ID", str(certificate.get("run_id", ""))),
        ("Issued (UTC)", _format_timestamp(certificate.get("issued_at"))),
        ("Release policy", str(policy)),
        ("Differential privacy", str(certificate.get("engine_mode") or "No formal DP guarantee").replace("Formal DP guarantee: ", "Formal, ")),
        ("Synthetic rows", _format_count(certificate.get("dataset_rows"))),
        ("Training rows", _format_count(certificate.get("training_rows"))),
        ("Rounds evaluated", _format_count(process.get("rounds_evaluated"))),
        ("Candidates evaluated", _format_count(process.get("candidates_evaluated"))),
    ]
    y = _draw_identity_grid(pdf, identity, left, y, right - left)

    y -= 14
    y = _draw_section_title(pdf, "Release gate results", left, y)
    _draw_gate_table(pdf, certificate, left, y, right - left)

    _draw_seal(pdf, right - 50, 104, certificate)
    signature = str(certificate.get("signature", ""))
    pdf.setFillColor(HexColor(_MUTED))
    pdf.setFont("Helvetica-Bold", 7.5)
    pdf.drawString(left, 126, "DIGITAL SEAL")
    pdf.setFillColor(HexColor(_INK))
    pdf.setFont("Courier", 8)
    pdf.drawString(left, 114, f"HMAC-SHA256  {signature[:32]}")
    pdf.drawString(left, 104, f"             {signature[32:64]}")
    pdf.setStrokeColor(HexColor(_RULE))
    pdf.setLineWidth(0.8)
    pdf.line(left, 91, left + 250, 91)
    pdf.setFillColor(HexColor(_MUTED))
    pdf.setFont("Helvetica", 7.5)
    pdf.drawString(left, 80, "Sealed by the SynthGauntlet Arbiter. Verify with Check certificate.")
    _draw_footer(pdf, width, 1)


def _draw_method_page(pdf, certificate: dict) -> None:
    from reportlab.lib.colors import HexColor

    width, height = pdf._pagesize
    left, right = 54, width - 54
    _draw_frame(pdf, width, height)
    _draw_header(pdf, certificate, width, height, "Certification method and integrity record")

    y = height - 146
    y = _draw_section_title(pdf, "How this release was produced and certified", left, y)
    for index, (title, body) in enumerate(_method_steps(certificate), start=1):
        y = _draw_step(pdf, index, title, body, left, y, right - left)
        y -= 3

    y -= 4
    y = _draw_section_title(pdf, "Integrity record", left, y)
    records = [
        ("Synthetic CSV fingerprint (SHA-256)", str(certificate.get("synthetic_sha256", ""))),
        ("Training partition fingerprint (SHA-256)", str(certificate.get("training_sha256", ""))),
        ("Certificate signature (HMAC-SHA256)", str(certificate.get("signature", ""))),
    ]
    box_height = 16 + len(records) * 25
    _panel(pdf, left, y - box_height + 8, right - left, box_height)
    row_y = y - 6
    for label, value in records:
        pdf.setFillColor(HexColor(_MUTED))
        pdf.setFont("Helvetica-Bold", 7.5)
        pdf.drawString(left + 14, row_y, label.upper())
        pdf.setFillColor(HexColor(_INK))
        pdf.setFont("Courier", 8.5)
        pdf.drawString(left + 14, row_y - 11, value)
        row_y -= 25
    y -= box_height + 8

    y = _draw_section_title(pdf, "Independent verification", left, y)
    verify_height = 90
    _panel(pdf, left, y - verify_height + 8, right - left, verify_height)
    _draw_qr(pdf, str(certificate.get("verify_url", "")), right - 82, y - verify_height + 15, 70)
    steps = (
        "1. Open SynthGauntlet and choose Check certificate.",
        "2. Upload the synthetic CSV and this certificate PDF.",
        "3. VALID confirms the file and this record are unchanged. TAMPERED means "
        "the CSV bytes or a certificate field differ from what was sealed.",
    )
    step_y = y - 8
    for line in steps:
        step_y = _draw_paragraph(pdf, line, left + 14, step_y, right - left - 120, "Helvetica", 8.8, 11.5, _INK)
        step_y -= 2
    pdf.setFillColor(HexColor(_DEEP_INDIGO))
    pdf.setFont("Helvetica", 7.5)
    pdf.drawString(left + 14, y - verify_height + 16, str(certificate.get("verify_url", ""))[:90])
    y -= verify_height + 8

    y = _draw_section_title(pdf, "Scope of this certificate", left, y)
    scope = (
        f"{certificate.get('evidence_statement') or EVIDENCE_STATEMENT} Keep the CSV and this PDF "
        "together; re-saving the CSV in another program can change its bytes and its fingerprint."
    )
    _draw_paragraph(pdf, scope, left, y - 2, right - left, "Times-Italic", 9.2, 12, _MUTED)
    _draw_footer(pdf, width, 2)


def _method_steps(certificate: dict) -> list[tuple[str, str]]:
    process = certificate.get("process") or {}
    rounds = process.get("rounds_evaluated")
    candidates = process.get("candidates_evaluated")
    max_rounds = process.get("max_rounds")
    rescue = process.get("rescue_attempts") or 0
    engine = str(certificate.get("engine_mode") or "No formal DP guarantee")
    if rounds:
        effort = f"{rounds} round(s) and {candidates} candidate table(s) were evaluated"
        if max_rounds:
            effort += f" within a limit of {max_rounds} adversarial rounds"
        effort += "."
    else:
        effort = "Each round compares several candidate tables and keeps the strongest one."
    if rescue:
        effort += f" Auto-Heal ran {rescue} rescue attempt(s) and re-tested the repaired table."
    if engine.startswith("Formal DP guarantee"):
        engine_text = f"The table was sampled from Laplace-noised marginals ({engine})."
    else:
        engine_text = "The table was synthesized with the hybrid marginal engine."
    return [
        (
            "Source intake and holdout split",
            "The source table was cleaned and split into a training partition and a held-out test "
            "partition. Synthesis used only the training partition, whose fingerprint is recorded below.",
        ),
        (
            "Multi-candidate synthesis",
            f"{engine_text} The Forge agent proposed competing synthesis plans in each round. {effort}",
        ),
        (
            "Utility audit",
            "The Auditor measured distribution fidelity with Kolmogorov-Smirnov and total-variation "
            "distances, train-on-synthetic test-on-real model utility, and aggregate query equivalence "
            "against the held-out data.",
        ),
        (
            "Adversarial privacy attacks",
            "The Challenger attacked the synthetic table with nearest-neighbour re-identification, "
            "shadow-model membership inference, attribute inference, auxiliary linkage, and "
            "near-duplicate detection against the training records.",
        ),
        (
            "Deterministic release gate",
            "The Arbiter compared every measured result with the thresholds of the selected release "
            "policy. Deterministic code makes the certification decision. Language models may advise "
            "synthesis and receive aggregate statistics only.",
        ),
        (
            "Digital sealing",
            "The exported CSV was fingerprinted with SHA-256, and this certificate record was signed "
            "with HMAC-SHA256 and embedded in this PDF for later verification.",
        ),
    ]


def _draw_frame(pdf, width: float, height: float) -> None:
    from reportlab.lib.colors import HexColor, white

    pdf.setFillColor(white)
    pdf.rect(0, 0, width, height, fill=1, stroke=0)
    pdf.setStrokeColor(HexColor(_INDIGO))
    pdf.setLineWidth(1.6)
    pdf.roundRect(18, 18, width - 36, height - 36, 14, fill=0, stroke=1)
    pdf.setStrokeColor(HexColor("#c7d2fe"))
    pdf.setLineWidth(0.6)
    pdf.roundRect(24, 24, width - 48, height - 48, 11, fill=0, stroke=1)


def _draw_header(pdf, certificate: dict, width: float, height: float, subtitle: str) -> None:
    from reportlab.lib.colors import HexColor, white

    top = height - 24
    band = 86
    pdf.saveState()
    clip = pdf.beginPath()
    clip.roundRect(24, top - band - 12, width - 48, band + 12, 11)
    pdf.clipPath(clip, stroke=0, fill=0)
    pdf.setFillColor(HexColor(_NAVY))
    pdf.rect(24, top - band, width - 48, band, fill=1, stroke=0)
    strip = pdf.beginPath()
    strip.rect(24, top - band, width - 48, 4)
    pdf.clipPath(strip, stroke=0, fill=0)
    pdf.linearGradient(24, top - band, width - 24, top - band, (HexColor(_INDIGO), HexColor(_PURPLE)), extend=True)
    pdf.restoreState()

    _draw_logo(pdf, 50, top - 62, 40)
    pdf.setFillColor(white)
    pdf.setFont("Helvetica-Bold", 19)
    pdf.drawString(102, top - 40, "SynthGauntlet")
    pdf.setFillColor(HexColor("#cbd5e1"))
    pdf.setFont("Helvetica", 9)
    pdf.drawString(102, top - 55, subtitle)

    pdf.setFillColor(HexColor("#a5b4fc"))
    pdf.setFont("Helvetica-Bold", 7.5)
    pdf.drawRightString(width - 50, top - 36, "CERTIFICATE ID")
    pdf.setFillColor(white)
    pdf.setFont("Courier-Bold", 11)
    pdf.drawRightString(width - 50, top - 50, str(certificate.get("run_id", "")))
    pdf.setFillColor(HexColor("#cbd5e1"))
    pdf.setFont("Helvetica", 7.5)
    pdf.drawRightString(width - 50, top - 63, _format_timestamp(certificate.get("issued_at")))


def _draw_logo(pdf, x: float, y: float, size: float, framed: bool = True) -> None:
    from reportlab.lib.colors import HexColor, white

    if framed:
        pdf.saveState()
        clip = pdf.beginPath()
        clip.roundRect(x, y, size, size, size * 0.24)
        pdf.clipPath(clip, stroke=0, fill=0)
        pdf.linearGradient(x, y + size, x + size, y, (HexColor(_INDIGO), HexColor(_PURPLE)), extend=True)
        pdf.restoreState()
        inner, pad = size * 0.5, size * 0.25
    else:
        inner, pad = size, 0

    def point(u: float, v: float) -> tuple[float, float]:
        return x + pad + (u - 2) / 20 * inner, y + size - pad - (v - 2) / 20 * inner

    pdf.saveState()
    pdf.setStrokeColor(white)
    pdf.setLineWidth(max(inner * 0.11, 0.8))
    pdf.setLineCap(1)
    pdf.setLineJoin(1)
    for points, closed in (
        (((12, 2), (2, 7), (12, 12), (22, 7)), True),
        (((2, 17), (12, 22), (22, 17)), False),
        (((2, 12), (12, 17), (22, 12)), False),
    ):
        path = pdf.beginPath()
        path.moveTo(*point(*points[0]))
        for item in points[1:]:
            path.lineTo(*point(*item))
        if closed:
            path.close()
        pdf.drawPath(path, stroke=1, fill=0)
    pdf.restoreState()


def _draw_status_pill(pdf, center_x: float, y: float, label: str) -> None:
    from reportlab.lib.colors import HexColor
    from reportlab.pdfbase.pdfmetrics import stringWidth

    text_width = stringWidth(label, "Helvetica-Bold", 9.5)
    pill_width = text_width + 44
    pdf.setFillColor(HexColor("#d1fae5"))
    pdf.setStrokeColor(HexColor("#10b981"))
    pdf.setLineWidth(1)
    pdf.roundRect(center_x - pill_width / 2, y - 7, pill_width, 22, 11, fill=1, stroke=1)
    pdf.setFillColor(HexColor("#10b981"))
    pdf.circle(center_x - pill_width / 2 + 15, y + 4, 3.2, fill=1, stroke=0)
    pdf.setFillColor(HexColor("#065f46"))
    pdf.setFont("Helvetica-Bold", 9.5)
    pdf.drawString(center_x - pill_width / 2 + 25, y + 0.6, label)


def _draw_section_title(pdf, title: str, x: float, y: float) -> float:
    from reportlab.lib.colors import HexColor

    pdf.setFillColor(HexColor(_INDIGO))
    pdf.roundRect(x, y - 1, 3, 11, 1.5, fill=1, stroke=0)
    pdf.setFillColor(HexColor(_NAVY))
    pdf.setFont("Helvetica-Bold", 10.5)
    pdf.drawString(x + 10, y, title.upper())
    return y - 14


def _panel(pdf, x: float, y: float, width: float, height: float) -> None:
    from reportlab.lib.colors import HexColor

    pdf.setFillColor(HexColor(_PANEL))
    pdf.setStrokeColor(HexColor(_RULE))
    pdf.setLineWidth(0.8)
    pdf.roundRect(x, y, width, height, 8, fill=1, stroke=1)


def _draw_identity_grid(pdf, items: list[tuple[str, str]], x: float, y: float, width: float) -> float:
    from reportlab.lib.colors import HexColor

    rows = (len(items) + 1) // 2
    row_height = 27
    height = rows * row_height + 10
    _panel(pdf, x, y - height + 6, width, height)
    column = width / 2
    for index, (label, value) in enumerate(items):
        col, row = index % 2, index // 2
        cx = x + 14 + col * column
        cy = y - 8 - row * row_height
        pdf.setFillColor(HexColor(_MUTED))
        pdf.setFont("Helvetica-Bold", 7)
        pdf.drawString(cx, cy, label.upper())
        pdf.setFillColor(HexColor(_INK))
        font = "Courier-Bold" if label == "Certificate ID" else "Helvetica-Bold"
        pdf.setFont(font, 9.5)
        pdf.drawString(cx, cy - 12, _fit(value, font, 9.5, column - 24))
    pdf.setStrokeColor(HexColor(_RULE))
    pdf.line(x + column, y - height + 14, x + column, y - 2)
    return y - height


def _draw_gate_table(pdf, certificate: dict, x: float, y: float, width: float) -> float:
    from reportlab.lib.colors import HexColor, white

    scorecard = certificate.get("scorecard") or {}
    thresholds = certificate.get("thresholds") or {}
    columns = (0.0, 0.40, 0.56, 0.72, 0.86)
    header_height, row_height = 20, 17.5
    pdf.setFillColor(HexColor(_NAVY))
    pdf.roundRect(x, y - header_height + 6, width, header_height, 6, fill=1, stroke=0)
    pdf.setFillColor(white)
    pdf.setFont("Helvetica-Bold", 7.5)
    for label, offset in zip(("CHECK", "FAMILY", "RESULT", "REQUIRED", "OUTCOME"), columns):
        pdf.drawString(x + 12 + offset * width, y - 7, label)
    row_y = y - header_height + 6
    for index, (label, key, threshold_key, direction, family) in enumerate(_GATE_ROWS):
        top = row_y - index * row_height
        if index % 2 == 0:
            pdf.setFillColor(HexColor(_PANEL))
            pdf.rect(x, top - row_height, width, row_height, fill=1, stroke=0)
        value = scorecard.get(key)
        limit = thresholds.get(threshold_key)
        baseline = top - 11.8
        pdf.setFillColor(HexColor(_INK))
        pdf.setFont("Helvetica-Bold", 8.5)
        pdf.drawString(x + 12, baseline, label)
        pdf.setFillColor(HexColor("#047857" if family == "Utility" else "#be123c"))
        pdf.setFont("Helvetica", 8)
        pdf.drawString(x + 12 + columns[1] * width, baseline, family)
        pdf.setFillColor(HexColor(_INK))
        pdf.setFont("Helvetica-Bold", 8.5)
        pdf.drawString(x + 12 + columns[2] * width, baseline, _format_percent(value))
        pdf.setFont("Helvetica", 8)
        required = "n/a" if limit is None else ("at least " if direction == "min" else "at most ") + _format_percent(limit)
        pdf.drawString(x + 12 + columns[3] * width, baseline, required)
        _draw_outcome(pdf, x + 12 + columns[4] * width, top - row_height / 2, _outcome(value, limit, direction))
    bottom = row_y - len(_GATE_ROWS) * row_height
    pdf.setStrokeColor(HexColor(_RULE))
    pdf.setLineWidth(0.8)
    pdf.line(x, bottom, x + width, bottom)
    return bottom


def _outcome(value, limit, direction: str) -> str:
    if value is None or limit is None:
        return "RECORDED"
    passed = float(value) >= float(limit) if direction == "min" else float(value) <= float(limit)
    return "PASS" if passed else "FAIL"


def _draw_outcome(pdf, x: float, middle: float, outcome: str) -> None:
    from reportlab.lib.colors import HexColor

    fill, text = {
        "PASS": ("#d1fae5", "#065f46"),
        "FAIL": ("#ffe4e6", "#9f1239"),
    }.get(outcome, ("#e2e8f0", "#334155"))
    pdf.setFillColor(HexColor(fill))
    pdf.roundRect(x, middle - 6.5, 54, 13, 6.5, fill=1, stroke=0)
    pdf.setFillColor(HexColor(text))
    pdf.setFont("Helvetica-Bold", 7)
    pdf.drawCentredString(x + 27, middle - 2.4, outcome)


def _draw_step(pdf, index: int, title: str, body: str, x: float, y: float, width: float) -> float:
    from reportlab.lib.colors import HexColor, white

    radius = 9
    pdf.saveState()
    clip = pdf.beginPath()
    clip.circle(x + radius, y - radius + 4, radius)
    pdf.clipPath(clip, stroke=0, fill=0)
    pdf.linearGradient(x, y + 4, x + 2 * radius, y - 2 * radius + 4, (HexColor(_INDIGO), HexColor(_PURPLE)), extend=True)
    pdf.restoreState()
    pdf.setFillColor(white)
    pdf.setFont("Helvetica-Bold", 9)
    pdf.drawCentredString(x + radius, y - radius + 1, str(index))
    pdf.setFillColor(HexColor(_NAVY))
    pdf.setFont("Helvetica-Bold", 10)
    pdf.drawString(x + 28, y - 4, title)
    return _draw_paragraph(pdf, body, x + 28, y - 17, width - 28, "Helvetica", 8.6, 11.3, _INK)


def _draw_seal(pdf, cx: float, cy: float, certificate: dict) -> None:
    from reportlab.lib.colors import HexColor, white

    radius = 40
    pdf.saveState()
    clip = pdf.beginPath()
    clip.circle(cx, cy, radius)
    pdf.clipPath(clip, stroke=0, fill=0)
    pdf.linearGradient(cx - radius, cy + radius, cx + radius, cy - radius, (HexColor(_INDIGO), HexColor(_PURPLE)), extend=True)
    pdf.restoreState()
    pdf.setStrokeColor(white)
    pdf.setLineWidth(1.2)
    pdf.circle(cx, cy, radius - 5, fill=0, stroke=1)
    pdf.setLineWidth(0.5)
    pdf.circle(cx, cy, radius - 8, fill=0, stroke=1)
    _draw_logo(pdf, cx - 9, cy + 6, 18, framed=False)
    pdf.setFillColor(white)
    pdf.setFont("Helvetica-Bold", 8)
    pdf.drawCentredString(cx, cy - 6, "CERTIFIED")
    pdf.setFont("Helvetica", 5.8)
    pdf.drawCentredString(cx, cy - 15, "SYNTHGAUNTLET")
    issued = str(certificate.get("issued_at") or "")[:10]
    if issued:
        pdf.drawCentredString(cx, cy - 23, issued)


def _draw_footer(pdf, width: float, page: int) -> None:
    from reportlab.lib.colors import HexColor

    pdf.setStrokeColor(HexColor(_RULE))
    pdf.setLineWidth(0.6)
    pdf.line(54, 56, width - 54, 56)
    pdf.setFillColor(HexColor(_MUTED))
    pdf.setFont("Helvetica", 7.5)
    pdf.drawString(54, 44, "SynthGauntlet release certificate. Technical evidence of synthesis and attack testing.")
    pdf.drawRightString(width - 54, 44, f"Page {page} of 2")


def _draw_paragraph(
    pdf, text: str, x: float, y: float, width: float, font: str, size: float,
    leading: float, color: str, centered: bool = False,
) -> float:
    from reportlab.lib.colors import HexColor

    pdf.setFillColor(HexColor(color))
    pdf.setFont(font, size)
    for line in _wrap_to_width(text, font, size, width):
        if centered:
            pdf.drawCentredString(x + width / 2, y, line)
        else:
            pdf.drawString(x, y, line)
        y -= leading
    return y


def _wrap_to_width(text: str, font: str, size: float, width: float) -> list[str]:
    from reportlab.pdfbase.pdfmetrics import stringWidth

    lines, current = [], ""
    for word in text.split():
        trial = f"{current} {word}".strip()
        if current and stringWidth(trial, font, size) > width:
            lines.append(current)
            current = word
        else:
            current = trial
    if current:
        lines.append(current)
    return lines


def _fit(text: str, font: str, size: float, width: float) -> str:
    from reportlab.pdfbase.pdfmetrics import stringWidth

    if stringWidth(text, font, size) <= width:
        return text
    while text and stringWidth(text + "...", font, size) > width:
        text = text[:-1]
    return text + "..."


def _status_label(status: str) -> str:
    return {
        "CERTIFIED": "CERTIFIED SAFE AND USEFUL",
        "CERTIFIED_AFTER_RESCUE": "CERTIFIED AFTER AUTO-HEAL RESCUE",
    }.get(status, status.replace("_", " "))


def _format_percent(value) -> str:
    if value is None:
        return "n/a"
    try:
        return f"{float(value) * 100:.1f}%"
    except (TypeError, ValueError):
        return str(value)


def _format_count(value) -> str:
    if value is None:
        return "n/a"
    try:
        return f"{int(value):,}"
    except (TypeError, ValueError):
        return str(value)


def _format_timestamp(value) -> str:
    if not value:
        return "n/a"
    try:
        return datetime.fromisoformat(str(value)).strftime("%d %b %Y, %H:%M UTC")
    except ValueError:
        return str(value)


def _draw_qr(pdf, url: str, x: float, y: float, size: float = 72) -> None:
    from reportlab.lib.colors import HexColor

    try:
        import qrcode
        from reportlab.lib.utils import ImageReader

        image = qrcode.make(url, border=1)
        stream = io.BytesIO()
        image.save(stream, format="PNG")
        stream.seek(0)
        pdf.drawImage(ImageReader(stream), x, y, size, size, mask="auto")
    except Exception:
        pdf.setFillColor(HexColor(_INK))
        pdf.setFont("Helvetica", 7)
        pdf.drawString(x, y + size / 2, "Verify:")
        pdf.drawString(x, y + size / 2 - 10, url[-42:])
