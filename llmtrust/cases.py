from __future__ import annotations

from pathlib import Path

import yaml

from llmtrust.types import Case, DecisionPoint


def load_case(path: Path) -> Case:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    points = []
    for dp in raw["decision_points"]:
        sources = dp.get("sources") or {}
        points.append(DecisionPoint(
            id=dp["id"],
            title=dp["title"],
            decision_date=dp["decision_date"],
            cutoff=dp["cutoff"],
            options=list(dp["options_to_present"]),
            actual_decision=" ".join(str(dp.get("actual_decision", "")).split()),
            edgar_forms=list(sources.get("edgar_forms", [])),
            email_custodians=list(sources.get("email_custodians", [])),
            role=" ".join(str(dp.get("role", "")).split()),
            scenario_brief=" ".join(str(dp.get("scenario_brief", "")).split()),
            stake_statement=" ".join(str(dp.get("stake_statement", "")).split()),
        ))
    return Case(
        case_id=raw["case_id"],
        company=raw["company"],
        decision_points=points,
        evaluation_only=list(raw.get("evaluation_only", [])),
        leakage_terms=list(raw.get("leakage_terms", [])),
    )
