from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DecisionPoint:
    id: str; title: str; decision_date: str; cutoff: str
    options: list[str]; actual_decision: str
    edgar_forms: list[str]; email_custodians: list[str]
    role: str; scenario_brief: str; stake_statement: str


@dataclass(frozen=True)
class Case:
    case_id: str; company: str
    decision_points: list[DecisionPoint]
    evaluation_only: list[dict]       # {"description": str, "url": str | None}
    leakage_terms: list[str]


@dataclass(frozen=True)
class SourceDoc:
    date: str                         # ISO date
    kind: str                         # "filing" | "email" | "document"
    title: str
    text: str


@dataclass(frozen=True)
class Prompt:
    system: str; user: str
    option_map: dict[str, str]        # letter -> original option text


@dataclass(frozen=True)
class TrialSpec:
    case_id: str; decision_id: str; level: int; stake: bool
    scenario: str                     # "real" | "twin"
    arm: str; repeat: int

    @property
    def trial_id(self) -> str:
        # e.g. "enron.EN-3b.L3.stake.twin.claude-haiku.r2"
        stake = "stake" if self.stake else "nostake"
        return (f"{self.case_id}.{self.decision_id}.L{self.level}.{stake}"
                f".{self.scenario}.{self.arm}.r{self.repeat}")


@dataclass(frozen=True)
class Completion:
    text: str; model_id: str; provider: str
    input_tokens: int; output_tokens: int; finished_at: str
    # True when the provider reported no usage and the charge is the pre-call estimate
    cost_estimated: bool = False
