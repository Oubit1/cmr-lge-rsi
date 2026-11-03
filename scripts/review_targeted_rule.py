#!/usr/bin/env python3
"""Run a governed multi-agent review of one de-identified QC rule candidate."""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

import yaml

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from agents.shadow_analysis import (  # noqa: E402
    MAINTENANCE_PROMPTS,
    REVIEW_PROMPT,
    SPECIALIST_PROMPTS,
    SUPERVISOR_PROMPT,
)
from contracts.compat import model_dump  # noqa: E402
from contracts.models import (  # noqa: E402
    MaintenanceProposal,
    ReviewDecision,
    SpecialistAssessment,
    SupervisorDecision,
)
from providers.registry import ProviderRegistry  # noqa: E402


def expand_env(value: Any) -> Any:
    if isinstance(value, str):
        match = re.fullmatch(
            r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?:\|([A-Za-z_][A-Za-z0-9_]*))?\}", value)
        if match:
            primary, fallback = match.groups()
            resolved = os.getenv(primary) or (os.getenv(fallback) if fallback else None)
            if not resolved:
                raise RuntimeError(f"missing model environment variable: {primary}"
                                   + (f" (and fallback {fallback})" if fallback else ""))
            return resolved
    if isinstance(value, dict):
        return {key: expand_env(item) for key, item in value.items()}
    if isinstance(value, list):
        return [expand_env(item) for item in value]
    return value


async def review(args: argparse.Namespace) -> dict[str, Any]:
    evidence = json.loads(args.evidence.read_text(encoding="utf-8"))
    agents = expand_env(yaml.safe_load(args.agents.read_text(encoding="utf-8")))["agents"]
    registry = ProviderRegistry(args.providers)

    specialist_cfg = agents["cmr_specialist"]
    specialist = await registry.get(specialist_cfg["provider"]).generate_structured(
        system_prompt=SPECIALIST_PROMPTS["cmr_specialist"],
        input_data={"role": "cmr_specialist", "targeted_rule_evidence": evidence,
                    "mode": "bounded_rule_review_no_label_mutation"},
        output_schema=SpecialistAssessment,
        model=specialist_cfg["model"],
    )

    async def maintainer(role: str) -> MaintenanceProposal:
        cfg = agents[role]
        return await registry.get(cfg["provider"]).generate_structured(
            system_prompt=MAINTENANCE_PROMPTS[role],
            input_data={
                "role": role,
                "targeted_rule_evidence": evidence,
                "cmr_specialist": model_dump(specialist),
                "constraints": {
                    "label_mutation": False,
                    "classification_only": True,
                    "internal_failures_must_remain_middle": True,
                },
            },
            output_schema=MaintenanceProposal,
            model=cfg["model"],
        )

    maintenance = await asyncio.gather(
        maintainer("qc_code_optimizer"), maintainer("rule_curator"),
        maintainer("skill_maintainer"),
    )
    supervisor_cfg = agents["supervisor"]
    supervisor_input = {
        "targeted_rule_evidence": evidence,
        "specialist_assessments": [model_dump(specialist)],
        "maintenance_proposals": [model_dump(item) for item in maintenance],
        "evaluation_policy": {
            "primary": ["confirmed_case_recall", "internal_middle_retention",
                        "false_boundary_reclassification_rate"],
            "safety": ["zero_label_mutations", "reversible_manifest_and_png_move"],
        },
    }
    supervisor = await registry.get(supervisor_cfg["provider"]).generate_structured(
        system_prompt=SUPERVISOR_PROMPT,
        input_data=supervisor_input,
        output_schema=SupervisorDecision,
        model=supervisor_cfg["model"],
    )
    reviewer_cfg = agents["independent_reviewer"]
    reviewer = await registry.get(reviewer_cfg["provider"]).generate_structured(
        system_prompt=REVIEW_PROMPT,
        input_data={**supervisor_input, "supervisor_decision": model_dump(supervisor)},
        output_schema=ReviewDecision,
        model=reviewer_cfg["model"],
    )
    return {
        "evidence": evidence,
        "cmr_specialist": model_dump(specialist),
        "maintenance": [model_dump(item) for item in maintenance],
        "supervisor": model_dump(supervisor),
        "independent_reviewer": model_dump(reviewer),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--providers", type=Path, default=PROJECT / "configs/providers.yaml")
    parser.add_argument("--agents", type=Path, default=PROJECT / "configs/agents.production.yaml")
    args = parser.parse_args()
    result = asyncio.run(review(args))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "output": str(args.output),
        "supervisor_decision": result["supervisor"]["decision"],
        "reviewer_decision": result["independent_reviewer"]["decision"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
