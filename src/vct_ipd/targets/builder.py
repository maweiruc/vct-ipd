"""Explicit one-source adoption and supported unit normalization, not synthesis."""

from dataclasses import dataclass
import json

from vct_ipd.core import CalibrationTarget, CalibrationTargetSet, SummarySpec, Tolerance
from vct_ipd.core.base import Contract, Identifier
from vct_ipd.evaluation.metrics import validate_metric


class TargetRule(Contract):
    evidence_id: Identifier
    target_id: Identifier
    tolerance: Tolerance  # Explicit canonical-unit acceptance tolerance, never inferred from a CI.


def unit_factor(source, destination):
    if source == destination:
        return 1.0
    factors = {("month", "year"): 1 / 12, ("year", "month"): 12.0}
    if (source, destination) not in factors:
        raise ValueError(f"unsupported or unspecified unit conversion: {source} -> {destination}")
    return factors[source, destination]


@dataclass
class BuildResult:
    targets: CalibrationTargetSet
    unadopted: dict[str, str]


class TargetBuilder:
    def build(self, task, schema, evidence, rules):
        evidence.validate_references(schema)
        sources = {e.id: e for e in evidence.items}
        adopted, seen, items = set(), set(), []
        for rule in rules:
            source = sources.get(rule.evidence_id)
            if source is None:
                raise ValueError("adoption rule references unknown evidence")
            if source.population != task.population or source.treatment != task.treatment:
                raise ValueError("automatic adoption requires identical declared population and treatment")
            if source.endpoint is not None and (source.endpoint.name != task.endpoint.name
                                                or source.endpoint.time_origin != task.endpoint.time_origin):
                raise ValueError("automatic endpoint/origin transport is unsupported, regardless of adoption notes")
            if source.estimate_status != "available":
                raise ValueError(f"requested evidence {source.id} is unavailable: {source.estimate_reason}")
            summary = source.summary.model_dump(mode="json")
            metric = source.summary.metric
            destination = source.summary.unit
            value_factor, time_factor = 1.0, 1.0
            if metric in ("mean", "sd", "quantile"):
                variable = next(v for v in schema.variables if v.name == source.summary.variables[0])
                destination = variable.unit
                value_factor = unit_factor(source.summary.unit, destination)
                if value_factor != 1 and any(c.variable == variable.name for c in source.summary.conditions):
                    raise ValueError("conditional threshold unit normalization is not supported; supply canonical conditions")
            elif metric in ("survival_probability", "median_survival"):
                destination = task.endpoint.unit
                time_factor = unit_factor(source.summary.unit, destination)
                summary["time_origin"] = task.endpoint.time_origin
                if metric == "median_survival":
                    value_factor = time_factor
                else:
                    summary["at_time"] *= time_factor
                if time_factor != 1 and any(c.variable in source.summary.variables for c in source.summary.conditions):
                    raise ValueError("survival condition time conversion requires an explicit future mapping")
            summary["unit"] = destination
            summary = SummarySpec.model_validate(summary)
            validate_metric(summary, schema)
            if metric == "validity":
                raise ValueError("validity is a design constraint, not an adopted statistical estimate")
            key = json.dumps(summary.model_dump(mode="json"), sort_keys=True)
            if key in seen:
                raise ValueError("multiple sources for the same target require explicit selection; no implicit pooling/conflict range")
            seen.add(key)
            uncertainty = source.uncertainty.model_dump(mode="json") if source.uncertainty else None
            if uncertainty:
                if uncertainty["interval"]:
                    uncertainty["interval"] = {k: v * value_factor if v is not None else None
                                               for k, v in uncertainty["interval"].items()}
                if uncertainty["standard_error"] is not None:
                    uncertainty["standard_error"] *= abs(value_factor)
            rule_record = {"source_ids": [source.id], "source_version": source.version,
                           "from_unit": source.summary.unit, "to_unit": destination,
                           "value_factor": value_factor, "time_factor": time_factor,
                           "normalized_uncertainty": uncertainty,
                           "uncertainty_unit": "1" if metric in ("proportion", "correlation", "survival_probability") else destination,
                           "operation": "multiply value/uncertainty by value_factor; evaluation time by time_factor"}
            items.append(CalibrationTarget(
                id=rule.target_id, summary=summary, desired={"center": source.estimate * value_factor},
                evidence_ids=[source.id], tolerance=rule.tolerance,
                rationale=f"Explicit one-source adoption of {source.id}; normalization {rule_record}",
                extensions={"normalization": rule_record}))
            adopted.add(source.id)
        items.insert(0, CalibrationTarget(id="row-validity", summary=SummarySpec(metric="validity", variables=[v.name for v in schema.variables]),
                                         desired={"center": True}, hard_constraint=True, priority="high",
                                         rationale="All rows must satisfy the declared schema and observation process"))
        result = CalibrationTargetSet(id="calibration-targets", task=task.ref, schema_ref=schema.ref,
                                      evidence_set=evidence.ref, items=items)
        result.validate_references(evidence, schema)
        result.validate_task(task, schema)
        return BuildResult(result, {e.id: "Not selected by an explicit adoption rule" for e in evidence.items if e.id not in adopted})
