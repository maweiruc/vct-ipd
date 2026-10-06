"""Evaluate actual rows per seed, then aggregate estimates without pooling patients."""

import hashlib
import statistics

from vct_ipd.core import EvaluationReport, TargetResult
from vct_ipd.evaluation.metrics import Measurement, measure, unavailable, validate_metric, validity_errors
from vct_ipd.generators.statistical import csv_text


def discrepancy(value, target):
    desired = target.desired
    if desired.range:
        distance = max((desired.range.lower - value) if desired.range.lower is not None else 0,
                       (value - desired.range.upper) if desired.range.upper is not None else 0, 0)
    else:
        distance = abs(value - desired.center)
    tolerance = 0.0
    if target.tolerance:
        if target.tolerance.absolute is not None:
            tolerance = target.tolerance.absolute
        else:
            if type(desired.center) not in (int, float):
                raise ValueError("relative tolerance requires an explicit numeric center")
            tolerance = target.tolerance.relative * abs(desired.center)
    return distance, tolerance


def aggregate(measurements):
    if all(m.status == "available" for m in measurements):
        return Measurement(statistics.fmean(m.value for m in measurements))
    failed = [m for m in measurements if m.status != "available"]
    if len(failed) == len(measurements) and all(m.status == "not_reached" for m in failed):
        bound = min(m.lower_bound for m in failed) if all(m.lower_bound is not None for m in failed) else None
        return unavailable("Median not reached in every replica; no pooled estimate", "not_reached", bound)
    return unavailable("At least one replica has an unavailable result; no selective averaging")


class Evaluator:
    def evaluate(self, samples, task, schema, targets, state, *, phase):
        if not samples:
            raise ValueError("evaluation needs at least one dataset")
        state.validate_references(task, schema, targets)
        counts = []
        for sample in samples:
            sample.metadata.validate_references(state, schema)
            if sample.metadata.sample_size != len(sample.rows):
                raise ValueError("metadata sample size differs from actual rows")
            content = csv_text(sample.rows, [v.name for v in schema.variables]).encode()
            if hashlib.sha256(content).hexdigest() != sample.metadata.data.content_hash:
                raise ValueError("dataset content differs from its declared hash")
            counts.append(validity_errors(sample.rows, schema, sample.latent))
        results, per_seed, loss = [], {}, 0.0
        for target in targets.items:
            validate_metric(target.summary, schema)
            if target.summary.metric == "validity":
                valid = not any(counts)
                result = TargetResult(target=target.ref, status="PASS" if valid else "FAIL",
                                      structured_result={"check": "row-and-observation-validity", "passed": valid,
                                                         "details": {"invalid_rows_by_seed": counts}},
                                      diagnostic="Any invalid row in any replica fails this hard check")
                if not valid:
                    loss += 1e6
            else:
                values = [unavailable("Dataset contains invalid rows") if invalid else measure(sample.rows, target.summary, schema)
                          for sample, invalid in zip(samples, counts)]
                per_seed[target.id] = [{"seed": sample.metadata.seed, **value.fields()} for sample, value in zip(samples, values)]
                value = aggregate(values)
                status, distance = "FAIL", None
                if value.status == "available":
                    distance, allowed = discrepancy(value.value, target)
                    status = "PASS" if distance <= allowed else "FAIL"
                    loss += {"low": 0.5, "medium": 1, "high": 2}[target.priority] * min(distance / max(allowed, 1e-6), 1e6)
                else:
                    loss += 1e6
                if target.evidence_limited:
                    status = "EVIDENCE_LIMITED"
                result = TargetResult(target=target.ref, status=status, discrepancy=distance,
                                      diagnostic="Arithmetic mean of replica estimates; each replica retained; "
                                                 + (target.evidence_limitation or "tolerance tested on the aggregate"),
                                      **value.fields())
            result.validate_target(target)
            results.append(result)
        hard_failure = any(t.hard_constraint and r.status != "PASS" for t, r in zip(targets.items, results))
        statuses = {r.status for r in results}
        status = ("FAIL" if hard_failure else "PASS" if statuses == {"PASS"}
                  else "EVIDENCE_LIMITED" if statuses <= {"PASS", "EVIDENCE_LIMITED"}
                  else "PARTIAL" if "PASS" in statuses else "FAIL")
        report = EvaluationReport(
            id=f"{state.id}-{phase}-report", generator_state=state.ref, target_set=targets.ref,
            schema_ref=schema.ref, datasets=[s.metadata.ref for s in samples], run_id=state.run_id,
            iteration=state.iteration, status=status, results=results,
            summary=f"{phase}: {sum(r.status == 'PASS' for r in results)}/{len(results)} targets pass",
            evaluator_version="m2.1-evaluator-v1", overall_metrics={"loss": loss, "phase": phase,
                "seeds": [s.metadata.seed for s in samples], "per_seed": per_seed,
                "aggregation": "mean of available per-replica estimates; all replicas required",
                "hard_validity_pass": not hard_failure})
        report.validate_references(state, targets, [s.metadata for s in samples])
        return report
