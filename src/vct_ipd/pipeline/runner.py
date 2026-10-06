"""Explicit validation boundaries and a bounded deterministic calibration run."""

from dataclasses import dataclass

from vct_ipd.core import Action, CalibrationTargetSet, DataSchema, EvidenceSet, GeneratorState, TaskSpec
from vct_ipd.calibration.search import CoordinateSearch, better, feasible, stop_reason
from vct_ipd.evaluation.metrics import validate_metric
from vct_ipd.evaluation.runner import Evaluator, discrepancy
from vct_ipd.generators.statistical import StatisticalGenerator, snapshot


def ready_to_run(task, schema, evidence, targets, state, generator):
    # Reconstruct to reject objects made invalid through mutable nested containers.
    objects = [cls.model_validate(obj.model_dump(mode="json")) for cls, obj in
               ((TaskSpec, task), (DataSchema, schema), (EvidenceSet, evidence),
                (CalibrationTargetSet, targets), (GeneratorState, state))]
    task, schema, evidence, targets, state = objects
    evidence.validate_references(schema)
    targets.validate_references(evidence, schema)
    targets.validate_task(task, schema)
    state.validate_references(task, schema, targets)
    if not targets.items or not any(t.summary.metric != "validity" for t in targets.items):
        raise ValueError("ready-to-run requires nonempty statistical targets, not just a draft/validity check")
    checks = [t for t in targets.items if t.summary.metric == "validity"]
    if len(checks) != 1 or not checks[0].hard_constraint or checks[0].desired.center is not True or checks[0].desired.range is not None or checks[0].summary.conditions:
        raise ValueError("exactly one unconditional hard row-validity target is required")
    if set(checks[0].summary.variables) != {v.name for v in schema.variables}:
        raise ValueError("row validity must cover the complete schema")
    for target in targets.items:
        validate_metric(target.summary, schema)
        if target.summary.metric != "validity":
            if target.desired.center is not None and type(target.desired.center) not in (float, int):
                raise ValueError("only scalar numeric statistical targets are supported")
            discrepancy(0.0, target)  # Validate relative-range/zero-target tolerance semantics.
    if not task.generation.comparison_seeds or not task.generation.validation_seeds:
        raise ValueError("fixed comparison and fresh validation seeds are required")
    if task.calibration.no_improvement_patience is None:
        raise ValueError("an explicit no-improvement patience is required")
    if state.iteration != 0:
        raise ValueError("this runner starts at an initial state; resume is not implemented")
    generator.validate_configuration(task, schema, state)
    return objects


@dataclass
class RunResult:
    task: TaskSpec
    schema: DataSchema
    evidence: EvidenceSet
    targets: CalibrationTargetSet
    states: list
    reports: list
    actions: list
    samples: list
    best_index: int
    validation_report: object
    validation_samples: list
    stop_reason: str
    best_feasible: bool
    optimizer: dict

    @property
    def success(self):
        return self.best_feasible and self.stop_reason == "calibrated" and self.validation_report.status == "PASS"


def evaluate_state(task, schema, evidence, targets, state, generator, evaluator, *, phase="comparison"):
    """Shared numerical boundary; callers decide when final validation is allowed."""
    evidence.validate_references(schema)
    targets.validate_references(evidence, schema)
    targets.validate_task(task, schema)
    state.validate_references(task, schema, targets)
    generator.validate_configuration(task, schema, state)
    seeds = task.generation.comparison_seeds if phase == "comparison" else task.generation.validation_seeds
    replicas = [generator.sample(task, schema, state, seed, phase=phase) for seed in seeds]
    for sample in replicas:
        sample.metadata.validate_references(state, schema)
    report = evaluator.evaluate(replicas, task, schema, targets, state, phase=phase)
    report.validate_references(state, targets, [s.metadata for s in replicas])
    return snapshot(report), replicas


def finalize(task, schema, evidence, targets, states, reports, actions, samples,
             best_index, reason, optimizer, generator, evaluator):
    """One fresh evaluation of an already locked state/objective; no decisions."""
    validation, replicas = evaluate_state(task, schema, evidence, targets, states[best_index],
                                          generator, evaluator, phase="validation")
    return RunResult(task, schema, evidence, targets, states, reports, actions, samples,
                     best_index, validation, replicas, reason, feasible(reports[best_index], targets), optimizer)


def run(task, schema, evidence, targets, state, *, generator=None, evaluator=None, search=None):
    generator, evaluator, search = generator or StatisticalGenerator(), evaluator or Evaluator(), search or CoordinateSearch()
    optimizer = {"implementation": f"{type(search).__module__}.{type(search).__qualname__}",
                 "replay": "Every evaluated parameter proposal is saved in state/action history"}
    if isinstance(search, CoordinateSearch):
        optimizer.update(dimensions=[list(d) for d in search.dimensions],
                         halve_steps_every_proposals=4 * len(search.dimensions), version="coordinate-search-v2")
    task, schema, evidence, targets, state = ready_to_run(task, schema, evidence, targets, state, generator)
    states, reports, actions, samples = [], [], [], []
    best_index, stale = 0, 0
    while True:
        report, replicas = evaluate_state(task, schema, evidence, targets, state, generator, evaluator)
        states.append(snapshot(state))
        reports.append(snapshot(report))
        samples.extend(replicas)
        if len(reports) == 1 or better(report, reports[best_index], targets):
            best_index, stale = len(reports) - 1, 0
        else:
            stale += 1
        best_state, best_report = states[best_index], reports[best_index]
        reason = stop_reason(best_report, evaluated=len(reports), budget=task.calibration.max_iterations,
                             stale=stale, patience=task.calibration.no_improvement_patience)
        if reason:
            if not feasible(best_report, targets):
                reason = "structural_failure"
            stop = Action(id=f"{state.run_id}-stop", type="STOP", payload={"reason": reason},
                          reason=f"Comparison-seed calibration stopped: {reason}",
                          evaluation_report=best_report.ref, generator_state=best_state.ref,
                          run_id=best_state.run_id, iteration=best_state.iteration)
            stop.validate_references(best_report, best_state)
            actions.append(snapshot(stop))
            break
        parameters = search.propose(snapshot(best_state), len(reports) - 1)
        action = Action(id=f"{state.run_id}-tune-{len(reports)}", type="TUNE_PARAMETERS",
                        payload={"component": "statistical", "parameters": parameters},
                        reason=f"Bounded coordinate proposal from best snapshot {best_state.id}",
                        evaluation_report=report.ref, generator_state=state.ref,
                        run_id=state.run_id, iteration=state.iteration,
                        extensions={"proposal_base": best_state.ref.model_dump(mode="json")})
        action.validate_references(report, state)
        actions.append(snapshot(action))
        state = generator.update(state, action, report)
    return finalize(task, schema, evidence, targets, states, reports, actions, samples, best_index,
                    reason, optimizer, generator, evaluator)
