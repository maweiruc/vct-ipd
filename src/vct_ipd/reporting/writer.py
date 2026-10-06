"""Write an auditable offline run package; never overwrite a prior run."""

import hashlib
import json
from pathlib import Path
import platform

import pydantic

from vct_ipd.calibration.search import feasible
from vct_ipd.generators.statistical import csv_text


def write_run(result, destination):
    root = Path(destination)
    root.mkdir(parents=True, exist_ok=False)
    def save(relative, value):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        if hasattr(value, "model_dump"):
            value = value.model_dump(mode="json")
        path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    for name in ("task", "schema", "evidence", "targets"):
        save(f"{name}.json", getattr(result, name))
    save("generator/states.json", [s.model_dump(mode="json") for s in result.states])
    best_state = result.states[result.best_index]
    save("generator/best.json", best_state)
    save("history/actions.json", [a.model_dump(mode="json") for a in result.actions])
    for report in result.reports:
        save(f"evaluation/iteration_{report.iteration:03d}.json", report)
    save("evaluation/initial.json", result.reports[0])
    save("evaluation/best.json", result.reports[result.best_index])
    save("evaluation/final_validation.json", result.validation_report)
    all_samples = result.samples + result.validation_samples
    for sample in all_samples:
        relative = Path(sample.metadata.data.uri)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("run dataset URI must stay within the run package")
        content = csv_text(sample.rows, [v.name for v in result.schema.variables])
        if hashlib.sha256(content.encode()).hexdigest() != sample.metadata.data.content_hash:
            raise ValueError("dataset hash mismatch at write boundary")
        (root / relative).parent.mkdir(parents=True, exist_ok=True)
        (root / relative).write_text(content)
        save(f"latent/{relative.stem}.json", sample.latent)
    save("datasets/manifest.json", [s.metadata.model_dump(mode="json") for s in all_samples])
    # Preserve every diagnostic dataset above, but export a canonical cohort only
    # when comparison selection and every fresh replica satisfy the hard checks.
    export_blocked_reason = None
    if not result.best_feasible or not feasible(result.reports[result.best_index], result.targets):
        export_blocked_reason = "No hard-feasible comparison candidate exists."
    elif not feasible(result.validation_report, result.targets):
        export_blocked_reason = "Final validation failed required hard checks; diagnostic datasets retained."
    exported_dataset = None
    if export_blocked_reason is None:
        # One fresh-seed replica, never concatenated replicas. Soft failures are
        # exportable with their actual non-success status preserved in the manifest.
        final_sample = result.validation_samples[0]
        (root / "synthetic_ipd.csv").write_text(csv_text(final_sample.rows, [v.name for v in result.schema.variables]))
        exported_dataset = final_sample.metadata.ref.model_dump()
    manifest = {"run_id": best_state.run_id, "best_state": best_state.ref.model_dump(),
                "best_index": result.best_index, "best_feasible": result.best_feasible,
                "evaluated_states": len(result.states), "calibration_stop_reason": result.stop_reason,
                "validation_status": result.validation_report.status, "success": result.success,
                "exported_dataset": exported_dataset, "export_blocked_reason": export_blocked_reason,
                "comparison_seeds": result.task.generation.comparison_seeds,
                "validation_seeds": result.task.generation.validation_seeds,
                "pipeline_version": "m2.1-pipeline-v1", "evaluator_version": result.validation_report.evaluator_version,
                "optimizer": result.optimizer,
                "environment": {"python": platform.python_version(), "pydantic": pydantic.__version__},
                "limitations": "Hypothetical evidence only. Holdout seeds assess Monte Carlo stability, not clinical validity."}
    save("run.json", manifest)
    lines = ["# Hypothetical VCT-IPD run", "", manifest["limitations"], "",
             f"Calibration stop: **{result.stop_reason}**; validation: **{result.validation_report.status}**; success: **{result.success}**.",
             f"Selected {best_state.id} from {len(result.states)} states; no retuning on validation seeds.", "",
             (f"Canonical cohort export blocked: {export_blocked_reason}" if export_blocked_reason else
              "Canonical cohort exported: synthetic_ipd.csv (hard-valid; calibration/validation status shown above)."), "",
             "| Target | Initial | Best | Fresh seeds | Fresh status |", "| --- | ---: | ---: | ---: | --- |"]
    for initial, best, final in zip(result.reports[0].results, result.reports[result.best_index].results, result.validation_report.results):
        def display(r):
            return str(round(r.observed, 5)) if type(r.observed) in (int, float) else r.status
        lines.append(f"| {initial.target.id} | {display(initial)} | {display(best)} | {display(final)} | {final.status} |")
    (root / "report.md").write_text("\n".join(lines) + "\n")
    return root
