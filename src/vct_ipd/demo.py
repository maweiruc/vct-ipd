"""Offline hypothetical cohort demonstration. Run with python -m vct_ipd.demo."""

import argparse
from pathlib import Path

from vct_ipd.core import DataSchema, SummarySpec, TaskSpec, VariableSpec
from vct_ipd.evidence.local import IPDProvider, SummaryProvider, combine
from vct_ipd.generators.statistical import GeneratorConfig, ObservationConfig, Parameters, StatisticalGenerator
from vct_ipd.pipeline.runner import run
from vct_ipd.reporting.writer import write_run
from vct_ipd.targets.builder import TargetBuilder, TargetRule


def demo_context(*, sample_size=1200, budget=101, patience=30):
    task = TaskSpec(id="hypothetical-nsclc", population={"disease": "hypothetical NSCLC", "stage": "advanced"},
                    treatment={"setting": "single hypothetical cohort", "regimen": "DEMO-A"},
                    endpoint={"name": "OS", "time_origin": "treatment_start", "unit": "month"},
                    sample_size=sample_size, intended_use="Software demonstration with entirely synthetic evidence",
                    generation={"comparison_seeds": [101, 202], "validation_seeds": [303, 404]},
                    calibration={"max_iterations": budget, "no_improvement_patience": patience},
                    assumptions=[{"id": "demo-only", "statement": "All reference values are hypothetical.",
                                  "rationale": "Validate software behavior, not clinical realism or identifiability."}])
    schema = DataSchema(id="nsclc-schema", task=task.ref, variables=[
        VariableSpec(name="age", label="Baseline age", type="continuous", role="baseline_covariate", unit="year",
                     range={"lower": 18, "upper": 100}, generation_order=0),
        VariableSpec(name="sex", label="Sex", type="categorical", role="baseline_covariate",
                     allowed_values=["female", "male"], generation_order=0),
        VariableSpec(name="ecog", label="Binary demo ECOG", type="integer", role="baseline_covariate",
                     allowed_values=[0, 1], generation_order=1),
        VariableSpec(name="os_time", label="Observed OS", type="continuous", role="observed_time", unit="month",
                     range={"lower": 0}, generation_order=2),
        VariableSpec(name="os_event", label="Observed death", type="boolean", role="event_indicator", generation_order=2),
    ])
    config = GeneratorConfig(observation=ObservationConfig(follow_up=36, unit="month", dropout_rate=0.008))
    return task, schema, config


def prepare(inputs, *, sample_size=1200, budget=101, patience=30):
    task, schema, config = demo_context(sample_size=sample_size, budget=budget, patience=patience)
    inputs = Path(inputs)
    summaries = SummaryProvider(inputs / "summary.json").collect(task, schema)
    ipd = IPDProvider(inputs / "reference.csv", columns={"age": "AGE_YEARS", "sex": "SEX", "ecog": "ECOG",
                                                       "os_time": "OS_MONTHS", "os_event": "DEATH"},
                      summaries={f"e-ecog-{sex}": SummarySpec(metric="proportion", variables=["ecog"], category=1,
                                   conditions=[{"variable": "sex", "values": [sex]}]) for sex in ("female", "male")},
                      population=task.population, treatment=task.treatment, endpoint=task.endpoint,
                      nature="demo").collect(task, schema)
    evidence = combine(task, schema, [summaries, ipd])
    rules = [TargetRule(evidence_id=f"e-{name}", target_id=name, tolerance={"absolute": tolerance})
             for name, tolerance in (("age", 1.2), ("sex", 0.04), ("ecog-female", 0.05), ("ecog-male", 0.05),
                                     ("survival24", 0.04), ("median", 1.5))]
    built = TargetBuilder().build(task, schema, evidence, rules)
    initial = Parameters(age_mean=57, male_probability=0.3, ecog_intercept=-1.8, ecog_male=0,
                         weibull_scale=12)
    state = StatisticalGenerator().build(task, schema, built.targets, config=config, parameters=initial,
                                         run_id="hypothetical-nsclc")
    return task, schema, evidence, built.targets, state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, default=Path("examples/nsclc_os"))
    parser.add_argument("--output", type=Path, default=Path("runs/nsclc-os"))
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists; choose a new directory to preserve run history")
    result = run(*prepare(args.inputs))
    root = write_run(result, args.output)
    print((root / "report.md").read_text())
    print(f"Run package: {root.resolve()}")


if __name__ == "__main__":
    main()
