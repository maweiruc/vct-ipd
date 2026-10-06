"""Fixed small statistical model; no generated-code execution or row repair."""

import csv
from dataclasses import dataclass
import hashlib
import io
import math
import random
from statistics import NormalDist
from typing import Literal

from pydantic import Field

from vct_ipd.core import DataArtifact, GeneratorState, SyntheticDataset
from vct_ipd.core.base import Contract, Number, Text


class ObservationConfig(Contract):
    mechanism: Literal["fixed_individual"] = "fixed_individual"
    follow_up: Number = Field(gt=0, le=10000)
    unit: Text
    dropout_rate: Number = Field(default=0, ge=0, le=10)  # Per observation time unit.


class Columns(Contract):
    age: Text = "age"
    sex: Text = "sex"
    ecog: Text = "ecog"
    time: Text = "os_time"
    event: Text = "os_event"


class GeneratorConfig(Contract):
    family: Literal["conditional-baseline-weibull-ph"] = "conditional-baseline-weibull-ph"
    columns: Columns = Field(default_factory=Columns)
    observation: ObservationConfig


class Parameters(Contract):
    age_mean: Number = Field(default=65, ge=30, le=85)
    age_sd: Number = Field(default=8, ge=1, le=20)
    male_probability: Number = Field(default=0.6, ge=0.05, le=0.95)
    ecog_intercept: Number = Field(default=-0.6, ge=-4, le=4)
    ecog_age: Number = Field(default=0.35, ge=-2, le=2)
    ecog_male: Number = Field(default=1.2, ge=-4, le=4)
    weibull_scale: Number = Field(default=28, ge=2, le=80)
    weibull_shape: Number = Field(default=1.25, ge=0.5, le=3)
    log_hazard_age: Number = Field(default=0.1, ge=-2, le=2)
    log_hazard_ecog: Number = Field(default=0.35, ge=-2, le=2)


def csv_text(rows, columns):
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=columns, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()


@dataclass
class Sample:
    metadata: SyntheticDataset
    rows: list[dict]
    latent: list[dict]


def snapshot(model):
    return type(model).model_validate_json(model.model_dump_json())


class StatisticalGenerator:
    def validate_configuration(self, task, schema, state):
        if state.generator_type != "statistical":
            raise ValueError("unsupported generator type")
        config = GeneratorConfig.model_validate(state.model_specification)
        Parameters.model_validate(state.parameters)
        c, variables = config.columns, {v.name: v for v in schema.variables}
        names = list(c.model_dump().values())
        if len(set(names)) != 5 or set(names) != variables.keys():
            raise ValueError("generator supports exactly five distinctly mapped columns")
        if schema.constraints or any(v.constraints or v.derived_from or v.numeric_encoding for v in schema.variables):
            raise ValueError("generator does not execute custom constraints, encodings or derivations")
        if any(not v.required or v.missingness != "forbidden" for v in schema.variables):
            raise ValueError("this generator requires complete required columns")
        age, sex, ecog = (variables[name] for name in (c.age, c.sex, c.ecog))
        if any(v.role != "baseline_covariate" for v in (age, sex, ecog)):
            raise ValueError("baseline column role mismatch")
        if age.type != "continuous" or age.unit != "year" or not age.range or age.range.lower is None or age.range.upper is None or not 0 <= age.range.lower < age.range.upper <= 120:
            raise ValueError("age requires finite ordered bounds within [0, 120] years")
        if sex.type != "categorical" or set(sex.allowed_values or []) != {"female", "male"}:
            raise ValueError("sex must declare female/male categories")
        if ecog.type != "integer" or set(ecog.allowed_values or []) != {0, 1}:
            raise ValueError("this demo supports binary ECOG [0, 1]")
        time, event = schema.survival_columns()
        if (time.name, event.name) != (c.time, c.event) or time.type != "continuous":
            raise ValueError("survival role mapping mismatch")
        if task.endpoint.name != "OS" or config.observation.unit != task.endpoint.unit or time.unit != task.endpoint.unit:
            raise ValueError("generator requires a single OS endpoint and resolved observation time units")
        if task.follow_up is not None and (task.follow_up.unit != config.observation.unit or task.follow_up.duration != config.observation.follow_up):
            raise ValueError("FollowUp conflicts with explicit observation configuration")
        return config

    def build(self, task, schema, targets, *, config, parameters, run_id="demo"):
        state = GeneratorState(id=f"{run_id}-g0", run_id=run_id, task=task.ref, schema_ref=schema.ref,
                               target_set=targets.ref, generator_type="statistical", iteration=0,
                               model_specification=config.model_dump(mode="json"), parameters=parameters.model_dump(mode="json"),
                               code_version="m2-statistical-v1", config_version="1", modification_reason="Initial parameters",
                               assumptions=[
                                   {"id": "factorization", "statement": "Age and sex are independent; ECOG depends on both through the configured logistic model.",
                                    "rationale": "Explicit small demonstration model, not identified by marginal summaries."},
                                   {"id": "weibull-ph", "statement": "OS follows Weibull proportional hazards with the configured covariate coefficients.",
                                    "rationale": "A fixed model family supplies otherwise unidentified structure."},
                                   {"id": "censoring", "statement": "Individual administrative follow-up is fixed; dropout is independent exponential censoring.",
                                    "rationale": "Explicit simulation observation-process assumption, not inferred from follow-up prose."}])
        state.validate_references(task, schema, targets)
        self.validate_configuration(task, schema, state)
        return state

    def sample(self, task, schema, state, seed, *, phase="comparison"):
        if phase not in ("comparison", "validation"):
            raise ValueError("unknown sampling phase")
        if type(seed) is not int or seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        config = self.validate_configuration(task, schema, state)
        p, c = Parameters.model_validate(state.parameters), config.columns
        streams = {name: random.Random(int.from_bytes(hashlib.sha256(f"{seed}:{name}".encode()).digest(), "big"))
                   for name in ("age", "sex", "ecog", "event", "censor")}
        age_spec = next(v for v in schema.variables if v.name == c.age)
        distribution = NormalDist(p.age_mean, p.age_sd)
        low, high = distribution.cdf(age_spec.range.lower), distribution.cdf(age_spec.range.upper)
        if not low < high or low >= 1 or high <= 0:
            raise ValueError("truncated age distribution has no numerically resolvable mass")
        rows, latent = [], []
        event_bool = next(v for v in schema.variables if v.name == c.event).type == "boolean"
        for _ in range(task.sample_size):
            q = low + (high - low) * streams["age"].random()
            # Endpoint protection is for inverse-CDF arithmetic, not row calibration.
            age = distribution.inv_cdf(min(math.nextafter(1.0, 0.0), max(math.nextafter(0.0, 1.0), q)))
            male = streams["sex"].random() < p.male_probability
            ecog_prob = 1 / (1 + math.exp(-(p.ecog_intercept + p.ecog_age * (age - 65) / 10 + p.ecog_male * male)))
            ecog = int(streams["ecog"].random() < ecog_prob)
            eta = p.log_hazard_age * (age - 65) / 10 + p.log_hazard_ecog * ecog
            event_time = p.weibull_scale * (streams["event"].expovariate(1) / math.exp(eta)) ** (1 / p.weibull_shape)
            dropout = streams["censor"].expovariate(config.observation.dropout_rate) if config.observation.dropout_rate else None
            censor_time = min(config.observation.follow_up, dropout) if dropout is not None else config.observation.follow_up
            event = event_time <= censor_time
            rows.append({c.age: age, c.sex: "male" if male else "female", c.ecog: ecog,
                         c.time: min(event_time, censor_time), c.event: event if event_bool else int(event)})
            latent.append({"event_time": event_time, "censoring_time": censor_time, "dropout_time": dropout})
        name = f"iteration_{state.iteration:03d}_{phase}_{seed}"
        content = csv_text(rows, [v.name for v in schema.variables]).encode()
        metadata = SyntheticDataset(id=f"{state.run_id}-{name}", run_id=state.run_id, iteration=state.iteration,
                                    generator_state=state.ref, schema_ref=schema.ref, seed=seed, sample_size=len(rows),
                                    data=DataArtifact(uri=f"datasets/{name}.csv", format="csv",
                                                      content_hash=hashlib.sha256(content).hexdigest()))
        metadata.validate_references(state, schema)
        return Sample(metadata, rows, latent)

    def update(self, previous, action, report):
        action.validate_references(report, previous)
        if action.type != "TUNE_PARAMETERS":
            raise ValueError("statistical generator supports parameter tuning only")
        parameters = Parameters.model_validate(action.payload.parameters)
        data = previous.model_dump(mode="json")
        data.update(id=f"{previous.run_id}-g{previous.iteration + 1}", iteration=previous.iteration + 1,
                    parent_state=previous.ref.model_dump(), parameters=parameters.model_dump(mode="json"),
                    modification_reason=action.reason)
        child = GeneratorState.model_validate(data)
        child.validate_parent(previous)
        return child
