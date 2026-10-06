"""Small, deterministic row metrics and right-censored Kaplan–Meier estimates."""

from dataclasses import dataclass
import math
import statistics

from vct_ipd.core import DataSchema, EstimateStatus, Metric, SummarySpec
from vct_ipd.core.schema import same_value

SUPPORTED = frozenset({Metric.VALIDITY, Metric.MEAN, Metric.SD, Metric.QUANTILE,
                       Metric.PROPORTION, Metric.CORRELATION,
                       Metric.SURVIVAL_PROBABILITY, Metric.MEDIAN_SURVIVAL})

# Absolute survival-probability roundoff allowance, only for the median crossing.
# This neither rounds S(t) nor changes target/calibration tolerances.
KM_HALF_ABS_TOL = 1e-12


@dataclass(frozen=True)
class Measurement:
    value: float | None = None
    status: EstimateStatus = EstimateStatus.AVAILABLE
    reason: str | None = None
    lower_bound: float | None = None

    def fields(self, value_name="observed"):
        return {value_name: self.value, "estimate_status": self.status,
                "estimate_reason": self.reason,
                "estimate_bound": {"lower": self.lower_bound} if self.lower_bound is not None else None}


def unavailable(reason, status=EstimateStatus.NOT_ESTIMABLE, lower_bound=None):
    return Measurement(None, status, reason, lower_bound)


def validate_metric(summary: SummarySpec, schema: DataSchema):
    if summary.metric not in SUPPORTED:
        raise ValueError(f"unsupported active metric: {summary.metric}")
    if summary.method is not None and not (summary.metric == Metric.CORRELATION and summary.method == "pearson"):
        raise ValueError(f"unsupported metric method: {summary.method}")
    summary.validate_schema(schema, canonical=True)


def selected(rows, summary):
    def includes(row):
        for condition in summary.conditions:
            value = row.get(condition.variable)
            if value is None:
                return False
            if condition.values is not None:
                if not any(same_value(value, v) for v in condition.values):
                    return False
            elif ((condition.range.lower is not None and value < condition.range.lower)
                  or (condition.range.upper is not None and value > condition.range.upper)):
                return False
        return True
    return [row for row in rows if includes(row)]


def validity_errors(rows, schema: DataSchema, latent=None):
    """Count invalid rows, including observed/latent survival logic when available."""
    if schema.constraints or any(v.constraints for v in schema.variables):
        raise ValueError("expression/description constraints have no executable evaluator yet")
    time, event = schema.survival_columns()
    invalid = 0
    if latent is not None and len(latent) != len(rows):
        raise ValueError("latent and observed row counts differ")
    for index, row in enumerate(rows):
        try:
            for variable in schema.variables:
                if variable.required and variable.name not in row:
                    raise ValueError("missing required column")
                if variable.name not in row:
                    continue
                value = row[variable.name]
                if value is None:
                    if variable.missingness == "forbidden":
                        raise ValueError("missing value")
                    continue
                if isinstance(value, float) and not math.isfinite(value):
                    raise ValueError("nonfinite value")
                variable.validate_value(value)
            if row.get(time.name) is None or row.get(event.name) is None or row[time.name] < 0:
                raise ValueError("invalid observed survival")
            if latent is not None:
                record = latent[index]
                t, c = record["event_time"], record["censoring_time"]
                if not all(math.isfinite(v) and v >= 0 for v in (t, c)):
                    raise ValueError("invalid latent survival")
                if row[time.name] != min(t, c) or bool(row[event.name]) != (t <= c):
                    raise ValueError("observation process mismatch")
        except (ValueError, TypeError, KeyError):
            invalid += 1
    return invalid


def kaplan_meier(times, events, *, at_time=None):
    """Right-continuous KM; deaths precede tied censoring in the risk set.

    No extrapolation beyond the last observed time, even when the curve is zero.
    A nonreached median bound uses only the last event time, never max censoring.
    """
    if len(times) != len(events):
        raise ValueError("times and events have unequal lengths")
    if at_time is not None and (not math.isfinite(at_time) or at_time < 0):
        raise ValueError("invalid evaluation time")
    if any(type(t) not in (int, float) or not math.isfinite(t) or t < 0 for t in times):
        raise ValueError("invalid event/censoring time")
    if any(type(e) not in (int, bool) or e not in (0, 1) for e in events):
        raise ValueError("invalid event indicator")
    if not times:
        return unavailable("Empty sample/subgroup")
    if at_time is not None and at_time > max(times):
        return unavailable("Requested time exceeds observed support; no tail extrapolation")
    counts = {}
    for t, e in zip(times, events):
        deaths, total = counts.get(t, (0, 0))
        counts[t] = (deaths + int(e), total + 1)
    risk, survival, last_event = len(times), 1.0, None
    for time in sorted(counts):
        if at_time is not None and time > at_time:
            break
        deaths, total = counts[time]
        if deaths:
            survival *= 1 - deaths / risk
            last_event = time
            if at_time is None and (survival <= 0.5 or math.isclose(
                    survival, 0.5, rel_tol=0.0, abs_tol=KM_HALF_ABS_TOL)):
                return Measurement(float(time))
        risk -= total
    if at_time is not None:
        return Measurement(survival)
    if last_event is None:
        return unavailable("No observed events; median and a supported event-based bound unavailable")
    return unavailable("KM did not reach 0.5 within event support", EstimateStatus.NOT_REACHED, float(last_event))


def measure(rows, summary: SummarySpec, schema: DataSchema):
    validate_metric(summary, schema)
    if summary.metric == Metric.VALIDITY:
        raise ValueError("validity is evaluated as a structured row check")
    rows = selected(rows, summary)
    if not rows:
        return unavailable("Empty sample/subgroup")
    if any(row.get(name) is None for row in rows for name in summary.variables):
        return unavailable("Missing analysis values; no implicit complete-case policy")
    if summary.metric in (Metric.SURVIVAL_PROBABILITY, Metric.MEDIAN_SURVIVAL):
        time, event = schema.survival_columns()
        return kaplan_meier([r[time.name] for r in rows], [r[event.name] for r in rows],
                           at_time=summary.at_time)
    if summary.metric == Metric.PROPORTION:
        return Measurement(sum(same_value(r[summary.variables[0]], summary.category) for r in rows) / len(rows))
    by_name = {v.name: v for v in schema.variables}
    def numbers(name):
        variable = by_name[name]
        if variable.numeric_encoding:
            return [float(next(e.score for e in variable.numeric_encoding if same_value(e.value, r[name]))) for r in rows]
        return [float(r[name]) for r in rows]
    x = numbers(summary.variables[0])
    if not all(math.isfinite(v) for v in x):
        return unavailable("Nonfinite analysis values")
    if summary.metric == Metric.MEAN:
        return Measurement(statistics.fmean(x))
    if summary.metric == Metric.SD:
        return Measurement(statistics.stdev(x)) if len(x) >= 2 else unavailable("SD requires two observations")
    if summary.metric == Metric.QUANTILE:
        x.sort()
        index = (len(x) - 1) * summary.quantile
        lo, hi = math.floor(index), math.ceil(index)
        return Measurement(x[lo] + (x[hi] - x[lo]) * (index - lo))
    y = numbers(summary.variables[1])
    if len(x) < 2 or len(set(x)) < 2 or len(set(y)) < 2:
        return unavailable("Correlation requires two observations and nonconstant variables")
    return Measurement(statistics.correlation(x, y))
