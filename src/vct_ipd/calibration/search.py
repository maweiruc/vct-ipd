"""Bounded coordinate proposals and explicit comparison-seed stopping rules."""

from vct_ipd.generators.statistical import Parameters


class CoordinateSearch:
    # Fixed, small tunable set. Every proposal is applied to the best feasible parameters.
    dimensions = (("age_mean", 4, 30, 85), ("male_probability", 0.15, 0.05, 0.95),
                  ("ecog_intercept", 0.6, -4, 4), ("ecog_male", 0.6, -4, 4),
                  ("weibull_scale", 8, 2, 80))

    def propose(self, best, attempt):
        width = 2 * len(self.dimensions)
        name, step, lower, upper = self.dimensions[(attempt % width) // 2]
        direction = 1 if attempt % 2 == 0 else -1
        # Two complete sweeps at each resolution; finite run budget caps search.
        step *= 0.5 ** (attempt // (2 * width))
        parameters = dict(best.parameters)
        parameters[name] = min(upper, max(lower, parameters[name] + direction * step))
        return Parameters.model_validate(parameters).model_dump(mode="json")


def feasible(report, targets):
    results = {r.target.id: r for r in report.results}
    return all(results[t.id].status == "PASS" for t in targets.items if t.hard_constraint)


def better(candidate, incumbent, targets):
    c, b = feasible(candidate, targets), feasible(incumbent, targets)
    if c != b:
        return c
    # Within hard-feasible states, acceptance takes precedence over center loss.
    # An already passing state must be the state selected for stopping/validation.
    if c and (candidate.status == "PASS") != (incumbent.status == "PASS"):
        return candidate.status == "PASS"
    return candidate.overall_metrics["loss"] < incumbent.overall_metrics["loss"] - 1e-9


def stop_reason(report, *, evaluated, budget, stale, patience):
    if report.status == "PASS":
        return "calibrated"
    if report.status == "EVIDENCE_LIMITED":
        return "evidence_limited"
    if stale >= patience:
        return "no_improvement"
    if evaluated >= budget:
        return "max_iterations"
    return None
