"""Assembly only: reuse the accepted providers, builder, generator and readiness checks."""

from vct_ipd.core import SummarySpec, TaskSpec, DataSchema
from vct_ipd.evidence.local import SummaryProvider, IPDProvider, combine
from vct_ipd.generators.statistical import StatisticalGenerator, GeneratorConfig, Parameters
from vct_ipd.targets.builder import TargetBuilder, TargetRule
from vct_ipd.pipeline.runner import ready_to_run


def assemble(task, schema, config, p, manifest):
    # Source labels are from the registered profile, NEVER from task fields.
    metadata = p['source_metadata']
    summaries = SummaryProvider(manifest['summary.json']['path']).collect(task, schema)
    ipd = IPDProvider(manifest['reference.csv']['path'], columns=p['columns'],
        summaries={f'e-ecog-{sex}': SummarySpec(metric='proportion', variables=['ecog'], category=1,
                     conditions=[{'variable': 'sex', 'values': [sex]}]) for sex in ('female', 'male')},
        **metadata).collect(task, schema)
    evidence = combine(task, schema, [summaries, ipd])
    targets = TargetBuilder().build(task, schema, evidence,
                                   [TargetRule.model_validate(r) for r in p['rules']]).targets
    generator = StatisticalGenerator()
    state = generator.build(task, schema, targets, config=config,
                            parameters=Parameters.model_validate(p['parameters']), run_id=task.id)
    return tuple(ready_to_run(task, schema, evidence, targets, state, generator))


def objects(request, normalized, p):
    data = dict(p['task'])
    data.update(sample_size=normalized['sample_size'], original_request=request.text,
                follow_up={'duration': normalized['follow_up_months'], 'unit': 'month',
                           'description': 'Fixed individual administrative follow-up; profile independent dropout.'},
                calibration={**data['calibration'], 'max_iterations': normalized['budget']})
    task = TaskSpec.model_validate(data)
    schema = DataSchema.model_validate({**p['schema'], 'task': task.ref.model_dump()})
    config = GeneratorConfig.model_validate({**p['config'], 'observation': {
        **p['config']['observation'], 'follow_up': normalized['follow_up_months']}})
    return task, schema, config
