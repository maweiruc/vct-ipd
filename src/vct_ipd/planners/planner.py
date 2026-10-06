"""Deterministic normalization, separate capability/evidence gates and M2 preflight."""

import re
from vct_ipd.llm.providers import ProviderError, parse_draft, safe_metadata
from .contracts import PlanningRequest, PlanningResult, Provenance, Issue, Draft
from .profile import profile, context, digest, input_manifest
from .bridge import objects, assemble

ALIASES = {
    'endpoints': {'overall survival': 'OS', '总生存期': 'OS', 'os': 'OS', 'pfs': 'PFS', '无进展生存期': 'PFS'},
    'disease': {'nsclc': 'hypothetical NSCLC', 'hypothetical nsclc': 'hypothetical NSCLC', '假设性非小细胞肺癌': 'hypothetical NSCLC'},
    'stage': {'晚期': 'advanced', 'advanced': 'advanced'},
    'intent': {'hypothetical': 'hypothetical', '假设性': 'hypothetical', 'clinical': 'clinical', '真实临床': 'clinical'},
    'origin': {'treatment start': 'treatment_start', '治疗开始': 'treatment_start'},
    'follow_up_kind': {'fixed per-patient': 'fixed_individual', '固定个体随访': 'fixed_individual'},
    'follow_up_unit': {'months': 'month', '月': 'month', 'years': 'year', '年': 'year'},
    'variables': {'年龄': 'age', '性别': 'sex', 'ecog': 'ecog', 'os time': 'os_time', 'os event': 'os_event'},
}
SCALARS = {'disease', 'stage', 'geography', 'line_of_therapy', 'regimen', 'setting', 'origin',
           'intent', 'arms', 'sample_size', 'budget', 'follow_up_kind', 'follow_up_duration', 'follow_up_unit', 'ecog_values'}
# A small conservative backstop, not a general semantic parser. Even negated uses
# require clarification rather than risking silently executing the opposite intent.
GUARDS = ('skip validation', 'set success=true', '跳过验证', 'not hypothetical', '真实患者',
          'not os', '不要os', 'recruitment', 'enrollment', '入组期', '招募', 'median follow-up', '中位随访')


def error_result(request, p, code, message, *, manifest=None, metadata=None):
    return PlanningResult(request=request, status='ERROR', profile_hash=digest(p), context_hash=digest(context(p)),
        input_manifest=manifest or {}, provider_metadata=metadata or {},
        issues=[Issue(code=code, boundary='input' if code == 'input_invalid' else 'extraction', message=message)])


def resolve(request, draft, p, manifest, metadata=None):
    result = PlanningResult(request=request, draft=draft, status='NEEDS_CLARIFICATION',
        profile_hash=digest(p), context_hash=digest(context(p)), input_manifest=manifest,
        provider_metadata=metadata or {}, warnings=['Hypothetical profile and synthetic evidence only; no clinical validation.',
        'Structured extraction and verified quotations do not prove semantic understanding; review the saved draft.'])
    def issue(code, boundary, message):
        result.issues.append(Issue(code=code, boundary=boundary, message=message))
    if not draft.claims and not draft.additional_requirements:
        issue('empty_interpretation', 'extraction', 'No stated intent extracted; clarify the scenario')
    for phrase in GUARDS:
        if phrase in request.text.casefold():
            issue('text_requires_review', 'extraction', f'Explicit ambiguity/override marker requires review: {phrase}')
    for ambiguity in draft.ambiguities:
        issue('ambiguous', 'extraction', ambiguity)
    values = {}
    for claim in draft.claims:
        if not claim.quote.strip() or claim.quote not in request.text:
            issue('invalid_quote', 'extraction', f'Unattributed claim: {claim.field}')
            continue
        value = ALIASES.get(claim.field, {}).get(claim.value.casefold(), claim.value)
        result.provenance.append(Provenance(field=claim.field, source='user', value=claim.value, quote=claim.quote))
        if value != claim.value:
            result.provenance.append(Provenance(field=claim.field, source='normalization', value=value,
                                               quote=claim.quote, rule='Registered lexical alias'))
        if claim.field not in SCALARS | {'endpoints', 'variables'}:
            issue('unsupported_requirement', 'capability', f'{claim.field}: {claim.value}')
        if claim.field in SCALARS and claim.field in values:
            issue('contradictory', 'extraction', f'Multiple claims for {claim.field}; no last-value-wins')
        values.setdefault(claim.field, []).append(value)
    for requirement in draft.additional_requirements:
        valid = bool(requirement.quote.strip()) and requirement.quote in request.text
        issue('pending_requirement' if valid else 'invalid_quote', 'capability' if valid else 'extraction',
              requirement.description + '; requires supported structured configuration/evidence, not prose adoption')
        result.provenance.append(Provenance(field='additional_requirement', source='unresolved',
                                           value=requirement.description, quote=requirement.quote))
    expected = {'disease': p['task']['population']['disease'], 'stage': 'advanced',
                'geography': None, 'line_of_therapy': None, 'regimen': 'DEMO-A',
                'setting': p['task']['treatment']['setting'], 'origin': 'treatment_start',
                'intent': 'hypothetical', 'arms': '1', 'ecog_values': '0,1',
                'endpoints': ['OS'], 'variables': p['capabilities']['variables'],
                'follow_up_kind': 'fixed_individual', 'sample_size': '1200', 'budget': '101'}
    resolved = {}
    for field, default in expected.items():
        if field in values:
            value = values[field] if field in ('endpoints', 'variables') else values[field][0]
            resolved[field] = value
            if field in ('sample_size', 'budget'):
                continue
            matches = (set(value) <= set(default) and bool(value)) if field == 'variables' else value == default
            if not matches:
                boundary = 'evidence' if field in ('disease', 'stage', 'geography', 'line_of_therapy', 'regimen', 'setting', 'origin', 'intent') else 'capability'
                issue('evidence_mismatch' if boundary == 'evidence' else 'backend_unsupported', boundary,
                      f'Requested {field}={value!r}; selected profile/backend provides {default!r}')
        else:
            resolved[field] = default
            result.provenance.append(Provenance(field=field, source='profile', value=default,
                                               rule=f"Explicitly selected {p['id']}:{p['version']}; hypothetical default"))
    for field in ('sample_size', 'budget'):
        value = resolved[field]
        if not re.fullmatch(r'[1-9][0-9]*', value) or len(value) > 8 or int(value) > p['limits'][field]:
            issue('resource_limit', 'resource', f'{field} must be an integer in [1, {p["limits"][field]}]')
        else:
            resolved[field] = int(value)
            result.provenance.append(Provenance(field=field, source='normalization', value=int(value), rule='Strict positive integer text'))
    follow_fields = {'follow_up_duration', 'follow_up_unit', 'follow_up_kind'}
    if follow_fields & values.keys():
        if not follow_fields <= values.keys():
            issue('follow_up_ambiguous', 'extraction', 'Duration, unit and fixed individual semantics must all be explicit')
        else:
            duration, unit = values['follow_up_duration'][0], values['follow_up_unit'][0]
            if not re.fullmatch(r'[0-9]+(?:\.[0-9]+)?', duration) or len(duration) > 16 or unit not in ('month', 'year'):
                issue('follow_up_invalid', 'capability', 'Unsupported duration/unit')
            else:
                months = float(duration) * (12 if unit == 'year' else 1)
                months = int(months) if months.is_integer() else months
                resolved['follow_up_months'] = months
                if not p['limits']['follow_up_months'][0] <= months <= p['limits']['follow_up_months'][1]:
                    issue('follow_up_limit', 'resource', 'Profile requires 24–120 months, including support for S(24)')
                result.provenance.append(Provenance(field='follow_up_months', source='normalization', value=months,
                                                   rule=f'{duration} {unit} times {12 if unit == "year" else 1}; rate remains 0.008/month'))
    else:
        resolved['follow_up_months'] = 36
        result.provenance.append(Provenance(field='follow_up_months', source='profile', value=36,
                                           rule='Profile fixed individual follow-up'))
    for field, value in [('seeds', p['task']['generation']), ('calibration_policy', p['task']['calibration']),
                         ('target_rules', p['rules']), ('observation', p['config']['observation']),
                         ('initial_parameters', p['parameters']), ('schema', p['schema'])]:
        result.provenance.append(Provenance(field=field, source='profile', value=value, rule='Fixed registered policy, not LLM output'))
    result.provenance.append(Provenance(field='model_assumptions', source='assumption',
        value=['Conditional baseline factorization', 'Weibull proportional hazards', 'Independent censoring'],
        rule='Inherited statistical model assumptions, not established by these summaries'))
    result.normalized = resolved
    if result.issues:
        result.status = 'UNSUPPORTED' if any(i.boundary in ('capability', 'evidence', 'resource') for i in result.issues) else 'NEEDS_CLARIFICATION'
        return result
    try:
        task, schema, execution = objects(request, resolved, p)
        assemble(task, schema, execution, p, manifest)  # Loads evidence and validates; no samples.
    except (ValueError, OSError, TypeError, KeyError) as exc:
        issue('preflight_failed', 'input', f'Local evidence/configuration preflight failed: {exc}')
        result.status = 'ERROR'
        return result
    result.task, result.data_schema, result.execution = task, schema, execution
    result.status = 'READY'
    return result


def plan(request: PlanningRequest, provider, directory):
    request = PlanningRequest.model_validate(request.model_dump())
    p = profile(request.profile)
    try:
        manifest = input_manifest(p, directory)
    except (ValueError, OSError) as exc:
        return error_result(request, p, 'input_invalid', str(exc))
    try:
        raw = provider.extract(request.text, context(p))
        draft = parse_draft(raw, p['limits']['response_bytes'])
    except ProviderError as exc:
        return error_result(request, p, exc.code, str(exc), manifest=manifest, metadata=safe_metadata(provider))
    except Exception:
        # Do not persist transport exception bodies, which can contain secrets.
        return error_result(request, p, 'provider_error', 'Extraction provider failed', manifest=manifest, metadata=safe_metadata(provider))
    return resolve(request, draft, p, manifest, safe_metadata(provider))
