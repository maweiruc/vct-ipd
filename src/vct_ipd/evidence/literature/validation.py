"""Conservative labelled-passage grammar; never infer biomedical scope from a query."""
import re
from .contracts import Scope, Proposal, EvidenceBatch, CandidateResult
from .retrieval import sha
from vct_ipd.core import Evidence
from vct_ipd.planners.profile import digest

NORMALIZER = 'labelled-statistics-v1'
NUM = r'(?:0|[1-9][0-9]*)(?:\.[0-9]+)?'
UNIT = r'(?:months?|years?)'
SCOPE_KEYS = {'Population': 'disease', 'Stage': 'stage', 'Geography': 'geography', 'Line': 'line_of_therapy',
              'Regimen': 'regimen', 'Setting': 'setting', 'Endpoint': 'endpoint', 'Origin': 'origin', 'Subgroup': 'subgroup'}


class ReviewNeeded(ValueError):
    pass


def fraction(text):
    if re.fullmatch(NUM + '%', text):
        return float(text[:-1]) / 100
    if re.fullmatch(r'[0-9]+/[1-9][0-9]*', text):
        n, d = map(int, text.split('/'))
        if n > d:
            raise ReviewNeeded('Numerator exceeds stated denominator')
        return n / d
    if re.fullmatch(NUM, text) and 0 <= float(text) <= 1:
        return float(text)
    raise ReviewNeeded('Probability needs a fraction, explicit percent or numerator/denominator')


def parse_block(quote):
    """Exactly two lines: source scope, then one statistic with optional suffixes."""
    lines = quote.split('\n')
    if len(lines) != 2:
        raise ReviewNeeded('Only two-line labelled scope/statistic blocks are machine-verifiable')
    scope = {v: None for v in SCOPE_KEYS.values()}
    seen = set()
    for token in lines[0].split('; '):
        key, sep, value = token.partition(': ')
        if not sep or key not in SCOPE_KEYS or key in seen or not value:
            raise ReviewNeeded('Ambiguous or unsupported scope label')
        seen.add(key); scope[SCOPE_KEYS[key]] = None if value == '?' else value
    fields = dict(metric='', value=None, unit=None, category=None, at_time=None, endpoint=None,
                  estimate_status='available', lower_bound=None, ci_lower=None, ci_upper=None,
                  confidence_level=None, standard_error=None, sample_size=None)
    parts = lines[1].split('; ')
    first = parts[0]
    m = re.fullmatch(f'Mean age: ({NUM}) ({UNIT})', first)
    if m:
        fields.update(metric='mean', value=float(m[1]), unit=m[2].rstrip('s'))
    elif m := re.fullmatch(r'Proportion (male|female): (.+)', first):
        fields.update(metric='proportion', value=fraction(m[2]), category=m[1])
    elif m := re.fullmatch(f'(OS|PFS) survival at ({NUM}) ({UNIT}): (.+)', first):
        fields.update(metric='survival_probability', value=fraction(m[4]), endpoint=m[1], at_time=float(m[2]), unit=m[3].rstrip('s'))
    elif m := re.fullmatch(f'Median (OS|PFS): ({NUM}|not reached|not reported|not estimable) ({UNIT})', first):
        fields.update(metric='median_survival', endpoint=m[1], unit=m[3].rstrip('s'))
        if m[2].startswith('not '):
            fields['estimate_status'] = m[2].replace(' ', '_')
        else:
            fields['value'] = float(m[2])
    else:
        raise ReviewNeeded('Unsupported statistic semantics: no mean/median, OS/PFS or follow-up substitution')
    option_keys = set()
    for part in parts[1:]:
        if m := re.fullmatch(f'({NUM})% CI: (.+) to (.+)', part):
            key = 'ci'
            fields['confidence_level'] = float(m[1]) / 100
            if fields['metric'] in ('proportion', 'survival_probability'):
                fields['ci_lower'], fields['ci_upper'] = fraction(m[2]), fraction(m[3])
            else:
                def quantity(value):
                    x = re.fullmatch(f'({NUM}) ({UNIT})', value)
                    if not x or x[2].rstrip('s') != fields['unit']:
                        raise ReviewNeeded('Uncertainty units must explicitly match the source point units')
                    return float(x[1])
                fields['ci_lower'], fields['ci_upper'] = quantity(m[2]), quantity(m[3])
        elif m := re.fullmatch(r'SE: (.+)', part):
            key = 'se'
            if fields['metric'] in ('proportion', 'survival_probability'):
                fields['standard_error'] = fraction(m[1])
            else:
                x = re.fullmatch(f'({NUM}) ({UNIT})', m[1])
                if not x or x[2].rstrip('s') != fields['unit']:
                    raise ReviewNeeded('SE units mismatch')
                fields['standard_error'] = float(x[1])
        elif m := re.fullmatch(r'N: ([1-9][0-9]*)', part):
            key = 'n'; fields['sample_size'] = int(m[1])
        elif m := re.fullmatch(f'Lower bound: ({NUM}) ({UNIT})', part):
            key = 'bound'
            if m[2].rstrip('s') != fields['unit']:
                raise ReviewNeeded('Bound units mismatch')
            fields['lower_bound'] = float(m[1])
        else:
            raise ReviewNeeded('Unsupported uncertainty/qualifier (SD, SE, IQR and CI are not interchangeable)')
        if key in option_keys:
            raise ReviewNeeded('Repeated statistic qualifier')
        option_keys.add(key)
    if fields['ci_lower'] is not None:
        if not 0 < fields['confidence_level'] < 1 or fields['ci_lower'] > fields['ci_upper']:
            raise ReviewNeeded('Invalid confidence interval')
        if fields['value'] is None or not fields['ci_lower'] <= fields['value'] <= fields['ci_upper']:
            raise ReviewNeeded('CI must contain its available point estimate')
    if fields['estimate_status'] != 'available' and fields['standard_error'] is not None:
        raise ReviewNeeded('SE without an available point requires review')
    if '/' in first and fields['sample_size'] is not None:
        count = re.search(r'([0-9]+)/([1-9][0-9]*)$', first)
        if count and int(count[2]) != fields['sample_size']:
            raise ReviewNeeded('Count denominator differs from statistic N; subgroup denominator is ambiguous')
    return Scope.model_validate(scope), Proposal.model_validate(fields)


def evidence_id(doc, support):
    return 'lit-' + sha(doc.id + ':' + str(support.start))[:20]


def validate(draft, documents):
    batch, by_id, seen = EvidenceBatch(), {d.id: d for d in documents}, set()
    for index, candidate in enumerate(draft.candidates):
        status, reasons, eid = 'rejected', [], None
        doc = by_id.get(candidate.document_id)
        try:
            if doc is None:
                raise ValueError('Unknown source identity')
            s = candidate.support
            if sha(doc.text) != doc.content_hash or not 0 <= s.start < s.end <= len(doc.text) or doc.text[s.start:s.end] != s.quote:
                raise ValueError('Quote/offset/content hash mismatch')
            if (s.start and doc.text[max(0, s.start-2):s.start] != '\n\n') or (s.end < len(doc.text) and doc.text[s.end:s.end+2] != '\n\n'):
                raise ValueError('Quote must cover a complete paragraph block')
            scope, stat = parse_block(s.quote)
            if scope != candidate.scope or stat != candidate.statistic:
                raise ValueError('Proposed numerical value, scope or statistic disagrees with the source grammar')
            if candidate.unknowns:
                raise ReviewNeeded('Extractor reported unresolved fields: ' + '; '.join(candidate.unknowns))
            if any(getattr(scope, key) is None for key in ('disease', 'regimen', 'setting', 'endpoint', 'origin', 'subgroup')):
                raise ReviewNeeded('Required source population/arm/endpoint/origin/subgroup is unknown; never filled from target')
            if doc.registered_scope is not None and scope != doc.registered_scope:
                raise ValueError('Passage scope conflicts with independently registered source context')
            if scope.endpoint != 'OS' or stat.endpoint not in (None, 'OS') or scope.subgroup not in ('all', 'sex=male', 'sex=female'):
                status = 'unsupported'; raise ReviewNeeded('Current backend supports OS and declared sex subgroups only')
            key = (doc.id, s.start, s.end)
            if key in seen:
                status = 'duplicate'; raise ReviewNeeded('Repeated extraction of the same passage is not new evidence')
            seen.add(key)
            metric = stat.metric
            conditions = [] if scope.subgroup == 'all' else [{'variable': 'sex', 'values': [scope.subgroup.split('=')[1]]}]
            summary = {'metric': metric, 'variables': ['age'] if metric == 'mean' else ['sex'] if metric == 'proportion' else ['os_time', 'os_event'],
                       'conditions': conditions, 'unit': stat.unit, 'category': stat.category,
                       'at_time': stat.at_time, 'endpoint': stat.endpoint,
                       'time_origin': scope.origin if stat.endpoint else None}
            uncertainty = None
            if stat.ci_lower is not None or stat.standard_error is not None:
                uncertainty = {'interval': {'lower': stat.ci_lower, 'upper': stat.ci_upper} if stat.ci_lower is not None else None,
                               'confidence_level': stat.confidence_level, 'standard_error': stat.standard_error}
            eid = evidence_id(doc, s)
            record = Evidence(id=eid, population={'disease': scope.disease, 'stage': scope.stage,
                        'geography': scope.geography, 'line_of_therapy': scope.line_of_therapy},
                treatment={'setting': scope.setting, 'regimen': scope.regimen},
                # Baseline units describe age (or no dimension for sex), not OS.
                # Preserve reported endpoint/origin in the scope sidecar without
                # manufacturing the time unit required by a core Endpoint object.
                endpoint={'name': scope.endpoint, 'time_origin': scope.origin, 'unit': stat.unit} if stat.endpoint else None,
                summary=summary, estimate=stat.value, estimate_status=stat.estimate_status,
                estimate_reason='Source explicitly reports ' + stat.estimate_status if stat.estimate_status != 'available' else None,
                estimate_bound={'lower': stat.lower_bound} if stat.lower_bound is not None else None,
                uncertainty=uncertainty, sample_size=stat.sample_size,
                provenance={'source_type': 'literature', 'reference': doc.title, 'location': f'characters {s.start}:{s.end}',
                            'nature': doc.nature, 'uri': doc.uri, 'content_hash': doc.content_hash,
                            'extraction_method': NORMALIZER, 'notes': 'Grounded labelled passage; no clinical applicability endorsement'},
                extensions={'literature': {'document_id': doc.id, 'source_key': doc.source_key, 'doi': doc.doi,
                            'scope': scope.model_dump(), 'support': s.model_dump(), 'normalizer': NORMALIZER}})
            if doc.nature != 'demo' or doc.registered_scope is None:
                status = 'needs_review'; reasons.append('Grounded real/unregistered source requires separate adoption review')
                # Retain the validated object but never auto-adopt it into the demo.
            else:
                status = 'accepted'
            batch.validated.append(record)
        except ReviewNeeded as exc:
            status = status if status in ('unsupported', 'duplicate') else 'needs_review'
            reasons.append(str(exc)); eid = None
        except ValueError as exc:
            status = 'rejected'; reasons.append(str(exc)); eid = None
        batch.validation.append(CandidateResult(index=index, status=status, reasons=reasons, evidence_id=eid))
    batch.conflicts = find_conflicts(batch.validated)
    return batch


def find_conflicts(evidence):
    groups = {}
    for item in evidence:
        summary = item.summary.model_dump(mode='json')
        factor = 1.0
        if summary['unit'] in ('year', 'month'):
            canonical = 'year' if summary['metric'] == 'mean' else 'month'
            factor = 1 if summary['unit'] == canonical else 12 if canonical == 'month' else 1/12
            summary['unit'] = canonical
            if summary['at_time'] is not None:
                summary['at_time'] *= factor
                factor = 1  # Survival probability itself does not change.
        key = digest([item.population.model_dump(), item.treatment.model_dump(), summary])
        groups.setdefault(key, []).append((item.id, (item.estimate * factor if item.estimate is not None else None, item.estimate_status)))
    return [[eid for eid, _ in items] for items in groups.values() if len({v for _, v in items}) > 1]
