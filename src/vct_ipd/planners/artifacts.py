"""Non-overwriting planning packages and deterministic saved-plan revalidation."""

import json
import re
from pathlib import Path
from .contracts import PlanningResult
from .profile import profile, context, digest, input_manifest
from .planner import resolve
from .bridge import assemble
from vct_ipd.pipeline.runner import run
from vct_ipd.reporting.writer import write_run


def revalidate(result, directory):
    result = PlanningResult.model_validate_json(result.model_dump_json())
    p = profile(result.request.profile)
    manifest = input_manifest(p, directory)
    if result.profile_hash != digest(p) or result.context_hash != digest(context(p)) or result.input_manifest != manifest:
        raise ValueError('Stale profile, capability, extraction schema or bound input manifest')
    if result.status != 'READY' or result.draft is None or result.issues:
        raise ValueError('Only issue-free READY plans can execute/replay')
    checked = resolve(result.request, result.draft, p, manifest, result.provider_metadata)
    if checked.status != 'READY' or checked.model_dump(mode='json') != result.model_dump(mode='json'):
        raise ValueError('Saved plan differs from deterministic reconstruction; replan explicitly')
    return checked, p


def save_planning(result, root):
    folder = Path(root) / 'planning'
    folder.mkdir(parents=True, exist_ok=False)
    payloads = {'draft.json': result.draft.model_dump(mode='json') if result.draft else None,
                'result.json': result.model_dump(mode='json'),
                'provenance.json': [v.model_dump(mode='json') for v in result.provenance],
                'provider_metadata.json': result.provider_metadata, 'input_manifest.json': result.input_manifest}
    (folder / 'request.txt').write_text(result.request.text)
    hashes = {'request.txt': digest(result.request.text)}
    for name, value in payloads.items():
        (folder / name).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
        hashes[name] = digest(value)
    (folder / 'integrity.json').write_text(json.dumps(hashes, indent=2) + '\n')


def load_plan(root, directory):
    folder = Path(root) / 'planning'
    expected_names = {'request.txt', 'draft.json', 'result.json', 'provenance.json', 'provider_metadata.json', 'input_manifest.json'}
    # Bounds before parsing; no manifest-provided paths are followed.
    for name in expected_names | {'integrity.json'}:
        if (folder / name).stat().st_size > 250000:
            raise ValueError('Oversized planning artifact')
    hashes = json.loads((folder / 'integrity.json').read_text())
    if (not isinstance(hashes, dict) or set(hashes) != expected_names
            or any(not isinstance(v, str) or not re.fullmatch(r'[0-9a-f]{64}', v) for v in hashes.values())):
        raise ValueError('Unexpected planning integrity manifest')
    for name in expected_names:
        text = (folder / name).read_text()
        value = text if name == 'request.txt' else json.loads(text)
        if digest(value) != hashes[name]:
            raise ValueError('Changed planning artifact: ' + name)
    result = PlanningResult.model_validate_json((folder / 'result.json').read_text())
    # Sidecars must also agree with the canonical validated result.
    expected = {'request.txt': result.request.text, 'draft.json': result.draft.model_dump(mode='json') if result.draft else None,
                'provenance.json': [v.model_dump(mode='json') for v in result.provenance],
                'provider_metadata.json': result.provider_metadata, 'input_manifest.json': result.input_manifest}
    if any(hashes[name] != digest(value) for name, value in expected.items()):
        raise ValueError('Planning sidecars disagree with result')
    return revalidate(result, directory)[0]


def deliver(result, directory, destination, *, execute=False):
    root = Path(destination)
    if root.exists():
        raise FileExistsError('Choose a new output directory')
    outcome = None
    if execute and result.status == 'READY':
        checked, p = revalidate(result, directory)
        inputs = assemble(checked.task, checked.data_schema, checked.execution, p, checked.input_manifest)
        outcome = run(*inputs)
        write_run(outcome, root)  # Writer owns directory creation; planning follows.
    else:
        root.mkdir(parents=True, exist_ok=False)
    save_planning(result, root)
    if result.status == 'ERROR':
        return 3
    if result.status != 'READY':
        return 2
    return 0 if outcome is None or outcome.success else 4
