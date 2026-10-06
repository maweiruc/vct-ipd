"""Bounded extraction transport. Mock/replay never calls a network."""

import json
import re
from pathlib import Path
from typing import Protocol
from pydantic import ValidationError, TypeAdapter, JsonValue
from vct_ipd.planners.contracts import Draft
from vct_ipd.planners.profile import digest


class ProviderError(Exception):
    def __init__(self, code, message):
        self.code = code
        super().__init__(message)


class ExtractionProvider(Protocol):
    def extract(self, text: str, context: dict) -> str: ...
    def metadata(self) -> dict: ...


def safe_metadata(provider):
    """Metadata must not mask an extraction failure with a second exception."""
    try:
        return TypeAdapter(dict[str, JsonValue]).validate_python(provider.metadata(), strict=True)
    except Exception:
        # Deliberately confined to ancillary metadata, not extraction or readiness.
        return {'provider': type(provider).__name__, 'metadata_error': True}


def parse_draft(raw, limit=32000):
    if not isinstance(raw, str) or len(raw.encode()) > limit:
        raise ProviderError('response_size', 'Provider output is not bounded text')
    try:
        draft = Draft.model_validate_json(raw)
    except (ValidationError, ValueError) as exc:
        raise ProviderError('schema_failure', 'Malformed JSON or invalid extraction schema') from exc
    if len(draft.claims) + len(draft.additional_requirements) + len(draft.ambiguities) > 100:
        raise ProviderError('response_size', 'Too many extracted fields')
    return draft


class ScriptedProvider:
    """Test double, not a language parser. No fallback after an error."""
    def __init__(self, response):
        self.response, self.calls = response, 0

    def metadata(self):
        return {'provider': 'scripted', 'nature': 'human_authored_mock', 'calls': self.calls}

    def extract(self, text, context):
        self.calls += 1
        if isinstance(self.response, Exception):
            raise self.response
        return self.response


class ReplayProvider:
    def __init__(self, path):
        path = Path(path)
        if path.stat().st_size > 64000:
            raise ProviderError('response_size', 'Oversized recording')
        try:
            record = json.loads(path.read_text())
            required = {'nature', 'request_hash', 'context_hash', 'draft'}
            if not isinstance(record, dict) or set(record) != required:
                raise ValueError('Expected the recording object with its four required fields')
            if record['nature'] not in ('human_authored_mock', 'captured_live'):
                raise ValueError('Invalid recording origin')
            if any(not isinstance(record[k], str) or not re.fullmatch(r'[0-9a-f]{64}', record[k])
                   for k in ('request_hash', 'context_hash')):
                raise ValueError('Invalid recording digest')
            parse_draft(json.dumps(record['draft'], ensure_ascii=False))
        except (ValueError, TypeError, ProviderError) as exc:
            raise ProviderError('recording_invalid', 'Invalid recording envelope or draft') from exc
        self.recording = record
        self.calls = 0

    def metadata(self):
        return {'provider': 'replay', 'nature': self.recording['nature'],
                'recording_hash': digest(self.recording), 'calls': self.calls}

    def extract(self, text, context):
        self.calls += 1
        record = self.recording
        if record.get('nature') not in ('human_authored_mock', 'captured_live'):
            raise ProviderError('recording_invalid', 'Recording must declare its origin')
        if record.get('request_hash') != digest(text) or record.get('context_hash') != digest(context):
            raise ProviderError('recording_mismatch', 'Recording is bound to another request or context')
        return json.dumps(record['draft'], ensure_ascii=False)
