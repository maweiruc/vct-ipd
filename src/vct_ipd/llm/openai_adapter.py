"""Optional Responses API adapter. No SDK import or network activity on import."""

import importlib.metadata
import json
import os
from .providers import ProviderError
from vct_ipd.planners.contracts import Draft
from vct_ipd.planners.profile import INSTRUCTIONS


class OpenAIProvider:
    schema = Draft
    instructions = INSTRUCTIONS
    schema_name = 'planning_draft'
    def __init__(self, model, *, client=None):
        self.model, self.client, self.calls = model, client, 0

    def metadata(self):
        try:
            sdk = importlib.metadata.version('openai')
        except importlib.metadata.PackageNotFoundError:
            sdk = None
        return {'provider': 'openai', 'model': self.model, 'sdk_version': sdk,
                'nature': 'mocked_transport' if self.client is not None else 'live_adapter',
                'calls': self.calls, 'settings': {'timeout_seconds': 30, 'max_retries': 0,
                                                'max_output_tokens': 4096, 'store': False}}

    def extract(self, text, context):
        if not self.model:
            raise ProviderError('missing_model', 'Set VCT_IPD_MODEL or --model explicitly')
        if not os.environ.get('OPENAI_API_KEY'):
            raise ProviderError('missing_credentials', 'OPENAI_API_KEY is required for live planning')
        try:
            if self.client is None:
                from openai import OpenAI
                client = OpenAI(api_key=os.environ['OPENAI_API_KEY'], timeout=30, max_retries=0,
                                base_url='https://api.openai.com/v1')
            else:
                client = self.client
            self.calls += 1
            response = client.responses.create(model=self.model, store=False, max_output_tokens=4096,
                input=[{'role': 'system', 'content': self.instructions + '\nApproved context:\n' + json.dumps(context)},
                       {'role': 'user', 'content': text}],
                text={'format': {'type': 'json_schema', 'name': self.schema_name,
                                 'strict': True, 'schema': self.schema.model_json_schema()}})
        except ImportError as exc:
            raise ProviderError('missing_sdk', 'Install the optional vct-ipd[openai] extra') from exc
        except Exception as exc:
            code = 'timeout' if 'timeout' in type(exc).__name__.lower() else 'transport_error'
            raise ProviderError(code, 'OpenAI request failed; no retry or mock fallback') from exc
        if response.status != 'completed':
            raise ProviderError('incomplete', 'OpenAI response did not complete')
        for output in response.output:
            if getattr(output, 'type', None) == 'message':
                for item in output.content:
                    if getattr(item, 'type', None) == 'refusal':
                        raise ProviderError('refusal', 'Provider refused extraction')
        raw = response.output_text
        if not raw:
            raise ProviderError('empty_response', 'No structured output returned')
        return raw  # Shared boundary enforces byte limit, JSON and strict domain schema.
