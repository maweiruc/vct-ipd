"""Natural-language planning CLI. All network use requires --provider openai."""

import argparse
import json
import os
from pathlib import Path
from vct_ipd.llm.providers import ReplayProvider, ProviderError
from vct_ipd.llm.openai_adapter import OpenAIProvider
from vct_ipd.planners.contracts import PlanningRequest
from vct_ipd.planners.profile import profile
from vct_ipd.planners.planner import plan, error_result
from vct_ipd.planners.artifacts import deliver, load_plan


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--prompt')
    source.add_argument('--prompt-file', type=Path)
    source.add_argument('--saved-plan', type=Path)
    parser.add_argument('--profile', required=True, choices=['hypothetical-nsclc'])
    parser.add_argument('--inputs', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--provider', choices=['replay', 'openai'], default='replay')
    parser.add_argument('--recording', type=Path)
    parser.add_argument('--model', default=os.environ.get('VCT_IPD_MODEL'))
    parser.add_argument('--run', action='store_true')
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error('output already exists; choose a new directory')
    try:
        if args.saved_plan:
            result = load_plan(args.saved_plan, args.inputs)
            if result.request.profile != args.profile:
                raise ValueError('Saved profile differs from selected profile')
        else:
            if args.prompt_file and args.prompt_file.stat().st_size > 32000:
                raise ValueError('Prompt file exceeds byte limit')
            text = args.prompt_file.read_text() if args.prompt_file else args.prompt
            request = PlanningRequest(text=text, profile=args.profile)
            try:
                if args.provider == 'openai':
                    provider = OpenAIProvider(args.model)
                elif args.recording:
                    provider = ReplayProvider(args.recording)
                else:
                    raise ProviderError('missing_recording', 'Offline replay requires --recording; it is not a text parser')
                result = plan(request, provider, args.inputs)
            except (ProviderError, ValueError, OSError, KeyError) as exc:
                result = error_result(request, profile(request.profile), 'provider_setup', str(exc))
        code = deliver(result, args.inputs, args.output, execute=args.run)
        print(json.dumps({'status': result.status, 'normalized': result.normalized,
                          'issues': [i.model_dump() for i in result.issues],
                          'defaults': [p.field for p in result.provenance if p.source == 'profile'],
                          'warnings': result.warnings, 'exit_code': code}, ensure_ascii=False, indent=2))
        return code
    except (ValueError, OSError) as exc:
        # Setup/replay failures still leave an auditable package, never a cohort.
        args.output.mkdir(parents=True, exist_ok=False)
        folder = args.output / 'planning'
        folder.mkdir()
        (folder / 'error.json').write_text(json.dumps({'status': 'ERROR', 'code': 'setup_or_replay_error',
                                                      'message': str(exc)}, indent=2))
        print('ERROR: ' + str(exc))
        return 3


if __name__ == '__main__':
    raise SystemExit(main())
