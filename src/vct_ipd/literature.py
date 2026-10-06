"""M4 collection CLI. No retrieval/generation unless explicitly selected."""
import argparse
import json
import os
from pathlib import Path
from vct_ipd.evidence.literature.contracts import EvidenceRequest
from vct_ipd.evidence.literature.retrieval import LocalDocuments, EuropePMC, ReplaySearch, bounded_json
from vct_ipd.evidence.literature.extraction import ReplayExtractor, LiveExtractor
from vct_ipd.evidence.literature.service import collect
from vct_ipd.evidence.literature.adoption import adopt_demo
from vct_ipd.evidence.literature.storage import save_collection
from vct_ipd.llm.providers import ProviderError
from vct_ipd.pipeline.runner import run
from vct_ipd.reporting.writer import write_run


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--local-manifest', type=Path)
    source.add_argument('--europe-pmc', action='store_true')
    source.add_argument('--search-recording', type=Path)
    extraction = parser.add_mutually_exclusive_group()
    extraction.add_argument('--recording', type=Path)
    extraction.add_argument('--live-extraction', action='store_true')
    parser.add_argument('--model', default=os.environ.get('VCT_IPD_MODEL'))
    parser.add_argument('--cache-public-text', action='store_true', help='Explicitly permit local abstract caching under source rights')
    parser.add_argument('--run-demo', action='store_true')
    parser.add_argument('--adoption', type=Path)
    parser.add_argument('--inputs', type=Path)
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error('Choose a new output directory')
    collection = None
    try:
        request = EvidenceRequest.model_validate(bounded_json(args.request))
        retriever = (LocalDocuments(args.local_manifest) if args.local_manifest else ReplaySearch(args.search_recording)
                     if args.search_recording else EuropePMC(cache_text=args.cache_public_text))
        extractor = ReplayExtractor(args.recording) if args.recording else LiveExtractor(args.model) if args.live_extraction else None
        collection = collect(request, retriever, extractor)
        if args.run_demo:
            if not args.adoption or not args.inputs:
                raise ValueError('Demo execution requires explicit --adoption and --inputs')
            inputs, adoption = adopt_demo(collection, args.adoption, args.inputs)
            result = run(*inputs)
            write_run(result, args.output)
            save_collection(collection, args.output, adoption=adoption, run_written=True)
            print(json.dumps({'status': collection.status, 'stop': result.stop_reason,
                              'validation': result.validation_report.status, 'success': result.success}))
            return 0 if result.success else 4
        save_collection(collection, args.output)
        print(json.dumps({'status': collection.status, 'validated': len(collection.batch.validated)}))
        return 0 if collection.status in ('validated_evidence', 'retrieved_only') else 3 if collection.status in (
            'extraction_error', 'malformed_response', 'transport_error', 'timeout', 'server_error', 'http_error') else 2
    except (ProviderError, ValueError, OSError, TypeError, KeyError) as exc:
        if collection is not None:
            collection.status = 'adoption_blocked' if args.run_demo else 'collection_error'
            collection.diagnostics.append(str(exc))
            save_collection(collection, args.output)
        else:
            args.output.mkdir(parents=True, exist_ok=False)
            (args.output/'error.json').write_text(json.dumps({'status': 'ERROR', 'reason': str(exc)}, indent=2))
        print('ERROR: ' + str(exc))
        return 2 if collection is not None and args.run_demo else 3


if __name__ == '__main__':
    raise SystemExit(main())
