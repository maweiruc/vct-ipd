"""Explicit local selection and one bounded public adapter: Europe PMC core JSON."""
import hashlib
import json
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
import re
import urllib.error
import urllib.parse
import urllib.request
from .contracts import DocumentRecord, RetrievalResult, Scope
from vct_ipd.planners.profile import digest

ENDPOINT = 'https://www.ebi.ac.uk/europepmc/webservices/rest/search'


class LocalManifestError(ValueError):
    """Malformed local fields; messages describe the contract, not source values."""


def sha(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def normalize(text):
    return text.replace('\r\n', '\n').replace('\r', '\n')


def now():
    return datetime.now(timezone.utc).isoformat()


def bounded_json(path, limit=500000):
    path = Path(path)
    with path.open('rb') as stream:
        raw = stream.read(limit + 1)
    if len(raw) > limit:
        raise ValueError('Local JSON exceeds byte budget')
    return json.loads(raw)


def deduplicate(documents):
    # Group the complete bounded hit history, including discarded aliases. A
    # bridge hit can join two earlier groups. Empty text is never an identity.
    groups = []
    for doc in documents:
        keys = {doc.source_key} | ({'doi:' + doc.doi.casefold()} if doc.doi else set())
        if doc.text:
            keys.add('text:' + doc.content_hash)
        overlaps = [g for g in groups if g[0] & keys]
        if overlaps:
            group = overlaps[0]
            group[0].update(keys); group[1].append(doc)
            for other in overlaps[1:]:
                group[0].update(other[0]); group[1].extend(other[1]); groups.remove(other)
        else:
            groups.append([keys, [doc]])
    kept, notes = [], []
    for keys, members in groups:
        members.sort(key=documents.index)
        # Whole-record selection: text wins, otherwise first encounter wins.
        representative = next((d for d in members if d.text), members[0])
        kept.append(representative)
        if len(members) > 1:
            notes.extend('duplicate_source:' + d.id for d in members if d is not representative)
            notes.append('identity_group:' + json.dumps({
                'representative': representative.id,
                'aliases': sorted(k for k in keys if not k.startswith('text:')),
                'variants': [d.model_dump(exclude={'text'}) for d in members],
                'different_nonempty_text_versions': len({d.content_hash for d in members if d.text}) > 1,
            }, sort_keys=True))
    return kept, notes


class LocalDocuments:
    def __init__(self, manifest):
        self.path = Path(manifest)

    def search(self, request):
        if 'local' not in request.allowed_sources:
            raise ValueError('Local sources not allowed')
        config = bounded_json(self.path)
        if not isinstance(config, dict) or set(config) != {'version', 'documents'} or config['version'] != 'local-documents-v1' or not isinstance(config['documents'], list):
            raise ValueError('Invalid local document manifest')
        documents, used = [], 0
        entries = config['documents'][:min(request.max_documents, request.max_results)]
        for entry in entries:
            required = {'id', 'file', 'sha256', 'title', 'nature', 'rights', 'registered_scope'}
            if not isinstance(entry, dict) or set(entry) != required:
                raise ValueError('Invalid local source identity')
            if not isinstance(entry['id'], str) or not re.fullmatch('[a-z0-9-]{1,50}', entry['id']):
                raise LocalManifestError('Invalid local source id: expected 1-50 lowercase letters, digits or hyphens')
            if not isinstance(entry['file'], str) or not entry['file']:
                raise LocalManifestError('Invalid local source file: expected a nonempty text path')
            path = (self.path.parent / entry['file']).resolve()
            if not path.is_relative_to(self.path.parent.resolve()) or path.suffix not in ('.txt', '.md'):
                raise ValueError('Local files must be explicitly bound text files inside the manifest directory')
            with path.open('rb') as stream:
                raw = stream.read(request.max_bytes - used + 1)
            used += len(raw)
            if used > request.max_bytes:
                raise ValueError('Document byte budget exhausted')
            if hashlib.sha256(raw).hexdigest() != entry['sha256']:
                raise ValueError('Local document hash changed')
            text = normalize(raw.decode('utf-8'))
            documents.append(DocumentRecord(id=entry['id'], source='local', source_key='local:' + entry['id'],
                title=entry['title'], uri=str(path), nature=entry['nature'], access_scope='local_text',
                retrieved_at=now(), text=text, content_hash=sha(text), raw_hash=entry['sha256'],
                rights=entry['rights'], cache_text=True,
                registered_scope=Scope.model_validate(entry['registered_scope']) if entry['registered_scope'] else None))
        documents, diagnostics = deduplicate(documents)
        if len(entries) < len(config['documents']):
            diagnostics.append('document_limit')
        return RetrievalResult(status='documents' if documents else 'zero_hits', documents=documents,
            searches=[{'selection': 'explicit_local_manifest', 'manifest_hash': digest(config),
                       'request_hash': digest(request)}], diagnostics=diagnostics, mode='local', calls=0)


class ReplaySearch:
    """Recorded retrieval with exact request and document snapshot bindings."""
    def __init__(self, path):
        self.record = bounded_json(path)
        if (not isinstance(self.record, dict) or set(self.record) != {'nature', 'request_hash', 'response_hash', 'response'}
                or self.record['nature'] not in ('human_authored_mock', 'captured_public')):
            raise ValueError('Invalid search recording envelope')

    def search(self, request):
        record = self.record
        if record['request_hash'] != digest(request) or record['response_hash'] != digest(record['response']):
            raise ValueError('Stale search recording or changed document snapshot')
        result = RetrievalResult.model_validate(record['response'])
        if (len(result.documents) > min(request.max_results, request.max_documents)
                or sum(len(d.text.encode()) for d in result.documents) > request.max_bytes):
            raise ValueError('Recorded documents exceed request budget')
        for doc in result.documents:
            if doc.source not in request.allowed_sources or sha(doc.text) != doc.content_hash:
                raise ValueError('Recorded source not allowed or text hash mismatch')
        result.mode = 'replay:' + record['nature']
        result.calls = 0
        result.documents, duplicates = deduplicate(result.documents)
        result.diagnostics.extend(duplicates)
        return result


class TextOnly(HTMLParser):
    def __init__(self):
        super().__init__(); self.parts = []
    def handle_starttag(self, tag, attrs):
        if tag in ('p', 'br', 'div', 'h2', 'h3'):
            self.parts.append('\n')
    def handle_endtag(self, tag):
        if tag in ('p', 'div'):
            self.parts.append('\n')
    def handle_data(self, data):
        self.parts.append(data)


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None  # Never follow response-controlled URLs off the allowlist.


def public_get(params, limit, timeout):
    url = ENDPOINT + '?' + urllib.parse.urlencode(params)
    request = urllib.request.Request(url, headers={'Accept': 'application/json', 'User-Agent': 'VCT-IPD-M4/0.1'})
    try:
        with urllib.request.build_opener(NoRedirect).open(request, timeout=timeout) as response:
            return response.status, response.read(limit + 1)
    except urllib.error.HTTPError as exc:
        return exc.code, b''


class EuropePMC:
    def __init__(self, *, transport=None, cache_text=False):
        self.transport = transport or public_get
        self.mode = 'mocked_public_transport' if transport else 'live_public'
        self.cache_text = cache_text

    def search(self, request):
        if 'europe_pmc' not in request.allowed_sources or not request.queries:
            raise ValueError('Europe PMC requires explicit allowed source and query')
        result = RetrievalResult(status='zero_hits', mode=self.mode)
        used, documents, all_hits = 0, [], []
        for query in request.queries:
            if not query.strip() or len(query) > 1000:
                raise ValueError('Query is empty or too long')
            cursor = '*'
            for page in range(request.max_pages):
                # Raw hits, including variants, consume the document allowance.
                remaining = min(request.max_results, request.max_documents) - len(all_hits)
                if remaining <= 0:
                    result.diagnostics.append('document_limit'); break
                params = {'query': query, 'format': 'json', 'resultType': 'core', 'pageSize': remaining, 'cursorMark': cursor}
                result.calls += 1
                log = {'parameters': params, 'endpoint': ENDPOINT, 'timeout_seconds': 15, 'retries': 0}
                result.searches.append(log)
                try:
                    status, raw = self.transport(params, request.max_bytes - used, 15)
                    log['http_status'] = status
                    if status != 200:
                        result.status = {404: 'not_found', 429: 'rate_limited'}.get(status, 'server_error' if status >= 500 else 'http_error')
                        return result
                    used += len(raw)
                    if used > request.max_bytes:
                        result.status = 'byte_limit'; return result
                    data = json.loads(raw)
                    if (not isinstance(data, dict) or type(data.get('hitCount')) is not int
                            or not isinstance(data.get('resultList'), dict)
                            or not isinstance(data['resultList'].get('result'), list)):
                        raise ValueError('Malformed Europe PMC core envelope')
                    hits = data['resultList']['result']
                    if len(hits) > remaining:
                        raise ValueError('Service exceeded requested page size')
                    log.update(response_hash=hashlib.sha256(raw).hexdigest(), hit_count=data['hitCount'])
                    for hit in hits:
                        if not isinstance(hit, dict) or any(not isinstance(hit.get(k), str) or not hit[k] for k in ('id', 'source', 'title')):
                            raise ValueError('Missing retrieved identity/title')
                        if not re.fullmatch(r'[A-Za-z0-9_-]{1,60}', hit['id']) or not re.fullmatch('[A-Z]{2,8}', hit['source']):
                            raise ValueError('Invalid source identifier')
                        for name in ('abstractText', 'doi', 'pubYear'):
                            if hit.get(name) is not None and not isinstance(hit[name], str):
                                raise ValueError('Invalid core metadata type')
                        abstract = hit.get('abstractText') or ''
                        parser = TextOnly(); parser.feed(abstract)
                        text = normalize(''.join(parser.parts)).strip()
                        source_key = hit['source'] + ':' + hit['id']
                        all_hits.append(DocumentRecord(id='epmc-' + sha(source_key)[:20], source='europe_pmc',
                            source_key=source_key, title=hit['title'], doi=hit.get('doi'), year=hit.get('pubYear'),
                            uri='https://europepmc.org/article/' + hit['source'] + '/' + hit['id'], nature='observed',
                            access_scope='abstract' if text else 'metadata', retrieved_at=now(), text=text,
                            content_hash=sha(text), raw_hash=sha(abstract), text_version='epmc-html-text-v1',
                            rights='Source rights apply; public abstract availability is not a redistribution license.',
                            cache_text=self.cache_text))
                    documents, duplicates = deduplicate(all_hits)
                    result.documents = documents
                    result.diagnostics = [n for n in result.diagnostics if not n.startswith(('duplicate_source:', 'identity_group:'))] + duplicates
                    next_cursor = data.get('nextCursorMark')
                    if next_cursor is not None and not isinstance(next_cursor, str):
                        raise ValueError('Invalid pagination cursor')
                    if not hits or not next_cursor or next_cursor == cursor:
                        break
                    cursor = next_cursor
                    if page + 1 == request.max_pages:
                        result.diagnostics.append('pagination_limit')
                except (TimeoutError, urllib.error.URLError) as exc:
                    result.status = 'timeout' if isinstance(exc, TimeoutError) or isinstance(getattr(exc, 'reason', None), TimeoutError) else 'transport_error'
                    return result
                except OSError:
                    result.status = 'transport_error'; return result
                except (ValueError, UnicodeError, TypeError):
                    result.status = 'malformed_response'; return result
        result.documents = documents
        result.status = 'documents' if any(d.text for d in documents) else 'unavailable_text' if documents else 'zero_hits'
        return result
