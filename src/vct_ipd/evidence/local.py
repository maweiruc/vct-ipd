"""Offline summary and reference CSV providers; no retrieval or synthesis."""

import csv
import hashlib
import json
from pathlib import Path
from typing import Protocol

from vct_ipd.core import Evidence, EvidenceSet, Provenance
from vct_ipd.evaluation.metrics import measure, selected, validity_errors


class EvidenceProvider(Protocol):
    def collect(self, task, schema) -> EvidenceSet: ...


class SummaryProvider:
    def __init__(self, path, *, set_id="summary-evidence"):
        self.path, self.set_id = Path(path), set_id

    def collect(self, task, schema):
        content = self.path.read_bytes()
        result = EvidenceSet(id=self.set_id, task=task.ref, schema_ref=schema.ref,
                             items=[Evidence.model_validate(item) for item in json.loads(content)],
                             extensions={"local_input": {"uri": str(self.path.resolve()),
                                                         "sha256": hashlib.sha256(content).hexdigest()}})
        result.validate_references(schema)
        return result


def parse_cell(text, variable):
    if text == "":
        return None
    if variable.type == "boolean":
        if text not in ("True", "False", "true", "false", "0", "1"):
            raise ValueError(f"invalid Boolean CSV cell: {text}")
        return text in ("True", "true", "1")
    if variable.type == "continuous":
        return float(text)
    if variable.type == "integer":
        return int(text)
    if variable.allowed_values:
        matches = [v for v in variable.allowed_values if str(v) == text]
        if len(matches) != 1:
            raise ValueError(f"ambiguous or invalid category CSV cell: {text}")
        return matches[0]
    return text


class IPDProvider:
    def __init__(self, path, *, columns, summaries, population, treatment, endpoint,
                 nature, set_id="ipd-evidence"):
        self.path = Path(path)
        self.columns, self.summaries = dict(columns), dict(summaries)
        self.population, self.treatment, self.endpoint = population, treatment, endpoint
        self.nature, self.set_id = nature, set_id

    def collect(self, task, schema):
        content = self.path.read_bytes()
        if set(self.columns) != {v.name for v in schema.variables} or len(set(self.columns.values())) != len(self.columns):
            raise ValueError("CSV mapping must bind each schema column to a distinct header")
        reader = csv.DictReader(content.decode("utf-8").splitlines())
        if reader.fieldnames is None or len(set(reader.fieldnames)) != len(reader.fieldnames) or not set(self.columns.values()) <= set(reader.fieldnames):
            raise ValueError("missing or duplicate CSV headers")
        rows = [{v.name: parse_cell(row[self.columns[v.name]], v) for v in schema.variables} for row in reader]
        if not rows or validity_errors(rows, schema):
            raise ValueError("reference CSV is empty or violates its declared schema")
        items = []
        for evidence_id, summary in self.summaries.items():
            value = measure(rows, summary, schema)
            n = len(selected(rows, summary))
            items.append(Evidence(
                id=evidence_id, population=self.population, treatment=self.treatment,
                endpoint=self.endpoint if summary.endpoint else None, summary=summary,
                sample_size=n or None, **value.fields("estimate"),
                provenance=Provenance(source_type="reference_ipd", reference=self.path.name,
                                      location=f"columns {self.columns}; metric {summary.metric}",
                                      nature=self.nature, uri=str(self.path.resolve()),
                                      content_hash=hashlib.sha256(content).hexdigest(),
                                      extraction_method="vct-ipd-local-csv-v1")))
        result = EvidenceSet(id=self.set_id, task=task.ref, schema_ref=schema.ref, items=items)
        result.validate_references(schema)
        return result


def combine(task, schema, sets, *, set_id="combined-evidence"):
    for item in sets:
        item.validate_references(schema)
    result = EvidenceSet(id=set_id, task=task.ref, schema_ref=schema.ref,
                         items=[Evidence.model_validate_json(e.model_dump_json()) for s in sets for e in s.items],
                         extensions={"source_sets": [s.model_dump(mode="json") for s in sets]})
    result.validate_references(schema)
    return result
