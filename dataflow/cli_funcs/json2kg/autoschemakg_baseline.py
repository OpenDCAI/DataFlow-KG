from __future__ import annotations

import ast
import csv
import json
import os
import sys
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from openai import OpenAI


ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from atlas_rag.kg_construction.triple_config import ProcessingConfig
from atlas_rag.kg_construction.triple_extraction import KnowledgeGraphExtractor
from atlas_rag.llm_generator import LLMGenerator


Triple = tuple[str, str, str]


class AutoSchemaKGBaseline:
    def add_args(self, parser):
        parser.add_argument("--input", required=True, type=Path)
        parser.add_argument("--output", type=Path, default=ROOT.parent / "result" / "autoschemakg_output.json")
        parser.add_argument("--api-key", default=os.getenv("OPENAI_API_KEY"))
        parser.add_argument("--api-url", default=os.getenv("OPENAI_BASE_URL"))
        parser.add_argument("--model", default=os.getenv("MODEL_NAME", "gpt-4o-mini"))
        parser.add_argument("--work-dir", type=Path, default=None)
        parser.add_argument("--batch-size-triple", type=int, default=8)
        parser.add_argument("--batch-size-concept", type=int, default=32)
        parser.add_argument("--max-workers", type=int, default=8)
        parser.add_argument("--max-new-tokens", type=int, default=2048)
        parser.add_argument("--chunk-size", type=int, default=8192)
        parser.add_argument("--chunk-overlap", type=int, default=0)
        parser.add_argument("--no-concept", action="store_true")
        return parser

    def main(self, args) -> Path:
        if not args.api_key:
            raise ValueError("Pass --api-key or set OPENAI_API_KEY.")
        if not args.api_url:
            raise ValueError("Pass --api-url or set OPENAI_BASE_URL.")

        rows = load_jsonl(args.input)
        if args.work_dir is None:
            temp_parent = ROOT / ".tmp"
            temp_parent.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryDirectory(prefix="run_", dir=temp_parent) as work_dir:
                return self._run(args, rows, Path(work_dir))
        return self._run(args, rows, args.work_dir)

    def _run(self, args, rows: list[dict[str, Any]], work_dir: Path) -> Path:
        cfg = ProcessingConfig(
            model_path=args.model,
            data_directory=str(work_dir),
            filename_pattern="input",
            output_directory=str(work_dir / "autoschema_output"),
            batch_size_triple=args.batch_size_triple,
            batch_size_concept=args.batch_size_concept,
            max_workers=args.max_workers,
            max_new_tokens=args.max_new_tokens,
            remove_doc_spaces=True,
            include_concept=not args.no_concept,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
        )

        client = OpenAI(api_key=args.api_key, base_url=args.api_url)
        generator = LLMGenerator(client=client, model_name=args.model, max_workers=args.max_workers)
        generator.config.temperature = 0.0

        extractor = InMemoryKnowledgeGraphExtractor(model=generator, config=cfg, rows=rows)
        extractor.run_extraction()
        extractor.convert_json_to_csv()
        if cfg.include_concept:
            extractor.generate_concept_csv_temp(batch_size=args.batch_size_concept)
            extractor.create_concept_csv()
        extractor.convert_to_graphml()

        triples_by_id = load_triples(latest_extraction_file(Path(cfg.output_directory), cfg.filename_pattern))
        if cfg.include_concept:
            triples_by_id = add_concept_triples(
                triples_by_id,
                Path(cfg.output_directory) / "concept_csv",
                cfg.filename_pattern,
            )

        write_json(args.output, build_eval_rows(rows, triples_by_id))
        return args.output


class InMemoryKnowledgeGraphExtractor(KnowledgeGraphExtractor):
    def __init__(self, model, config: ProcessingConfig, rows: list[dict[str, Any]]):
        self._rows = rows
        super().__init__(model=model, config=config)

    def load_dataset(self) -> Any:
        return {"train": build_autoschema_samples(self._rows)}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if "raw_chunk" not in row:
                raise ValueError(f"{path}:{line_no} missing raw_chunk")
            rows.append(row)
    return rows


def build_autoschema_samples(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    samples = []
    for index, row in enumerate(rows):
        item_id = str(row.get("id") or f"item_{index:06d}")
        samples.append(
            {
                "id": item_id,
                "text": str(row["raw_chunk"]),
                "metadata": {
                    "lang": row.get("lang", "en"),
                    "source": row.get("source", ""),
                    "text_path": row.get("text_path", ""),
                },
            }
        )
    return samples


def latest_extraction_file(output_dir: Path, filename_pattern: str) -> Path:
    candidates = sorted(
        (output_dir / "kg_extraction").glob(f"*{filename_pattern}*.json"),
        key=lambda path: path.stat().st_mtime,
    )
    if not candidates:
        raise FileNotFoundError(f"No extraction result found under {output_dir / 'kg_extraction'}")
    return candidates[-1]


def load_triples(path: Path) -> dict[str, list[Triple]]:
    triples_by_id: dict[str, list[Triple]] = defaultdict(list)
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            triples_by_id[str(record.get("id", ""))].extend(extract_triples_from_record(record))
    return {item_id: dedupe_triples(triples) for item_id, triples in triples_by_id.items()}


def extract_triples_from_record(record: dict[str, Any]) -> list[Triple]:
    triples = []
    for key, value in record.items():
        if not key.endswith("_dict") or not isinstance(value, list):
            continue
        for item in value:
            if not isinstance(item, dict):
                continue
            relation_key = "relation" if "relation" in item else "Relation" if "Relation" in item else None
            if relation_key is None:
                continue
            relation = str(item.get(relation_key, "")).strip()
            entities = [
                str(v).strip()
                for k, v in item.items()
                if k != relation_key and isinstance(v, str) and str(v).strip()
            ]
            if relation and len(entities) >= 2:
                triples.append((entities[0], relation, entities[1]))
    return dedupe_triples(triples)


def add_concept_triples(
    triples_by_id: dict[str, list[Triple]],
    concept_dir: Path,
    filename_pattern: str,
) -> dict[str, list[Triple]]:
    concept_names = load_concept_names(concept_dir / f"concept_nodes_{filename_pattern}_from_json_with_concept.csv")
    entity_concepts = load_entity_concepts(
        concept_dir / f"concept_edges_{filename_pattern}_from_json_with_concept.csv",
        concept_names,
    )
    relation_concepts = load_relation_concepts(
        concept_dir / f"triple_edges_{filename_pattern}_from_json_with_concept.csv"
    )
    output = {}
    for item_id, triples in triples_by_id.items():
        enriched = []
        for subj, rel, obj in triples:
            enriched.append((subj, rel, obj))
            for node in (subj, obj):
                for concept in sorted(entity_concepts.get(node, ())):
                    enriched.append((node, "has_concept", concept))
            for concept in sorted(relation_concepts.get((subj, rel, obj), ())):
                enriched.append((rel, "has_concept", concept))
        output[item_id] = dedupe_triples(enriched)
    return output


def load_concept_names(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    names = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            concept_id = str(row.get("concept_id:ID", "")).strip()
            name = str(row.get("name", "")).strip()
            if concept_id and name:
                names[concept_id] = name
    return names


def load_entity_concepts(path: Path, concept_names: dict[str, str]) -> dict[str, set[str]]:
    concepts: dict[str, set[str]] = defaultdict(set)
    if not path.exists():
        return concepts
    with path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            node = str(row.get(":START_ID", "")).strip()
            name = concept_names.get(str(row.get(":END_ID", "")).strip())
            if node and name:
                concepts[node].add(name)
    return concepts


def load_relation_concepts(path: Path) -> dict[Triple, set[str]]:
    concepts: dict[Triple, set[str]] = defaultdict(set)
    if not path.exists():
        return concepts
    with path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            subj = str(row.get(":START_ID", "")).strip()
            rel = str(row.get("relation", "")).strip()
            obj = str(row.get(":END_ID", "")).strip()
            try:
                raw_concepts = ast.literal_eval(row.get("concepts", "") or "[]")
            except (SyntaxError, ValueError):
                raw_concepts = []
            if subj and rel and obj and isinstance(raw_concepts, list):
                for concept in raw_concepts:
                    if str(concept).strip():
                        concepts[(subj, rel, obj)].add(str(concept).strip())
    return concepts


def build_eval_rows(rows: list[dict[str, Any]], triples_by_id: dict[str, list[Triple]]) -> list[dict[str, Any]]:
    output = []
    for index, row in enumerate(rows):
        item_id = str(row.get("id") or f"item_{index:06d}")
        triples = triples_by_id.get(item_id, [])
        item = dict(row)
        item["id"] = item_id
        item["title"] = row.get("title") or Path(str(row.get("source", item_id))).stem
        item["text"] = row.get("raw_chunk", "")
        item["triple"] = [list(triple) for triple in triples]
        item["extracted_kg"] = triples_to_graph(triples)
        item["relational_facts"] = normalize_facts(row.get("fact", row.get("relational_facts")))
        output.append(item)
    return output


def triples_to_graph(triples: Iterable[Triple]) -> dict[str, list[Any]]:
    entities = set()
    edges = set()
    relations = []
    for subj, rel, obj in triples:
        entities.update((subj, obj))
        edges.add(rel)
        relations.append([subj, rel, obj])
    return {"entities": sorted(entities), "edges": sorted(edges), "relations": relations}


def normalize_facts(value: Any) -> dict[str, list[dict[str, Any]]]:
    if isinstance(value, dict) and any(key in value for key in ("high", "medium", "low", "unknown")):
        return {key: [normalize_fact(v) for v in value.get(key, [])] for key in ("high", "medium", "low", "unknown")}
    facts = value if isinstance(value, list) else [value] if value else []
    return {"high": [normalize_fact(v) for v in facts], "medium": [], "low": [], "unknown": []}


def normalize_fact(value: Any) -> dict[str, Any]:
    item = dict(value) if isinstance(value, dict) else {"fact": str(value)}
    item.setdefault("fact", str(value.get("fact", "")) if isinstance(value, dict) else str(value))
    item.setdefault("relational_type", "unknown")
    item.setdefault("confidence", "Unknown")
    item.setdefault("corrected_fact", None)
    item.setdefault("triples", {"relations": [], "attributes": []})
    item.setdefault("difficulty", "unknown")
    return item


def dedupe_triples(triples: Iterable[Triple]) -> list[Triple]:
    seen = set()
    output = []
    for triple in triples:
        if triple in seen:
            continue
        seen.add(triple)
        output.append(triple)
    return output


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
