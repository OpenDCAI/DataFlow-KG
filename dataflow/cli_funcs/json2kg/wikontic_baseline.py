from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import uuid
from pathlib import Path
from typing import Any, Iterable

from pymongo import MongoClient


ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

LOCAL_CONTRIEVER = Path("/home/liuxuem/Wikontic/facebook/contriever")
if not os.getenv("HF_MODEL_PATH") and LOCAL_CONTRIEVER.is_dir():
    os.environ["HF_MODEL_PATH"] = str(LOCAL_CONTRIEVER)


from wikontic.create_ontological_triplets_db import create_ontological_triplets_database
from wikontic.create_triplets_db import create_triplets_database
from wikontic.create_wikidata_ontology_db import create_wikidata_ontology_database
from wikontic.utils.dynamic_aligner import Aligner as DynamicAligner
from wikontic.utils.inference_with_db import InferenceWithDB
from wikontic.utils.openai_utils import LLMTripletExtractor
from wikontic.utils.structured_aligner import Aligner as StructuredAligner
from wikontic.utils.structured_inference_with_db import StructuredInferenceWithDB


Triple = tuple[str, str, str]


class WikonticBaseline:
    def add_args(self, parser):
        parser.add_argument("--input", required=True, type=Path)
        parser.add_argument("--output", type=Path, default=ROOT.parent / "result" / "wikontic_output.json")
        parser.add_argument("--api-key", default=os.getenv("OPENAI_API_KEY") or os.getenv("OPENROUTER_KEY"))
        parser.add_argument("--api-url", default=os.getenv("OPENAI_BASE_URL") or os.getenv("OPENROUTER_BASE_URL"))
        parser.add_argument("--model", default=os.getenv("MODEL_NAME", "gpt-4o-mini"))
        parser.add_argument("--proxy", default=os.getenv("PROXY_URL"))
        parser.add_argument("--mongo-uri", default=os.getenv("MONGO_URI", "mongodb://localhost:27018/?directConnection=true"))
        parser.add_argument("--triplets-db-name", default=os.getenv("TRIPLETS_DB", "triplets_db"))
        parser.add_argument("--ontology-db-name", default=os.getenv("ONTOLOGY_DB", "wikidata_ontology"))
        parser.add_argument("--mongo-container", default=os.getenv("MONGO_CONTAINER", "text2kg_mongo"))
        parser.add_argument("--plain-inference", action="store_true")
        parser.add_argument("--work-dir", type=Path, default=None)
        parser.add_argument("--overwrite-work-dir", action="store_true")
        parser.add_argument("--num-samples", type=int, default=None)
        return parser

    def main(self, args) -> Path:
        if not args.api_key:
            raise ValueError("Pass --api-key or set OPENAI_API_KEY.")
        if not args.api_url:
            raise ValueError("Pass --api-url or set OPENAI_BASE_URL.")

        rows = load_jsonl(args.input)
        if args.num_samples is not None:
            rows = rows[: args.num_samples]

        if args.work_dir is None:
            temp_parent = ROOT / ".tmp"
            temp_parent.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryDirectory(prefix="run_", dir=temp_parent) as work_dir:
                return self._run(args, rows, Path(work_dir))
        return self._run(args, rows, args.work_dir)

    def _run(self, args, rows: list[dict[str, Any]], work_dir: Path) -> Path:
        prepare_work_dir(work_dir, args.overwrite_work_dir)

        ensure_mongo_running(args.mongo_uri, args.mongo_container)
        client = MongoClient(args.mongo_uri)
        structured_inference = not args.plain_inference
        triplets_db_name = args.triplets_db_name or default_db_name(args.model, structured_inference)
        triplets_db = client.get_database(triplets_db_name)

        if structured_inference:
            ontology_db = client.get_database(args.ontology_db_name)
            if not has_collections(ontology_db, ("entity_types", "properties")):
                create_wikidata_ontology_database(mongo_uri=args.mongo_uri, database=args.ontology_db_name)
                ontology_db = client.get_database(args.ontology_db_name)
            if not has_collections(
                triplets_db,
                ("entity_aliases", "initial_triplets", "filtered_triplets", "ontology_filtered_triplets", "triplets"),
            ):
                create_ontological_triplets_database(mongo_uri=args.mongo_uri, db_name=triplets_db_name)
            aligner = StructuredAligner(ontology_db=ontology_db, triplets_db=triplets_db)
        else:
            if not has_collections(
                triplets_db,
                ("entity_aliases", "property_aliases", "triplets", "initial_triplets", "filtered_triplets"),
            ):
                create_triplets_database(mongo_uri=args.mongo_uri, db_name=triplets_db_name)
            aligner = DynamicAligner(triplets_db=triplets_db)

        extractor = LLMTripletExtractor(
            api_key=args.api_key,
            proxy=args.proxy,
            base_url=args.api_url,
            model=args.model,
        )
        inference = (
            StructuredInferenceWithDB(extractor=extractor, aligner=aligner, triplets_db=triplets_db)
            if structured_inference
            else InferenceWithDB(extractor=extractor, aligner=aligner, triplets_db=triplets_db)
        )

        clear_previous_sample_outputs(triplets_db, rows)
        run_extraction(rows, inference, structured=structured_inference)
        triples_by_id = load_triplets_from_mongo(triplets_db)
        write_json(args.output, build_eval_rows(rows, triples_by_id))
        return args.output


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


def prepare_work_dir(path: Path, overwrite: bool) -> None:
    if overwrite and path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def ensure_mongo_running(mongo_uri: str, container_name: str) -> None:
    if can_ping_mongo(mongo_uri):
        return

    result = subprocess.run(
        ["docker", "start", container_name],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if result.returncode != 0:
        raise RuntimeError(
            "MongoDB is not reachable and Docker container could not be started. "
            f"container={container_name}, stderr={result.stderr.strip()}"
        )

    if not can_ping_mongo(mongo_uri):
        raise RuntimeError(
            "MongoDB container was started but MongoDB is still not reachable. "
            f"uri={mongo_uri}, container={container_name}"
        )


def can_ping_mongo(mongo_uri: str) -> bool:
    try:
        MongoClient(mongo_uri, serverSelectionTimeoutMS=3000).admin.command("ping")
    except Exception:
        return False
    return True


def has_collections(db, names: Iterable[str]) -> bool:
    return set(names).issubset(set(db.list_collection_names()))


def default_db_name(model: str, structured: bool) -> str:
    safe_model = model.replace("/", "_").replace(".", "_").replace("-", "_")
    suffix = "onto" if structured else "plain"
    return f"wikontic_{safe_model}_{suffix}_{uuid.uuid4().hex[:8]}"


def run_extraction(rows: list[dict[str, Any]], inference, structured: bool) -> None:
    total = len(rows)
    for index, row in enumerate(rows):
        item_id = str(row.get("id") or f"item_{index:06d}")
        text = str(row["raw_chunk"])
        print(f"Processing {index + 1}/{total}: {item_id}", flush=True)
        if structured:
            inference.extract_triplets_with_ontology_filtering_and_add_to_db(
                text,
                sample_id=item_id,
                source_text_id=0,
            )
        print(f"Finished {index + 1}/{total}: {item_id}", flush=True)


def clear_previous_sample_outputs(triplets_db, rows: list[dict[str, Any]]) -> None:
    sample_ids = [str(row.get("id") or f"item_{index:06d}") for index, row in enumerate(rows)]
    for collection_name in (
        "entity_aliases",
        "triplets",
        "initial_triplets",
        "filtered_triplets",
        "ontology_filtered_triplets",
    ):
        if collection_name in triplets_db.list_collection_names():
            triplets_db.get_collection(collection_name).delete_many({"sample_id": {"$in": sample_ids}})
        else:
            inference.extract_triplets_and_add_to_db(
                text,
                sample_id=item_id,
                source_text_id=0,
            )


def load_triplets_from_mongo(triplets_db) -> dict[str, list[Triple]]:
    triples_by_id: dict[str, list[Triple]] = {}
    for doc in triplets_db.get_collection("triplets").find({}):
        sample_id = str(doc.get("sample_id", ""))
        triple = (doc.get("subject"), doc.get("relation"), doc.get("object"))
        if not sample_id or not all(isinstance(value, str) and value.strip() for value in triple):
            continue
        triples_by_id.setdefault(sample_id, []).append(tuple(value.strip() for value in triple))
    return {item_id: dedupe_triples(triples) for item_id, triples in triples_by_id.items()}


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
