"""
Test: FinKGTableTupleExtraction

Usage:
    cd DataFlow-KG
    conda activate dataflow-kg
    export DEEPSEEK_API_KEY=your_key
    python test/test_finkg_table_tuple_extractor.py

This test targets a single operator only:
    FinKGTableTupleExtraction
"""

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dataflow.serving.api_llm_serving_request import APILLMServing_request
from dataflow.utils.storage import FileStorage
from dataflow.operators.financial_kg import FinKGTableTupleExtraction
from dataflow.operators.financial_kg.generate.finkg_get_ontology import (
    FinKGGetBasicOntology,
)
from dataflow.operators.financial_kg.generate.finkg_table_tuple_extractor import (
    FinKGTableTupleExtractionLLM,
)


API_URL = os.environ.get(
    "DEEPSEEK_API_URL",
    "https://api.deepseek.com/chat/completions",
)
MODEL_NAME = os.environ.get("DEEPSEEK_MODEL_NAME", "deepseek-chat")
INPUT_FILE = "./dataflow/data_for_operator_testing/finkg_table2kg.json"
CACHE_DIR = "./.cache/test_finkg_table2kg"


def build_ontology_dict():
    ontology_op = FinKGGetBasicOntology()
    return {
        "entity_type": ontology_op.load_entity_types(),
        "relation_type": ontology_op.load_relation_types(),
        "attribute_type": ontology_op.load_attribute_types(),
    }


def test_table_normalization_only():
    helper = FinKGTableTupleExtractionLLM(llm_serving=None, lang="en")

    markdown_table = "| a | b |\n| --- | --- |\n| 1 | 2 |"
    assert helper._normalize_table_input(markdown_table) == markdown_table

    json_records = [{"entity": "Bank A", "city": "Charlotte"}]
    normalized = helper._normalize_table_input(json_records)
    assert "Bank A" in normalized
    assert "Charlotte" in normalized

    print("  Table normalization: OK")


def run_operator_test():
    storage = FileStorage(
        first_entry_file_name=INPUT_FILE,
        cache_path=CACHE_DIR,
        file_name_prefix="step",
        cache_type="json",
    )

    llm_serving = APILLMServing_request(
        api_url=API_URL,
        key_name_of_api_key="DEEPSEEK_API_KEY",
        model_name=MODEL_NAME,
        max_workers=5,
    )

    extractor = FinKGTableTupleExtraction(
        llm_serving=llm_serving,
        lang="en",
    )

    output_keys = extractor.run(
        storage=storage.step(),
        ontology_lists=build_ontology_dict(),
        input_key="raw_table",
        input_title_key="table_title",
        input_context_key="table_context",
        output_key="tuple",
        output_key_meta="entity_class",
        output_schema_key="table_schema",
    )

    print(f"Output keys: {output_keys}")

    step_path = os.path.join(CACHE_DIR, "step_step1.json")
    with open(step_path, "r", encoding="utf-8") as file:
        results = json.load(file)

    for idx, row in enumerate(results):
        tuples = row.get("tuple", [])
        classes = row.get("entity_class", [])
        schema = row.get("table_schema", {})

        print("\n" + "=" * 60)
        print(f"Row {idx}")
        print("=" * 60)
        print("table_type:", schema.get("table_type", "NA"))
        print("candidate_relations:", schema.get("candidate_relations", []))
        print("candidate_attributes:", schema.get("candidate_attributes", []))
        print("tuple_count:", len(tuples))

        for tuple_item, class_item in zip(tuples, classes):
            print(" ", tuple_item)
            print("   class=", class_item)

    print("\nResults saved to:", step_path)


if __name__ == "__main__":
    print("=" * 60)
    print("Offline helper test")
    print("=" * 60)
    test_table_normalization_only()

    print("\n" + "=" * 60)
    print("Single-operator table-to-KG test")
    print("=" * 60)
    run_operator_test()
