"""
Test: FinKGRelationChainInference (LLM-based multi-hop inference)

Usage:
    cd DataFlow-KG
    conda activate hab
    python test/test_finkg_relation_chain_inference.py

Test data covers 4 real-world financial scenarios:
  Row 0: Berkshire Hathaway ownership chain (EN)
  Row 1: 海航集团 guarantee chain + related-party (ZH)
  Row 2: Goldman Sachs / 1MDB guarantee-default chain (EN)
  Row 3: JPMorgan Chase corporate structure (EN)
"""

import sys
import os
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dataflow.utils.storage import FileStorage
from dataflow.serving import APILLMServing_request
from dataflow.operators.financial_kg.refine.finkg_relation_chain_inference import (
    FinKGRelationChainInference,
    FinKGRelationChainInferenceEngine,
)
from dataflow.operators.financial_kg.generate.finkg_get_ontology import (
    FinKGGetBasicOntology,
)


# ======================================================
# Config
# ======================================================
API_URL = "http://123.129.219.111:3000/v1/chat/completions"
MODEL_NAME = "gpt-4o"
INPUT_FILE = "./dataflow/data_for_operator_testing/finkg_inference.json"
CACHE_DIR = "./.cache/test_finkg_inference"
ONTOLOGY_CACHE = "./.cache/api/finkg_ontology.json"


# ======================================================
# Offline test: k-hop BFS (no API needed)
# ======================================================
def test_khop_bfs():
    print("=" * 60)
    print("Offline test: k-hop BFS neighborhood selection")
    print("=" * 60)

    engine = FinKGRelationChainInferenceEngine(k_hops=2)

    # Berkshire ownership chain from test data row 0
    tuples = [
        "<subj> Berkshire Hathaway <obj> National Indemnity <rel> owns <time> 1967",
        "<subj> Berkshire Hathaway <obj> GEICO <rel> owns <time> 1996",
        "<subj> Berkshire Hathaway <obj> General Re <rel> owns <time> 1998",
        "<subj> Warren Buffett <obj> Berkshire Hathaway <rel> controls <time> 2024",
        "<subj> GEICO <obj> InsuranceUnderwriterRole <rel> plays_role <time> 2024",
        "<subj> National Indemnity <obj> Berkshire Hathaway <rel> subsidiary_of <time> 1967",
        "<subj> Unrelated Corp <obj> Random Inc <rel> owns <time> 2020",
    ]

    # k=2: from {Warren Buffett, General Re}
    related = engine._find_related_tuples(tuples, "Warren Buffett", "General Re")
    print(f"\n  k=2 hops from (Warren Buffett, General Re):")
    for t in related:
        print(f"    {t}")
    assert not any("Unrelated" in t for t in related), "Unrelated tuple should be excluded"
    assert len(related) >= 2, "Should find chain through Berkshire"
    print(f"  OK: found {len(related)} related tuples, unrelated excluded")

    # k=1: fewer tuples
    engine_1 = FinKGRelationChainInferenceEngine(k_hops=1)
    related_1 = engine_1._find_related_tuples(tuples, "Warren Buffett", "General Re")
    print(f"\n  k=1 hops from (Warren Buffett, General Re):")
    for t in related_1:
        print(f"    {t}")
    assert len(related_1) <= len(related), "k=1 should find <= tuples than k=2"
    print(f"  OK: k=1 found {len(related_1)} tuples <= k=2's {len(related)}")

    # Chinese data: guarantee chain from test data row 1
    zh_tuples = [
        "<subj> 海航集团 <obj> 海航航空集团 <rel> controls <time> 2021",
        "<subj> 海航集团 <obj> 大新华航空 <rel> guarantor_of <time> 2021",
        "<subj> 大新华航空 <obj> 天津航空 <rel> guarantor_of <time> 2021",
        "<subj> 大新华航空 <obj> 海航集团 <rel> subsidiary_of <time> 2021",
        "<subj> 天津航空 <obj> 海航集团 <rel> subsidiary_of <time> 2021",
    ]
    zh_related = engine._find_related_tuples(zh_tuples, "海航集团", "天津航空")
    print(f"\n  k=2 hops from (海航集团, 天津航空):")
    for t in zh_related:
        print(f"    {t}")
    assert len(zh_related) >= 3, "Should find guarantee chain path"
    print(f"  OK: found {len(zh_related)} related tuples for Chinese data")

    # Edge cases
    assert engine._find_related_tuples([], "A", "B") == []
    assert engine._find_related_tuples(None, "A", "B") == []
    assert engine._find_related_tuples(["invalid string"], "A", "B") == []
    print("\n  OK: edge cases handled")

    # LLM response parsing
    good = '{"tuple": ["<subj> A <obj> C <rel> controls <time> 2024"], "entity_class": [["Corporation", "Corporation"]]}'
    result = engine._parse_llm_response(good)
    assert result["tuple"] == ["<subj> A <obj> C <rel> controls <time> 2024"]
    assert result["entity_class"] == [["Corporation", "Corporation"]]
    print("  OK: JSON parsing")

    bad = "not json"
    assert engine._parse_llm_response(bad)["tuple"] == []
    print("  OK: malformed response")

    md = '```json\n{"tuple": ["x"], "entity_class": []}\n```'
    assert engine._parse_llm_response(md)["tuple"] == ["x"]
    print("  OK: markdown-wrapped JSON")

    print("\nAll offline tests passed!\n")


# ======================================================
# Prepare ontology cache
# ======================================================
def prepare_ontology_cache():
    os.makedirs(os.path.dirname(ONTOLOGY_CACHE), exist_ok=True)
    ontology_op = FinKGGetBasicOntology()
    storage = FileStorage(
        first_entry_file_name="",
        cache_path=os.path.dirname(ONTOLOGY_CACHE),
        file_name_prefix="finkg_ontology",
        cache_type="json",
    )
    ontology_op.run(storage=storage.step())
    print(f"Ontology cached to {ONTOLOGY_CACHE}")


# ======================================================
# Online test: Full LLM inference (4 scenarios)
# ======================================================
def test_full_inference():
    print("=" * 60)
    print("Online test: Full LLM-based relation chain inference")
    print("=" * 60)

    if not os.path.exists(ONTOLOGY_CACHE):
        prepare_ontology_cache()

    llm_serving = APILLMServing_request(
        api_url=API_URL,
        model_name=MODEL_NAME,
        max_workers=5,
    )

    test_cases = [
        {
            "name": "Ownership penetration (Warren Buffett → General Re)",
            "entity_pair": ["Warren Buffett", "General Re"],
            "k_hops": 2,
        },
        {
            "name": "Guarantee chain (海航集团 → 天津航空)",
            "entity_pair": ["海航集团", "天津航空"],
            "k_hops": 2,
        },
        {
            "name": "Guarantee-default (IPIC → 1MDB)",
            "entity_pair": ["IPIC", "Goldman Sachs"],
            "k_hops": 2,
        },
        {
            "name": "Corporate structure (Federal Reserve → J.P. Morgan Securities LLC)",
            "entity_pair": ["Federal Reserve", "J.P. Morgan Securities LLC"],
            "k_hops": 2,
        },
    ]

    for idx, case in enumerate(test_cases):
        print(f"\n--- Case {idx + 1}: {case['name']} ---")

        cache_path = f"{CACHE_DIR}_case{idx + 1}"

        storage = FileStorage(
            first_entry_file_name=INPUT_FILE,
            cache_path=cache_path,
            file_name_prefix="step",
            cache_type="json",
        )

        op = FinKGRelationChainInference(
            llm_serving=llm_serving,
            lang="en",
        )

        output_keys = op.run(
            storage=storage.step(),
            entity_pair=case["entity_pair"],
            input_key_tuple="tuple",
            input_key_meta="finkg_ontology",
            output_key="inferred_tuple",
            evidence_key="entity_class",
            k_hops=case["k_hops"],
        )
        print(f"  Output keys: {output_keys}")

        step_path = os.path.join(cache_path, "step_step1.json")
        with open(step_path, "r", encoding="utf-8") as f:
            results = json.load(f)

        for i, row in enumerate(results):
            inferred = row.get("inferred_tuple", [])
            classes = row.get("entity_class", [])
            if inferred:
                print(f"  Row {i}: {len(inferred)} inferred tuple(s)")
                for j, t in enumerate(inferred):
                    c = classes[j] if j < len(classes) else "N/A"
                    print(f"    {t}  class={c}")
            else:
                print(f"  Row {i}: no inference (entities not in this row)")

    print("\n" + "=" * 60)
    print("All inference tests completed!")
    print("=" * 60)


if __name__ == "__main__":
    test_khop_bfs()
    test_full_inference()
