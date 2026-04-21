import os

from dataflow.core import LLMServingABC
from dataflow.operators.domain_kg.geospatial_kg import (
    GeoKGEventConsistenceEvaluator,
    GeoKGEventConsistenceFilter,
    GeoKGEventExtraction,
    GeoKGEventRationaleEvaluator,
    GeoKGEventRationaleFilter,
    GeoKGEventTupleLocationFilter,
    GeoKGEventTupleTimeFilter,
)
from dataflow.pipeline import PipelineABC
from dataflow.serving import APILLMServing_request
from dataflow.utils.storage import FileStorage


def _default_example_file(folder: str, file_name: str) -> str:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.abspath(os.path.join(script_dir, "..", "example_data", folder, file_name)),
        os.path.abspath(os.path.join(script_dir, "..", "..", "..", "example", folder, file_name)),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return candidates[0]


class GeoKGSpatiotemporalEventPipeline(PipelineABC):
    """Geospatial pipeline for extracting and refining spatio-temporal events.

    Required dataset columns:
    - `raw_chunk`: source text describing geographic events
    """

    def __init__(
        self,
        first_entry_file_name: str,
        llm_serving: LLMServingABC,
        cache_path: str = "./cache_local",
        file_name_prefix: str = "geokg_spatiotemporal_event_pipeline_step",
        cache_type: str = "jsonl",
        lang: str = "en",
        query_time_start: str = "Q1 2021",
        query_time_end: str = "2023-12-31",
        location_name: str = "China",
        rationale_min_score: float = 0.95,
        consistency_min_score: float = 0.95,
    ):
        super().__init__()
        if llm_serving is None:
            raise ValueError(
                "llm_serving is required for GeoKGSpatiotemporalEventPipeline"
            )

        self.storage = FileStorage(
            first_entry_file_name=first_entry_file_name,
            cache_path=cache_path,
            file_name_prefix=file_name_prefix,
            cache_type=cache_type,
        )
        self.query_time_start = query_time_start
        self.query_time_end = query_time_end
        self.location_name = location_name
        self.rationale_min_score = rationale_min_score
        self.consistency_min_score = consistency_min_score

        self.event_extraction_step1 = GeoKGEventExtraction(
            llm_serving=llm_serving,
            lang=lang,
        )
        self.time_filter_step2 = GeoKGEventTupleTimeFilter(merge_to_input=True)
        self.location_filter_step3 = GeoKGEventTupleLocationFilter(
            merge_to_input=True
        )
        self.rationale_eval_step4 = GeoKGEventRationaleEvaluator(
            llm_serving=llm_serving,
            lang=lang,
        )
        self.rationale_filter_step5 = GeoKGEventRationaleFilter(
            merge_to_input=True
        )
        self.consistency_eval_step6 = GeoKGEventConsistenceEvaluator(
            llm_serving=llm_serving,
            lang=lang,
        )
        self.consistency_filter_step7 = GeoKGEventConsistenceFilter(
            merge_to_input=True
        )

    def forward(self):
        self.event_extraction_step1.run(
            storage=self.storage.step(),
            input_key="raw_chunk",
            output_key="tuple",
        )
        self.time_filter_step2.run(
            storage=self.storage.step(),
            input_key="tuple",
            output_key="tuple",
            query_time_start=self.query_time_start,
            query_time_end=self.query_time_end,
        )
        self.location_filter_step3.run(
            storage=self.storage.step(),
            input_key="tuple",
            output_key="tuple",
            location_name=self.location_name,
        )
        self.rationale_eval_step4.run(
            storage=self.storage.step(),
            input_key="tuple",
            output_key="rationale_scores",
        )
        self.rationale_filter_step5.run(
            storage=self.storage.step(),
            input_key="tuple",
            score_key="rationale_scores",
            output_key="tuple",
            min_score=self.rationale_min_score,
        )
        self.consistency_eval_step6.run(
            storage=self.storage.step(),
            input_key="tuple",
            output_key="consistency_scores",
        )
        self.consistency_filter_step7.run(
            storage=self.storage.step(),
            input_key="tuple",
            score_key="consistency_scores",
            output_key="tuple",
            min_score=self.consistency_min_score,
        )

    def _compiled_forward(self, resume_step: int = 0):
        for idx, op_node in enumerate(self.op_nodes_list):
            if idx - 1 < resume_step:
                continue
            if op_node.op_obj is None:
                continue

            if op_node.llm_serving is not None:
                if (
                    self.active_llm_serving
                    and self.active_llm_serving is not op_node.llm_serving
                ):
                    self.active_llm_serving.cleanup()
                self.active_llm_serving = op_node.llm_serving

            op_node.op_obj.run(storage=op_node.storage, **op_node.kwargs)

            if op_node.llm_serving is not None:
                self.llm_serving_counter[self.active_llm_serving] -= 1
                if self.llm_serving_counter[self.active_llm_serving] == 0:
                    self.active_llm_serving.cleanup()
                    self.active_llm_serving = None


if __name__ == "__main__":
    input_file = os.environ.get(
        "DF_GEOKG_INPUT_FILE",
        _default_example_file("GeoKGSpatiotemporalEventPipeline", "input.json"),
    )

    llm_serving = APILLMServing_request(
        api_url=os.environ.get(
            "DF_API_URL",
            "https://api.openai.com/v1/chat/completions",
        ),
        key_name_of_api_key="DF_API_KEY",
        model_name=os.environ.get("DF_LLM_MODEL", "gpt-4o-mini"),
        max_workers=8,
        temperature=0.0,
    )

    pipeline = GeoKGSpatiotemporalEventPipeline(
        first_entry_file_name=input_file,
        llm_serving=llm_serving,
        cache_path="./cache_geokg_event",
        query_time_start="2024-01-01",
        query_time_end="2024-12-31",
        location_name="China",
        lang="en",
    )
    pipeline.forward()
