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



class GeoKGSpatiotemporalEventPipeline(PipelineABC):
    """Geospatial pipeline for extracting and refining spatio-temporal events.

    Required dataset columns:
    - `raw_chunk`: source text describing geographic events
    """

    def __init__(
        self,
        lang: str = "en",
        query_time_start: str = "Q1 2021",
        query_time_end: str = "2023-12-31",
        location_name: str = "China",
        rationale_min_score: float = 0.95,
        consistency_min_score: float = 0.95,
    ):
        super().__init__()

        self.storage = FileStorage(
            first_entry_file_name="../example_data/GeoKGSpatiotemporalEventPipeline/input.json",
            cache_path="./geokg",
            file_name_prefix="geokg_spatiotemporal_event_pipeline_step",
            cache_type="json",
        )
        self.query_time_start = query_time_start
        self.query_time_end = query_time_end
        self.location_name = location_name
        self.rationale_min_score = rationale_min_score
        self.consistency_min_score = consistency_min_score

        llm_serving = APILLMServing_request(
            api_url="http://123.129.219.111:3000/v1/chat/completions",
            model_name="gpt-4o",
            max_workers=20,
        )

        self.event_extraction_step1 = GeoKGEventExtraction(
            llm_serving=llm_serving,
            lang=lang,
        )
        
        self.time_filter_step2 = GeoKGEventTupleTimeFilter(
            merge_to_input=True
        )

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


if __name__ == "__main__":
    pipeline = GeoKGSpatiotemporalEventPipeline(
        query_time_start="2024-01-01",
        query_time_end="2024-12-31",
        location_name="China",
        lang="en",
    )
    pipeline.forward()
