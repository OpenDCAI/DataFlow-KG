import os

from dataflow.core import LLMServingABC
from dataflow.operators.domain_kg.financial_kg.filter.finkg_4tuple_ontology_filtering import (
    FinKGTupleFilter,
)
from dataflow.operators.domain_kg.financial_kg.generate.finkg_4tuple_extractor import (
    FinKGTupleExtraction,
)
from dataflow.operators.domain_kg.financial_kg.refine.finkg_entity_risk_assessment import (
    FinKGEntityRiskAssessment,
)
from dataflow.operators.domain_kg.utils.finkg_get_ontology import load_finkg_ontology
from dataflow.pipeline import PipelineABC
from dataflow.serving import APILLMServing_request
from dataflow.utils.storage import FileStorage


class FinKGRiskPipeline(PipelineABC):
    """Financial KG pipeline: raw text -> tuples -> filtered tuples -> risk answer.

    Required dataset columns:
    - `raw_chunk`: source financial text
    - `target_entity`: entity whose risk should be assessed
    """

    def __init__(
        self,
        lang: str = "en",
        target_ontology: str = "Corporation",
    ):
        super().__init__()

        self.storage = FileStorage(
            first_entry_file_name="../example_data/FinKGRiskPipeline/input.json",
            cache_path="./finkg",
            file_name_prefix="finkg_risk_pipeline_step",
            cache_type="json",
        )

        llm_serving = APILLMServing_request(
            api_url="http://123.129.219.111:3000/v1/chat/completions",
            model_name="gpt-4o",
            max_workers=20
        )

        self.ontology = load_finkg_ontology()
        self.target_ontology = target_ontology

        self.tuple_extraction_step1 = FinKGTupleExtraction(
            llm_serving=llm_serving,
            triple_type="relation",
            lang=lang,
        )
        self.tuple_filter_step2 = FinKGTupleFilter()

        self.risk_answer_step3 = FinKGEntityRiskAssessment(
            llm_serving=llm_serving,
            lang=lang
        )

    def forward(self):
        self.tuple_extraction_step1.run(
            storage=self.storage.step(),
            ontology_lists=self.ontology,
            input_key="raw_chunk",
            input_key_meta=None,
            output_key="tuple",
            output_key_meta="entity_class",
        )
        self.tuple_filter_step2.run(
            storage=self.storage.step(),
            ontology_lists=self.ontology,
            input_key_tuple="tuple",
            input_key_class="entity_class",
            output_key="tuple",
            input_key_meta=None,
            target_ontology=self.target_ontology,
        )
        self.risk_answer_step3.run(
            storage=self.storage.step(),
            input_key="tuple",
            output_key="risk_answer",
            output_key_score="risk_score",
        )


if __name__ == "__main__":
    pipeline = FinKGRiskPipeline(
        lang="en",
        target_ontology="Corporation",
    )
    pipeline.forward()
